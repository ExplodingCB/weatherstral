"""Export Timestral (fixed shape: 1 location x 5 variates x 168 hours -> 32 hours) to ONNX.

The exported graph includes everything forecast() does for this shape: causal scaling,
the transformer, the quantile head, the mirrored (negated) pass averaging, and the
non-negative clamp. The browser only has to supply the 5x168 history and read the quantiles.

    python export_onnx.py --model mistralai/Timestral-28M --out ../web/model/timestral-28m.onnx
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from forecast import CONTEXT, VARS, fetch
from timestral import TimestralForecaster
from timestral.forecaster import nonnegative_context

V = len(VARS)


class FrozenMasks:
    """Attention masks and positions depend only on the (fixed) observation pattern, so compute
    them once eagerly and hand the tracer the results as constants. This keeps ops the ONNX
    exporter can't handle (cummax, ...) out of the graph.
    """

    NAMES = ["compute_enc_key_validity", "build_forecast_time_mask", "build_two_stream_variate_mask",
             "group_first_observed_patch", "build_forecast_positions"]

    def __init__(self):
        import timestral.model as mmod
        self.mmod = mmod
        self.originals = {n: getattr(mmod, n) for n in self.NAMES}
        self.recorded = {n: [] for n in self.NAMES}
        self.calls = {n: 0 for n in self.NAMES}
        self.replay = False
        for n in self.NAMES:
            setattr(mmod, n, self._wrap(n))
        self.unique_rec, self.unique_calls = [], 0
        mmod.torch = _TorchProxy(self)

    def _wrap(self, name):
        def fn(*a, **k):
            if not self.replay:
                out = self.originals[name](*a, **k)
                self.recorded[name].append(out.detach().clone())
                return out
            i = self.calls[name]
            self.calls[name] += 1
            return self.recorded[name][i]
        return fn

    def unique(self, t, **kw):
        # torch.unique has a data-dependent output shape; the group ids are fixed, so freeze the result.
        if not self.replay:
            out = torch.unique(t, **kw)
            self.unique_rec.append(tuple(o.detach().clone() for o in out))
            return out
        out = self.unique_rec[self.unique_calls]
        self.unique_calls += 1
        return out

    def start_replay(self):
        self.replay = True
        self.calls = {n: 0 for n in self.NAMES}
        self.unique_calls = 0


class _TorchProxy:
    """Stands in for `torch` inside timestral.model, forwarding everything except unique."""

    def __init__(self, frozen):
        self._frozen = frozen

    def __getattr__(self, name):
        return self._frozen.unique if name == "unique" else getattr(torch, name)


def _patch_for_onnx():
    """Swap Tensor.unfold (not exportable) for an equivalent reshape in the per-patch scaler."""
    import timestral.forecaster as fmod
    import timestral.model as mmod
    import timestral.scaler as smod

    def causal_statistics(data, observation_mask, minimum_scale, stabilize_with_global, scale_factor_exponent=10.0):
        # Same maths as timestral.scaler.compute_causal_statistics, in float64. Welford's running
        # sum of squared deviations equals sum(x^2) - n * mean^2, which needs no shifted slice
        # (the exporter mishandles `means[..., :-1]` here).
        assert not stabilize_with_global
        w = (torch.ones_like(data) * observation_mask).to(torch.float64)
        x = data.to(torch.float64)
        cum_w = torch.cumsum(w, dim=-1).clamp_min(1.0)
        means = torch.cumsum(w * x, dim=-1) / cum_w
        m2 = torch.cumsum(w * x * x, dim=-1) - cum_w * means * means
        variance = m2.clamp_min(0.0) / torch.clamp(cum_w - 1.0, min=1.0)
        scales = torch.sqrt(variance).clamp(min=minimum_scale)
        return means.to(data.dtype), scales.to(data.dtype)

    def causal_patch_loc_scale(data, observation_mask, patch_size, minimum_scale, stabilize_with_global):
        with torch.no_grad():
            means, scales = causal_statistics(data, observation_mask, minimum_scale, stabilize_with_global)
            # unfold(-1, ps, ps)[..., -1] == last element of each consecutive patch
            patch_means = means.unflatten(-1, (-1, patch_size))[..., -1]
            patch_scales = scales.unflatten(-1, (-1, patch_size))[..., -1]
            return patch_means.repeat_interleave(patch_size, dim=-1), patch_scales.repeat_interleave(patch_size, dim=-1)

    for mod in (smod, mmod, fmod):
        mod.causal_patch_loc_scale = causal_patch_loc_scale

    def forecast_tokens(self, rows, n_patches):
        return self.forecast_query.view(1, 1, -1).repeat(rows, n_patches, 1)  # expand(-1) exports badly

    mmod.PatchRepresentation.forecast_tokens = forecast_tokens

    # The context is fully observed, so forward-filling is the identity (and `mask.all()` is data-dependent).
    fmod.ffill_with_mask = lambda inputs, mask: inputs

    # The legacy exporter has no symbolic for asinh, though ONNX has an Asinh op.
    torch.onnx.register_custom_op_symbolic("aten::asinh", lambda g, x: g.op("Asinh", x), 17)
    torch.onnx.register_custom_op_symbolic("aten::sinh", lambda g, x: g.op("Sinh", x), 17)

HORIZON_STEPS = 32  # one patch; the first few hours are what we use


class Wrapped(nn.Module):
    def __init__(self, fc: TimestralForecaster):
        super().__init__()
        self.fc = fc
        self.register_buffer("mask", torch.ones(2, V, CONTEXT, dtype=torch.bool))

    def _forecast_raw(self, inputs):
        # TimestralForecaster._forecast_raw without its data-dependent NaN check, for a fully observed context.
        fc = self.fc
        rows, mask, _, _, _, n_tgt = fc._assemble_rows(inputs, self.mask, HORIZON_STEPS, None, None)
        q = fc._block_ar_decode(rows, mask, HORIZON_STEPS, None, None, None)[:, :n_tgt]
        return fc._cap_quantiles(q, rows[:, :n_tgt], mask[:, :n_tgt])

    def forward(self, x):  # (1, V, CONTEXT) -> (1, V, 32, Q)
        both = self._forecast_raw(torch.cat([x, -x], dim=0))  # (2, V, 32, Q)
        q = 0.5 * (both[0:1] + (-both[1:2]).flip(-1))
        eligible = nonnegative_context(x, self.mask[0:1])
        return torch.where(eligible[..., None, None], q.clamp_min(0), q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Timestral-28M")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    fc = TimestralForecaster.from_pretrained(args.model)
    _patch_for_onnx()
    fc.model.eval()
    for p in fc.model.parameters():
        p.requires_grad_(False)

    d = fetch(39.7684, -86.1581)
    s = np.array([d["hourly"][v] for v in VARS], dtype=np.float32)
    now = next(i for i, t in enumerate(d["hourly"]["time"]) if t.startswith(d["current"]["time"][:13]))
    x = torch.from_numpy(s[:, now + 1 - CONTEXT: now + 1])[None].contiguous()

    frozen = FrozenMasks()
    wrapped = Wrapped(fc).eval()
    with torch.inference_mode():
        ref = fc.forecast(x, HORIZON_STEPS).quantiles
    frozen.recorded = {n: [] for n in FrozenMasks.NAMES}  # drop the calls made by forecast()
    frozen.calls = {n: 0 for n in FrozenMasks.NAMES}
    frozen.unique_rec = []
    with torch.inference_mode():
        mine = wrapped(x)  # records the masks for the two passes
    print("wrapper vs forecast():", float((ref - mine).abs().max()))
    # A second, different input to make sure nothing got baked in from the example (reference computed
    # now, before the masks are frozen for tracing).
    x2 = x * 0.8 + 3.0 * torch.sin(torch.arange(CONTEXT, dtype=torch.float32) / 5)
    with torch.inference_mode():
        ref2 = fc.forecast(x2, HORIZON_STEPS).quantiles.numpy()
    frozen.recorded = {n: [] for n in FrozenMasks.NAMES}
    frozen.calls = {n: 0 for n in FrozenMasks.NAMES}
    frozen.unique_rec = []
    with torch.inference_mode():
        wrapped(x)  # re-record the masks for the traced call
    frozen.start_replay()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(wrapped, (x,), args.out, input_names=["context"], output_names=["quantiles"],
                          opset_version=18, dynamo=True)

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"context": x.numpy()})[0]
    assert out.shape == tuple(ref.shape), (out.shape, tuple(ref.shape))
    diff = np.abs(out - ref.numpy())
    print("onnx vs pytorch, max abs diff (all quantiles):", diff.max(), "| median, first 5h:",
          np.abs(out[0, 0, :5, 10] - ref.numpy()[0, 0, :5, 10]).max())

    out2 = sess.run(None, {"context": x2.numpy()})[0]
    assert out2.shape == ref2.shape
    print("second input, max abs diff:", np.abs(out2 - ref2).max())


if __name__ == "__main__":
    main()
