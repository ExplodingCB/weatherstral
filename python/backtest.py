"""Backtest Timestral-28M on several cities: temperature error by lead hour, and rain odds.

Uses Open-Meteo's archive (hourly reanalysis), ~75 days per city, one forecast every 6 hours.
Rain "event" = at least 0.1 mm in that hour. The model's rain probability is read from its
predicted quantiles of hourly precipitation.
"""
import sys

import numpy as np
import requests
import torch

from forecast import CONTEXT, VARS
from timestral import TimestralForecaster

CITIES = {"Indianapolis": (39.77, -86.16), "Seattle": (47.61, -122.33), "Miami": (25.76, -80.19),
          "Denver": (39.74, -104.99), "Houston": (29.76, -95.37), "Boston": (42.36, -71.06)}
H = 24
THRESH = 0.1  # mm/h


def rain_prob(qs, levels, t=THRESH):
    """P(X > t) from sorted quantile values qs (..., Q) at probability `levels`."""
    levels = np.asarray(levels)
    v = np.asarray(qs)
    below = (v <= t).sum(-1)  # number of quantile values <= t
    i = np.clip(below - 1, 0, len(levels) - 2)
    lo = np.take_along_axis(v, i[..., None], -1)[..., 0]
    hi = np.take_along_axis(v, (i + 1)[..., None], -1)[..., 0]
    frac = np.where(hi > lo, (t - lo) / np.maximum(hi - lo, 1e-9), 0.0)
    cdf = levels[i] + (levels[i + 1] - levels[i]) * np.clip(frac, 0, 1)
    cdf = np.where(below == 0, 0.0, np.where(below >= len(levels), 1.0, cdf))
    # Beyond the outermost quantiles we only know "<1%" / ">99%".
    return np.clip(1 - cdf, 0.005, 0.995)


def get_city(lat, lon):
    import datetime as dt
    end = dt.date.today() - dt.timedelta(days=6)
    start = end - dt.timedelta(days=75 + 7)
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(VARS), "timezone": "auto", "temperature_unit": "fahrenheit", "wind_speed_unit": "mph"}, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    return np.array([[np.nan if x is None else x for x in h[v]] for v in VARS], dtype=np.float32)


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Timestral-28M"
    fc = TimestralForecaster.from_pretrained(model)
    lv = fc.quantile_levels
    mae = {k: np.zeros(H) for k in ("persist", "yday", "model")}
    n_fc = 0
    probs, events, base, persist_ev, leads = [], [], [], [], []
    for name, (lat, lon) in CITIES.items():
        s = get_city(lat, lon)
        s = s[:, ~np.isnan(s).any(0)] if np.isnan(s).any() else s
        cuts = list(range(CONTEXT, s.shape[1] - H, 6))
        ctx = np.stack([s[:, c - CONTEXT:c] for c in cuts])         # context ends at c-1 ("now")
        out = []
        for i in range(0, len(cuts), 32):
            with torch.inference_mode():
                out.append(fc.forecast(ctx[i:i + 32], H).quantiles.numpy())
        q = np.concatenate(out)                                      # (B, V, H, Q)
        truth = np.stack([s[:, c:c + H] for c in cuts])              # (B, V, H)
        now = ctx[:, :, -1]
        med = q[:, 0, :, 10]
        tt = truth[:, 0]
        persist = np.repeat(now[:, :1], H, 1)
        # "yesterday's change": now + how much the temperature moved over the same hours 24h earlier
        # (context[-1] is "now", so context[-25] is 24h earlier and lead k maps to context[-25 + k]).
        base24 = ctx[:, 0, -25]
        ydelta = np.stack([ctx[:, 0, -25 + k] - base24 for k in range(1, H + 1)], 1)
        yday = now[:, :1] + ydelta
        mae["persist"] += np.abs(persist - tt).sum(0)
        mae["yday"] += np.abs(yday - tt).sum(0)
        mae["model"] += np.abs(med - tt).sum(0)
        n_fc += len(cuts)
        p = rain_prob(q[:, 5], lv)                                   # variate 5 = precipitation
        ev = (truth[:, 5] >= THRESH).astype(np.float32)
        probs.append(p); events.append(ev)
        base.append(np.repeat((ctx[:, 5] >= THRESH).mean(-1, keepdims=True), H, 1))
        persist_ev.append(np.repeat((ctx[:, 5, -1:] >= THRESH).astype(np.float32), H, 1))
        print(f"{name}: {len(cuts)} forecasts, rain hours {int((truth[:, 5] >= THRESH).sum())}/{truth[:, 5].size}", flush=True)

    print(f"\nTemperature MAE (F) over {n_fc} forecasts")
    print("lead   persist  yesterday  Timestral")
    for h in [0, 1, 2, 4, 7, 11, 17, 23]:
        print(f"+{h+1:>2}h  {mae['persist'][h]/n_fc:>7.2f}  {mae['yday'][h]/n_fc:>9.2f}  {mae['model'][h]/n_fc:>9.2f}")
    print(f"avg    {mae['persist'].mean()/n_fc:>7.2f}  {mae['yday'].mean()/n_fc:>9.2f}  {mae['model'].mean()/n_fc:>9.2f}")

    P, E, B, PE = (np.concatenate(x) for x in (probs, events, base, persist_ev))
    print("\nRain (>=0.1 mm in the hour): Brier score, lower is better. Skill = 1 - Brier/Brier(recent-frequency baseline)")
    print("leads      rain-rate  Timestral  baseline  persist  skill   hit@50%  false-alarm@50%")
    for a, b in [(0, 3), (3, 6), (6, 12), (12, 24)]:
        p, e, bb, pe = P[:, a:b].ravel(), E[:, a:b].ravel(), B[:, a:b].ravel(), PE[:, a:b].ravel()
        br, bb_s, pe_s = ((p - e) ** 2).mean(), ((bb - e) ** 2).mean(), ((pe - e) ** 2).mean()
        pos = p >= 0.5
        hit = (pos & (e == 1)).sum() / max((e == 1).sum(), 1)
        fa = (pos & (e == 0)).sum() / max(pos.sum(), 1)
        print(f"+{a+1:>2}..{b:>2}h   {e.mean():>8.3f}  {br:>9.4f}  {bb_s:>8.4f}  {pe_s:>7.4f}  {1-br/bb_s:>5.2f}   {hit:>6.2f}   {fa:>6.2f}")
    print("\nReliability (all leads): predicted probability bin -> observed frequency (count)")
    p, e = P.ravel(), E.ravel()
    for lo, hi in [(0, .02), (.02, .1), (.1, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        m = (p >= lo) & (p < hi)
        if m.sum():
            print(f"  {lo:.2f}-{min(hi,1):.2f}: predicted {p[m].mean():.2f}  observed {e[m].mean():.2f}  (n={int(m.sum())})")


if __name__ == "__main__":
    main()
