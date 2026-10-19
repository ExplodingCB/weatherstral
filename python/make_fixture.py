"""Write a test fixture for the browser: a real context and PyTorch's reference forecast for it."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

from backtest import rain_prob
from forecast import CONTEXT, VARS, fetch
from timestral import TimestralForecaster

model = sys.argv[1] if len(sys.argv) > 1 else "mistralai/Timestral-28M"
out = sys.argv[2] if len(sys.argv) > 2 else "../web/test/fixture.json"
Path(out).parent.mkdir(parents=True, exist_ok=True)
d = fetch(39.7684, -86.1581)
s = np.array([d["hourly"][v] for v in VARS], dtype=np.float32)
now = next(i for i, t in enumerate(d["hourly"]["time"]) if t.startswith(d["current"]["time"][:13]))
x = s[:, now + 1 - CONTEXT: now + 1]
fc = TimestralForecaster.from_pretrained(model)
with torch.inference_mode():
    q = fc.forecast(x[None], 32).quantiles.numpy()  # (1, V, 32, Q), the real forecast() path
rain = rain_prob(q[0, VARS.index("precipitation")], fc.quantile_levels)  # (32,) P(>=0.1 mm/h)
json.dump({"context": x.tolist(), "quantiles": q.reshape(-1).tolist(), "shape": list(q.shape),
           "levels": list(fc.quantile_levels), "rain": rain.tolist()}, open(out, "w"))
print("wrote", out, q.shape, "median temp +1..5h:", q[0, 0, :5, 10].round(2))
