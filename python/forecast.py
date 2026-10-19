"""Timestral 5-hour weather forecast vs Open-Meteo's own forecast, plus a quick backtest.

Usage:
    python forecast.py [--lat 39.77] [--lon -86.16] [--model mistralai/Timestral-28M] [--json]

--json prints only a JSON document (used by the macOS app); otherwise a readable table.
"""
import argparse
import json
import sys

import numpy as np
import requests
import torch
from timestral import TimestralForecaster

HORIZON = 5
CONTEXT = 168  # hours of history fed to the model (7 days)
VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure", "cloud_cover", "wind_speed_10m", "precipitation"]
URL = "https://api.open-meteo.com/v1/forecast"


def fetch(lat, lon):
    r = requests.get(URL, params={
        "latitude": lat, "longitude": lon,
        "hourly": ",".join(VARS),
        "current": ",".join(VARS),
        "past_days": 14, "forecast_days": 2,
        "timezone": "auto",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
    }, timeout=20)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=39.7684)
    ap.add_argument("--lon", type=float, default=-86.1581)
    ap.add_argument("--model", default="mistralai/Timestral-28M")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    data = fetch(args.lat, args.lon)
    times = data["hourly"]["time"]
    now = data["current"]["time"][:13]  # "YYYY-MM-DDTHH"
    now_idx = next(i for i, t in enumerate(times) if t.startswith(now))
    series = np.array([data["hourly"][v] for v in VARS], dtype=np.float32)  # (V, T)

    model = TimestralForecaster.from_pretrained(args.model, device="cpu")

    def run(ctx_batch):
        with torch.no_grad():
            return model.forecast(ctx_batch, HORIZON)

    # ---- live forecast: all variables jointly, report temperature -------
    ctx = series[:, now_idx + 1 - CONTEXT: now_idx + 1]  # (V, CONTEXT), ends at the current hour
    multi = run(ctx[None])
    med = multi.median[0, 0].numpy()
    lo, hi = multi.quantile(0.1)[0, 0].numpy(), multi.quantile(0.9)[0, 0].numpy()
    om = series[0, now_idx + 1: now_idx + 1 + HORIZON]
    precip = data["hourly"]["precipitation"][now_idx + 1: now_idx + 1 + HORIZON]
    hours = [
        {"time": times[now_idx + 1 + h], "temp": float(med[h]), "low": float(lo[h]),
         "high": float(hi[h]), "openMeteoTemp": float(om[h]), "precipMM": float(precip[h])}
        for h in range(HORIZON)
    ]
    cur = data["current"]
    current = {"time": cur["time"], "temp": cur["temperature_2m"], "humidity": cur["relative_humidity_2m"],
               "pressure": cur["surface_pressure"], "cloudCover": cur["cloud_cover"],
               "windMPH": cur["wind_speed_10m"]}

    if args.json:
        json.dump({"lat": args.lat, "lon": args.lon, "model": args.model,
                   "current": current, "hours": hours}, sys.stdout)
        return

    print(f"Location {args.lat:.2f},{args.lon:.2f}   now = {cur['time']}   temp now = {cur['temperature_2m']:.1f}F\n")
    print(f"{'hour':<7}{'Timestral':>10}{'(10-90%)':>12}{'Open-Meteo':>12}{'precip mm':>11}")
    for h, row in enumerate(hours):
        print(f"{row['time'][11:16]:<7}{row['temp']:>9.1f}F{f'{lo[h]:.0f}-{hi[h]:.0f}':>12}"
              f"{row['openMeteoTemp']:>11.1f}F{row['precipMM']:>11.1f}")

    # ---- backtest on recent past ---------------------------------------
    # Cutoffs every 3h over the last 7 days. "Truth" is Open-Meteo's hourly history
    # (model analysis, not weather-station readings), so treat this as a rough check.
    # Baselines: persistence (same as now) and "yesterday's change" (now + the change that
    # happened over the same hours 24h ago), which already knows about the daily cycle.
    cutoffs = list(range(now_idx - 7 * 24, now_idx - HORIZON, 3))
    ctxs = np.stack([series[:, c + 1 - CONTEXT: c + 1] for c in cutoffs])
    truth = np.stack([series[0, c + 1: c + 1 + HORIZON] for c in cutoffs])
    persist = np.stack([np.full(HORIZON, series[0, c]) for c in cutoffs])
    yday = np.stack([series[0, c] + series[0, c - 23: c - 23 + HORIZON] - series[0, c - 24] for c in cutoffs])
    m = run(ctxs).median[:, 0].numpy()

    print(f"\nBacktest: {len(cutoffs)} forecasts over the last 7 days, temperature MAE (F) by lead hour")
    print(f"{'lead':<6}{'persistence':>13}{'yesterday':>11}{'Timestral':>11}")
    for h in range(HORIZON):
        print(f"+{h+1}h   {np.abs(persist[:, h]-truth[:, h]).mean():>12.2f}"
              f"{np.abs(yday[:, h]-truth[:, h]).mean():>11.2f}{np.abs(m[:, h]-truth[:, h]).mean():>11.2f}")
    print(f"avg   {np.abs(persist-truth).mean():>12.2f}{np.abs(yday-truth).mean():>11.2f}{np.abs(m-truth).mean():>11.2f}")


if __name__ == "__main__":
    main()
