# Weatherstral

AI weather that runs in your browser. Search any city and get the next 24 hours of temperature, rain chances, and sky conditions, predicted by [Timestral](https://huggingface.co/mistralai/Timestral-28M), a small time-series model from Mistral AI, running locally in the page with ONNX Runtime Web.

**Live:** https://weather.explodingcb.com (also in the [ChApp Store](https://explodingcb.com/apps/weatherstral/))

Timestral is a general-purpose forecaster, not a weather model. Weatherstral gives it the last 7 days of hourly readings for a place and lets it extrapolate. It is an experiment, and Open-Meteo's own forecast is shown next to it for comparison (it is the better bet).

## How it works

1. The page fetches 7+ days of hourly history from [Open-Meteo](https://open-meteo.com): temperature, humidity, pressure, cloud cover, wind, and precipitation.
2. The last 168 hours of those six series go into Timestral-28M (exported to ONNX, about 112 MB, run with the WASM backend, roughly 0.1 to 0.3 seconds per forecast). The six series are forecast jointly.
3. The model returns 21 quantiles for each of the next hours. The page shows the median temperature with a 10 to 90 percent band, the predicted cloud cover, and a rain probability read off the precipitation quantiles (the chance of at least 0.1 mm in the hour).
4. Open-Meteo's forecast for the same hours is fetched only to display next to it. It is never fed to the model.

Nothing is sent to a server of mine: the model runs in your browser, and the only network calls are to Open-Meteo (data and city search).

## How good is it

Backtest on Open-Meteo archive (reanalysis) data for six U.S. cities, about 75 days each, one forecast every 6 hours (1,800 forecasts). Temperature mean absolute error in °F:

| Lead | "Same as now" | "Same change as yesterday" | Timestral-28M |
|---|---|---|---|
| +1h | 1.44 | 1.03 | 0.70 |
| +3h | 3.83 | 2.22 | 1.58 |
| +5h | 6.05 | 2.93 | 2.03 |
| +12h | 9.29 | 4.16 | 2.76 |
| +24h | 3.32 | 5.17 | 3.11 |

Rain odds are well calibrated (when it says 60 percent, it rains about 52 percent of the time) but only modestly more skillful than the recent rain frequency: Brier skill 0.28 for hours 1 to 3, 0.16 for 4 to 6, 0.11 for 7 to 12, and 0.08 for 13 to 24. In practice it mostly reflects whether it has been wet lately, so it will not see a new storm coming after a dry spell.

Caveats: the "truth" in this test is reanalysis data, not weather-station readings, it covers six cities over one stretch of fall, and the history the live app uses is also Open-Meteo's analysis, so the newest hours may be partly modelled. Reproduce the numbers with `python python/backtest.py`.

## Run it yourself

The model files are not in this repo (they are 112 MB), so you build them from the original weights. You need Python 3.11, Node 20+, and a few GB of disk.

```bash
# 1. Python environment
python3.11 -m venv python/.venv
python/.venv/bin/pip install -r python/requirements.txt

# 2. Export Timestral-28M to ONNX, split it into 24 MiB chunks, and make a test fixture
cd python
.venv/bin/python export_onnx.py --out build/timestral-28m.onnx
.venv/bin/python split_model.py build/timestral-28m.onnx ../web/model
.venv/bin/python make_fixture.py
cd ..

# 3. Vendor the ONNX Runtime Web files and start a local server
npm install
npm run vendor
npm run dev        # http://localhost:8765
```

Open `http://localhost:8765/?selftest=1` to run the model in your browser on a fixture produced by PyTorch. It prints the largest difference between the two (about 0.0003 when everything is right).

Other scripts: `python/forecast.py` prints a 5-hour forecast for a place next to Open-Meteo's, plus a quick backtest (`--lat`, `--lon`, `--model`, `--json`). `python/backtest.py` runs the multi-city backtest above.

## Deploy

The site is static and served as a Cloudflare Worker with static assets (`wrangler.jsonc`). Cloudflare limits each static file to 25 MiB, which is why `split_model.py` writes 24 MiB chunks with content-hashed names plus a `manifest.json`; the page downloads the chunks in parallel, reassembles them, and caches them with the Cache API. Change the route in `wrangler.jsonc`, then:

```bash
npx wrangler login
npm run deploy
```

## What is in the repo

| Path | What |
|---|---|
| `web/` | The site: `index.html`, `app.js`, `style.css`, icons, `_headers` |
| `python/export_onnx.py` | Exports the model to ONNX at a fixed shape (1 place x 6 series x 168 hours in, 32 hours out) |
| `python/split_model.py` | Splits the export into deployable chunks and writes the manifest |
| `python/forecast.py`, `backtest.py`, `make_fixture.py` | CLI forecast, the backtest, and the browser test fixture |
| `brand/icon.svg` | The app icon |
| `wrangler.jsonc` | Cloudflare deploy config |

### About the ONNX export

The exported graph includes everything `TimestralForecaster.forecast()` does for this shape: causal per-patch scaling, the transformer, the quantile head, the mirrored (negated) second pass that is averaged in, and the non-negative clamp. A few things had to be rewritten so the exporter accepts them (a reshape in place of `Tensor.unfold`, a cumulative-sum form of the running variance, frozen attention masks, and no data-dependent branches). `export_onnx.py` checks the result against PyTorch on two different inputs, and the in-browser self test checks it again end to end.

## Licenses and credits

- The code in this repo is MIT licensed (see `LICENSE`).
- **Timestral-28M** is by Mistral AI under a Modified MIT license. It requires the license notice to ship with the weights (see `web/LICENSE-Timestral.txt`, which the live site also serves) and does not allow use by companies whose monthly revenue exceeds $20 million. The weights are not included in this repo.
- Weather data and city search by [Open-Meteo](https://open-meteo.com) (CC BY 4.0). Their free API is for non-commercial use.
- Runs on [ONNX Runtime Web](https://github.com/microsoft/onnxruntime) (MIT), which `npm run vendor` copies into `web/vendor/`.
