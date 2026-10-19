// Weatherstral: a 24-hour forecast from Timestral-28M running in the browser (onnxruntime-web, WASM).
import * as ort from './vendor/ort.wasm.min.mjs';

const CONTEXT = 168; // hours of history the exported model expects
const HOURS = 24; // hours we show (the model emits 32 per pass)
const STEPS = 32;
const Q = 21;
const LEVELS = [0.01, 0.05, ...Array.from({ length: 17 }, (_, i) => +(0.1 + i * 0.05).toFixed(2)), 0.95, 0.99];
const Q_LO = 2, Q_MED = 10, Q_HI = 18; // quantile levels 0.1 / 0.5 / 0.9
const VARS = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure', 'cloud_cover', 'wind_speed_10m', 'precipitation'];
const [V_TEMP, V_HUM, V_PRESS, V_CLOUD, V_WIND, V_PRECIP] = [0, 1, 2, 3, 4, 5];
const RAIN_MM = 0.1; // an hour with at least this much counts as rain
const MODEL_DIR = './model/'; // manifest.json lists the graph and the weight chunks (each under Cloudflare's 25 MiB limit)

const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));

let unit = store.get('gw:unit', 'F');
let place = store.get('gw:place', { name: 'Indianapolis', admin: 'Indiana', lat: 39.7684, lon: -86.1581 });
let model = null; // { series, cur, ... } for the rendered forecast
let runToken = 0;
const ui = { metric: 'temp', range: 24, hover: null };

// ---------- status ----------
function status(text, { error = false, progress = null, done = false } = {}) {
  const el = $('status');
  el.hidden = false;
  el.className = error ? 'error' : done ? 'done' : '';
  $('status-text').textContent = text;
  const bar = $('status-bar');
  bar.hidden = progress == null;
  if (progress != null) bar.firstElementChild.style.width = `${Math.round(progress * 100)}%`;
}

// ---------- model loading ----------
let sessionPromise = null;

async function fetchBytes(url, label, onProgress, cache) {
  if (cache) {
    const hit = await cache.match(url);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(bytes.length);
      return bytes;
    }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Could not download ${label} (${res.status})`);
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(value.length);
  }
  const bytes = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { bytes.set(c, off); off += c.length; }
  if (cache) { try { await cache.put(url, new Response(bytes)); } catch { /* quota */ } }
  return bytes;
}

function loadModel() {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    status('Loading the model…');
    const manifest = await getJSON(`${MODEL_DIR}manifest.json`);
    // Files are named by content hash, so the cache for a given version never goes stale.
    let cache = null;
    try { cache = await caches.open(`timestral-model-${manifest.version}`); } catch { /* insecure context or blocked */ }

    let received = 0;
    const mb = (n) => (n / 1048576).toFixed(0);
    const onProgress = (n) => {
      received += n;
      status(`Downloading the model (first visit only)… ${mb(received)} / ${mb(manifest.totalBytes)} MB`,
        { progress: received / manifest.totalBytes });
    };
    const [graph, ...parts] = await Promise.all([
      fetchBytes(`${MODEL_DIR}${manifest.graph}`, 'model graph', null, cache),
      ...manifest.parts.map((p) => fetchBytes(`${MODEL_DIR}${p.file}`, 'model weights', onProgress, cache)),
    ]);
    const weights = new Uint8Array(manifest.totalBytes);
    let off = 0;
    for (const p of parts) { weights.set(p, off); off += p.length; }
    if (off !== manifest.totalBytes) throw new Error('Model download was incomplete. Please reload.');

    status('Starting the model…');
    return ort.InferenceSession.create(graph, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
      externalData: [{ path: manifest.dataName, data: weights }],
    });
  })();
  sessionPromise.catch(() => { sessionPromise = null; });
  return sessionPromise;
}

// ---------- data ----------
async function getJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return res.json();
}

async function fetchWeather(lat, lon) {
  const p = new URLSearchParams({
    latitude: lat, longitude: lon,
    hourly: [...VARS, 'is_day', 'precipitation_probability'].join(','),
    current: [...VARS, 'is_day'].join(','),
    past_days: 8, forecast_days: 2, timezone: 'auto',
    temperature_unit: 'fahrenheit', wind_speed_unit: 'mph',
  });
  return getJSON(`https://api.open-meteo.com/v1/forecast?${p}`);
}

/** Fill gaps (nulls) so the model sees a fully observed series: carry forward, then back-fill the start. */
function fillGaps(arr) {
  const out = Float32Array.from(arr, (v) => (v == null ? NaN : v));
  let last = NaN;
  for (let i = 0; i < out.length; i++) { if (Number.isNaN(out[i])) out[i] = last; else last = out[i]; }
  let next = NaN;
  for (let i = out.length - 1; i >= 0; i--) { if (Number.isNaN(out[i])) out[i] = next; else next = out[i]; }
  return out;
}

function buildContext(weather) {
  const times = weather.hourly.time;
  const key = weather.current.time.slice(0, 13); // "YYYY-MM-DDTHH"
  const nowIdx = times.findIndex((t) => t.startsWith(key));
  if (nowIdx < CONTEXT - 1 || nowIdx + HOURS >= times.length) throw new Error('Not enough data returned for this location');
  const ctx = new Float32Array(VARS.length * CONTEXT);
  VARS.forEach((v, vi) => {
    const col = fillGaps(weather.hourly[v].slice(nowIdx + 1 - CONTEXT, nowIdx + 1));
    if (col.some(Number.isNaN)) throw new Error(`No ${v} data for this location`);
    ctx.set(col, vi * CONTEXT);
  });
  return { ctx, nowIdx };
}

async function forecast(session, ctx) {
  const out = await session.run({ context: new ort.Tensor('float32', ctx, [1, VARS.length, CONTEXT]) });
  return out.quantiles.data; // (1, V, 32, Q) flattened
}
const qAt = (q, v, h, k) => q[((v * STEPS) + h) * Q + k];

/** P(value > t), read off the model's 21 predicted quantiles by linear interpolation of the CDF. */
function exceedProb(q, v, h, t) {
  let below = 0;
  for (let k = 0; k < Q; k++) if (qAt(q, v, h, k) <= t) below++;
  if (below === 0) return 0.995;
  if (below === Q) return 0.005;
  const i = below - 1;
  const lo = qAt(q, v, h, i), hi = qAt(q, v, h, i + 1);
  const frac = hi > lo ? clamp((t - lo) / (hi - lo), 0, 1) : 0;
  return clamp(1 - (LEVELS[i] + (LEVELS[i + 1] - LEVELS[i]) * frac), 0.005, 0.995);
}

/** One row per hour: index 0 is "now" (observed), 1..24 are Timestral's forecast for +1h..+24h. */
function buildSeries(weather, nowIdx, q) {
  const cur = weather.current, H = weather.hourly;
  const series = [{
    iso: cur.time, temp: cur.temperature_2m, lo: cur.temperature_2m, hi: cur.temperature_2m, om: cur.temperature_2m,
    cloud: cur.cloud_cover, hum: cur.relative_humidity_2m, press: cur.surface_pressure, wind: cur.wind_speed_10m,
    rain: cur.precipitation >= RAIN_MM ? 1 : 0, omRain: null, isDay: cur.is_day === 1, now: true,
  }];
  for (let k = 1; k <= HOURS; k++) {
    const h = k - 1, at = nowIdx + k;
    const pp = H.precipitation_probability?.[at];
    series.push({
      iso: H.time[at],
      temp: qAt(q, V_TEMP, h, Q_MED), lo: qAt(q, V_TEMP, h, Q_LO), hi: qAt(q, V_TEMP, h, Q_HI), om: H.temperature_2m[at],
      cloud: clamp(qAt(q, V_CLOUD, h, Q_MED), 0, 100), hum: clamp(qAt(q, V_HUM, h, Q_MED), 0, 100),
      press: qAt(q, V_PRESS, h, Q_MED), wind: Math.max(0, qAt(q, V_WIND, h, Q_MED)),
      rain: exceedProb(q, V_PRECIP, h, RAIN_MM), omRain: pp == null ? null : pp / 100, isDay: H.is_day[at] === 1,
    });
  }
  return series;
}

// ---------- formatting ----------
const toUnit = (f) => (unit === 'F' ? f : (f - 32) * 5 / 9);
const deg = (f) => `${Math.round(toUnit(f))}°`;
const hourLabel = (iso) => {
  const h = Number(iso.slice(11, 13));
  return `${((h + 11) % 12) + 1}${h < 12 ? 'AM' : 'PM'}`;
};
const longHour = (iso) => hourLabel(iso).replace(/(AM|PM)/, ' $1');
const weekday = (iso) => new Date(`${iso.slice(0, 13)}:00Z`).toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });
const pct = (p) => `${clamp(Math.round(p * 20) * 5, 0, 95)}%`; // nearest 5%, never claim certainty
const speed = (mph) => (unit === 'F' ? `${Math.round(mph)} mph` : `${Math.round(mph * 1.609)} km/h`);

// ---------- weather icons & wording ----------
const rays = (cx, cy, r0, r1) => Array.from({ length: 8 }, (_, i) => {
  const a = (i * Math.PI) / 4;
  return `<line x1="${(cx + Math.cos(a) * r0).toFixed(1)}" y1="${(cy + Math.sin(a) * r0).toFixed(1)}" x2="${(cx + Math.cos(a) * r1).toFixed(1)}" y2="${(cy + Math.sin(a) * r1).toFixed(1)}"/>`;
}).join('');
const sun = (cx, cy, r) => `<g stroke="#fcd34d" stroke-width="2.4" stroke-linecap="round">${rays(cx, cy, r + 3.5, r + 7)}</g><circle cx="${cx}" cy="${cy}" r="${r}" fill="#fcd34d"/>`;
const moon = (s = 1, tx = 0, ty = 0) => `<path transform="translate(${tx} ${ty}) scale(${s})" d="M24 7a13 13 0 1 0 9 22A10.5 10.5 0 0 1 24 7z" fill="#e2e8f0"/>`;
const cloud = (fill, tr = '') => `<g fill="${fill}" ${tr ? `transform="${tr}"` : ''}><circle cx="14" cy="25" r="6"/><circle cx="22" cy="20" r="8"/><circle cx="30" cy="25.5" r="5.5"/><rect x="14" y="25" width="16" height="6.5" rx="3"/></g>`;
const ICONS = {
  sun: sun(20, 20, 7.5),
  moon: moon(),
  'partly-day': sun(14, 14, 5.5) + cloud('#f8fafc', 'translate(3 3)'),
  'partly-night': moon(0.7, 0, -1) + cloud('#cbd5e1', 'translate(3 3)'),
  cloud: cloud('#f1f5f9', 'translate(0 2)'),
  'cloud-night': cloud('#cbd5e1', 'translate(0 2)'),
  rain: cloud('#cbd5e1', 'translate(0 -4)') +
    '<g stroke="#7dd3fc" stroke-width="2.4" stroke-linecap="round"><line x1="15" y1="30" x2="13" y2="35"/><line x1="22" y1="30" x2="20" y2="35"/><line x1="29" y1="30" x2="27" y2="35"/></g>',
};
const icon = (kind) => `<svg viewBox="0 0 40 40" aria-hidden="true">${ICONS[kind]}</svg>`;

/** Icon + words for one hour, from the (predicted) rain odds and cloud cover. */
function describe(p) {
  const day = p.isDay;
  if (p.rain >= 0.5) return { icon: 'rain', text: 'Rain' };
  if (p.rain >= 0.3) return { icon: 'rain', text: 'Chance of rain' };
  if (p.cloud < 15) return { icon: day ? 'sun' : 'moon', text: day ? 'Sunny' : 'Clear' };
  if (p.cloud < 40) return { icon: day ? 'partly-day' : 'partly-night', text: day ? 'Mostly sunny' : 'Mostly clear' };
  if (p.cloud < 70) return { icon: day ? 'partly-day' : 'partly-night', text: 'Partly cloudy' };
  if (p.cloud < 90) return { icon: day ? 'cloud' : 'cloud-night', text: 'Mostly cloudy' };
  return { icon: day ? 'cloud' : 'cloud-night', text: 'Cloudy' };
}

function themeFor(now) {
  if (now.rain >= 0.5) return 'rain';
  if (!now.isDay) return 'night';
  return now.cloud < 50 ? 'clear-day' : 'cloud-day';
}

function rainSummary(series) {
  const now = series[0], fut = series.slice(1);
  let text, sub = '';
  if (now.rain >= 0.5) {
    const stop = fut.find((p) => p.rain < 0.35);
    text = stop ? `Rain should ease around ${longHour(stop.iso)}.` : 'Rain looks likely to continue for the next 24 hours.';
  } else {
    const likely = fut.find((p) => p.rain >= 0.5);
    const peak = fut.reduce((a, b) => (b.rain > a.rain ? b : a));
    if (likely) text = `Rain likely around ${longHour(likely.iso)} (${pct(likely.rain)}).`;
    else if (peak.rain >= 0.3) text = `Chance of rain around ${longHour(peak.iso)} (${pct(peak.rain)}).`;
    else text = 'No rain expected in the next 24 hours.';
  }
  const omPeak = fut.filter((p) => p.omRain != null).reduce((a, b) => (!a || b.omRain > a.omRain ? b : a), null);
  if (omPeak) {
    sub = omPeak.omRain >= 0.1
      ? `Open-Meteo's forecast peaks at ${pct(omPeak.omRain)} around ${longHour(omPeak.iso)}.`
      : 'Open-Meteo expects it to stay dry.';
  }
  return { text, sub };
}

// ---------- DOM helpers ----------
function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  Object.assign(n, props);
  n.append(...kids);
  return n;
}
const NS = 'http://www.w3.org/2000/svg';
function svgEl(parent, tag, attrs = {}, text) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  parent.append(n);
  return n;
}

// ---------- render ----------
function render() {
  const { series, weather } = model;
  const now = series[0];
  const desc = describe({ ...now, rain: now.rain });
  document.body.dataset.theme = themeFor(now);

  $('place').textContent = [place.name, place.admin].filter(Boolean).join(', ');
  $('now-temp').textContent = deg(now.temp);
  $('cond').textContent = desc.text;
  const temps = series.map((p) => p.temp);
  $('hl').textContent = `H:${deg(Math.max(...temps))}  L:${deg(Math.min(...temps))}`;

  const r = rainSummary(series);
  $('rain-text').replaceChildren(r.text, ...(r.sub ? [el('small', {}, r.sub)] : []));

  $('strip').replaceChildren(...series.map((p, i) => {
    const d = describe(p);
    const b = el('button', { className: 'hour', type: 'button' });
    b.setAttribute('role', 'listitem');
    b.append(
      el('span', { className: 't' }, i === 0 ? 'Now' : hourLabel(p.iso)),
      el('span', { className: 'ic', innerHTML: icon(d.icon) }),
      el('span', { className: 'p' }, i > 0 && p.rain >= 0.15 ? pct(p.rain) : ''),
      el('span', { className: 'd' }, deg(p.temp)),
    );
    b.addEventListener('click', () => setHover(i));
    return b;
  }));

  const at6 = series[6];
  const tile = (h, big, small) => el('div', { className: 'tile' }, el('h3', {}, h), el('p', {}, big), el('small', {}, small));
  const dPress = at6.press - now.press;
  $('tiles').replaceChildren(
    tile('Humidity', `${Math.round(now.hum)}%`, `In 6 hours: ${Math.round(at6.hum)}%`),
    tile('Wind', speed(now.wind), `In 6 hours: ${speed(at6.wind)}`),
    tile('Cloud cover', `${Math.round(now.cloud)}%`, `In 6 hours: ${Math.round(at6.cloud)}%`),
    tile('Pressure', `${Math.round(now.press)} hPa`,
      Math.abs(dPress) < 1 ? 'Steady over the next 6 hours' : dPress < 0 ? 'Falling over the next 6 hours' : 'Rising over the next 6 hours'),
  );

  const a = (href, text) => el('a', { href, target: '_blank', rel: 'noopener noreferrer' }, text);
  $('foot').replaceChildren(
    'Weatherstral uses Timestral-28M, a general-purpose time-series model from Mistral AI (not a weather model). It sees only ' +
    'the last 7 days of hourly readings at this spot (temperature, humidity, pressure, clouds, wind, rain) and extrapolates them. ' +
    'In a test on past data for six U.S. cities its temperatures were about 0.7°F off at +1 hour and 3°F off at +24 hours, ' +
    'better than "same as now" and "same change as yesterday". Its rain odds were well calibrated but only modestly better than ' +
    "recent rain frequency, and mostly useful for the next few hours. Open-Meteo's own forecast is shown for comparison and is " +
    'the better bet. Everything runs in your browser; only your search and the weather data go to Open-Meteo. ',
    el('br'), el('br'),
    'Weather data by ', a('https://open-meteo.com/', 'Open-Meteo.com'), ' (CC BY 4.0). Model: ',
    a('https://huggingface.co/mistralai/Timestral-28M', 'Timestral-28M'), ' by Mistral AI, ',
    a('/LICENSE-Timestral.txt', 'Modified MIT license'), '.',
  );

  $('status').hidden = true;
  $('view').hidden = false;
  drawChart();
  setHover(ui.hover);
}

// ---------- chart ----------
let chartCtx = null;

function drawChart() {
  if (!model) return;
  const box = $('chart');
  const W = Math.max(280, box.clientWidth), H = 232;
  const padL = 38, padR = 12, padT = 12, padB = 24;
  const N = ui.range;
  const pts = model.series.slice(0, N + 1);
  const step = (W - padL - padR) / N;
  const x = (i) => padL + i * step;

  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('width', W);
  svg.setAttribute('height', H);
  svg.setAttribute('role', 'img');
  const isTemp = ui.metric === 'temp';
  svg.setAttribute('aria-label', isTemp ? 'Temperature forecast chart' : 'Chance of rain forecast chart');

  let y;
  let ticks;
  if (isTemp) {
    const vals = pts.flatMap((p) => [p.lo, p.hi, p.om]).map(toUnit);
    let min = Math.min(...vals), max = Math.max(...vals);
    const span = Math.max(max - min, 4);
    min -= span * 0.12; max += span * 0.12;
    y = (f) => padT + (1 - (toUnit(f) - min) / (max - min)) * (H - padT - padB);
    ticks = Array.from({ length: 4 }, (_, i) => { const v = min + ((max - min) * i) / 3; return { yy: padT + (1 - (v - min) / (max - min)) * (H - padT - padB), label: `${Math.round(v)}°` }; });
  } else {
    y = (p) => padT + (1 - p) * (H - padT - padB);
    ticks = [0, 0.25, 0.5, 0.75, 1].map((p) => ({ yy: y(p), label: `${p * 100}%` }));
  }
  for (const t of ticks) {
    svgEl(svg, 'line', { class: 'grid', x1: padL, x2: W - padR, y1: t.yy, y2: t.yy });
    svgEl(svg, 'text', { x: padL - 7, y: t.yy + 4, 'text-anchor': 'end' }, t.label);
  }
  const every = N <= 6 ? 1 : N <= 12 ? 2 : 3;
  pts.forEach((p, i) => { if (i % every === 0) svgEl(svg, 'text', { x: x(i), y: H - 6, 'text-anchor': i === 0 ? 'start' : 'middle' }, i === 0 ? 'Now' : hourLabel(p.iso)); });

  if (N > 12) { // the later hours are a trend, not a precise forecast
    svgEl(svg, 'rect', { x: x(12), y: padT, width: W - padR - x(12), height: H - padT - padB, fill: 'rgb(255 255 255 / .05)' });
    svgEl(svg, 'text', { x: x(12) + 6, y: padT + 11 }, 'less certain');
  }

  const path = (fn, skipNull = false) => {
    let d = '', pen = false;
    pts.forEach((p, i) => {
      const v = fn(p);
      if (v == null) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)} `;
      pen = true;
    });
    return d;
  };

  if (isTemp) {
    const up = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.hi).toFixed(1)}`).join(' ');
    const down = pts.map((p, i) => `L${x(pts.length - 1 - i).toFixed(1)},${y(pts[pts.length - 1 - i].lo).toFixed(1)}`).join(' ');
    svgEl(svg, 'path', { d: `${up} ${down} Z`, fill: 'rgb(127 245 226 / .2)' });
    svgEl(svg, 'path', { d: path((p) => p.om), fill: 'none', stroke: '#fcd34d', 'stroke-width': 2, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' });
    svgEl(svg, 'path', { d: path((p) => p.temp), fill: 'none', stroke: '#7ff5e2', 'stroke-width': 3, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' });
  } else {
    const bw = Math.max(3, step * 0.62);
    pts.forEach((p, i) => {
      if (i === 0) return;
      const top = y(clamp(p.rain, 0.005, 0.95));
      svgEl(svg, 'rect', { x: x(i) - bw / 2, y: top, width: bw, height: Math.max(1, y(0) - top), rx: Math.min(3, bw / 2), fill: 'rgb(125 211 252 / .78)' });
    });
    svgEl(svg, 'path', { d: path((p) => p.omRain), fill: 'none', stroke: '#fcd34d', 'stroke-width': 2, 'stroke-dasharray': '6 5', 'stroke-linecap': 'round' });
  }

  // cursor layer
  const cursor = svgEl(svg, 'g', { visibility: 'hidden' });
  const vline = svgEl(cursor, 'line', { y1: padT, y2: H - padB, stroke: 'rgb(255 255 255 / .75)', 'stroke-width': 1.5 });
  const dotModel = svgEl(cursor, 'circle', { r: 5, fill: isTemp ? '#7ff5e2' : '#7dd3fc', stroke: '#0b1330', 'stroke-width': 2 });
  const dotOm = svgEl(cursor, 'circle', { r: 4, fill: '#fcd34d', stroke: '#0b1330', 'stroke-width': 2 });

  $('chart').replaceChildren(svg);
  chartCtx = { W, N, x, y, padL, step, pts, isTemp, cursor, vline, dotModel, dotOm };

  $('legend').replaceChildren(...(isTemp
    ? [el('span', { innerHTML: '<i class="sw"></i>Timestral <small>(shaded = likely range)</small>' }),
       el('span', { innerHTML: '<i class="sw om"></i>Open-Meteo\'s forecast' })]
    : [el('span', { innerHTML: '<i class="sw bar"></i>Timestral\'s rain chance' }),
       el('span', { innerHTML: '<i class="sw om"></i>Open-Meteo\'s chance' })]));
}

function setHover(i) {
  ui.hover = i;
  const tip = $('tip');
  const strip = $('strip');
  [...strip.children].forEach((c, k) => c.classList.toggle('on', k === i));
  if (i == null || !chartCtx || i > chartCtx.N) {
    if (chartCtx) chartCtx.cursor.setAttribute('visibility', 'hidden');
    tip.hidden = true;
    return;
  }
  const { x, y, W, isTemp, pts, cursor, vline, dotModel, dotOm, padL } = chartCtx;
  const p = pts[i];
  cursor.setAttribute('visibility', 'visible');
  vline.setAttribute('x1', x(i)); vline.setAttribute('x2', x(i));
  dotModel.setAttribute('cx', x(i)); dotModel.setAttribute('cy', isTemp ? y(p.temp) : y(clamp(p.rain, 0.005, 0.95)));
  const omVal = isTemp ? p.om : p.omRain;
  dotOm.setAttribute('visibility', omVal == null || (!isTemp && i === 0) ? 'hidden' : 'visible');
  if (omVal != null) { dotOm.setAttribute('cx', x(i)); dotOm.setAttribute('cy', y(omVal)); }

  const d = describe(p);
  const row = (k, v) => el('div', { className: 'row' }, el('span', {}, k), el('b', {}, v));
  const when = el('div', { className: 'when' }, i === 0 ? 'Now' : `${weekday(p.iso)} ${longHour(p.iso)}`);
  tip.replaceChildren(
    when,
    isTemp
      ? el('div', { className: 'big' }, deg(p.temp), ...(i > 0 ? [el('small', {}, `${deg(p.lo)}–${deg(p.hi)}`)] : []))
      : el('div', { className: 'big' }, i === 0 ? (p.rain ? 'Raining' : 'Dry') : pct(p.rain), ...(i > 0 ? [el('small', {}, 'chance of rain')] : [])),
    ...(isTemp
      ? [row('Open-Meteo', i === 0 ? 'observed' : deg(p.om)), row('Rain chance', i === 0 ? (p.rain ? 'raining' : 'dry') : pct(p.rain)), row('Sky', d.text)]
      : [row('Open-Meteo', p.omRain == null || i === 0 ? '–' : pct(p.omRain)), row('Temperature', deg(p.temp)), row('Sky', d.text)]),
  );
  tip.hidden = false;
  const onRight = x(i) > W / 2;
  tip.style.left = onRight ? `${padL + 4}px` : 'auto';
  tip.style.right = onRight ? 'auto' : '8px';

  const item = strip.children[i];
  if (item && (item.offsetLeft < strip.scrollLeft || item.offsetLeft + item.offsetWidth > strip.scrollLeft + strip.clientWidth)) {
    strip.scrollLeft = item.offsetLeft - strip.clientWidth / 2 + item.offsetWidth / 2;
  }
}

function indexFromEvent(e) {
  const r = $('chart').getBoundingClientRect();
  return clamp(Math.round((e.clientX - r.left - chartCtx.padL) / chartCtx.step), 0, chartCtx.N);
}
const chartEl = $('chart');
chartEl.addEventListener('pointerdown', (e) => { if (chartCtx) { try { chartEl.setPointerCapture(e.pointerId); } catch { /* ok */ } setHover(indexFromEvent(e)); } });
chartEl.addEventListener('pointermove', (e) => { if (chartCtx) setHover(indexFromEvent(e)); });
chartEl.addEventListener('pointerleave', (e) => { if (e.pointerType === 'mouse') setHover(null); });
chartEl.addEventListener('keydown', (e) => {
  if (!chartCtx || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
  e.preventDefault();
  setHover(clamp((ui.hover ?? 0) + (e.key === 'ArrowRight' ? 1 : -1), 0, chartCtx.N));
});
new ResizeObserver(() => { if (model) { drawChart(); setHover(ui.hover); } }).observe(chartEl);

function bindSegment(id, key, parse = (v) => v) {
  const group = $(id);
  group.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    ui[key] = parse(Object.values(b.dataset)[0]);
    group.querySelectorAll('button').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    if (model) { drawChart(); setHover(ui.hover != null && ui.hover <= ui.range ? ui.hover : null); }
  });
}
bindSegment('metric', 'metric');
bindSegment('range', 'range', Number);

// ---------- main flow ----------
async function load() {
  const token = ++runToken;
  $('view').hidden = true;
  ui.hover = null;
  try {
    status('Fetching the last week of weather…');
    const [session, weather] = await Promise.all([loadModel(), fetchWeather(place.lat, place.lon)]);
    if (token !== runToken) return;
    const { ctx, nowIdx } = buildContext(weather);
    status('Predicting…');
    const t0 = performance.now();
    const q = await forecast(session, ctx);
    const ms = Math.round(performance.now() - t0);
    if (token !== runToken) return;
    model = { weather, nowIdx, q, ms, series: buildSeries(weather, nowIdx, q) };
    render();
    console.log(`forecast took ${ms} ms`);
  } catch (e) {
    console.error(e);
    status(`Something went wrong: ${e.message}`, { error: true });
  }
}

function setUnit(u) {
  unit = u;
  store.set('gw:unit', u);
  document.querySelectorAll('.units button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.unit === u)));
  if (model) render();
}

function choosePlace(p) {
  place = p;
  store.set('gw:place', p);
  $('q').value = '';
  hideResults();
  load();
}

// ---------- search ----------
let results = [];
let active = -1;
let searchTimer = 0;

function hideResults() { $('results').hidden = true; active = -1; }

function showResults() {
  const ul = $('results');
  ul.replaceChildren(...results.map((r, i) => {
    const li = el('li', { role: 'option' }, r.name, el('small', {}, [r.admin, r.country].filter(Boolean).join(', ')));
    li.setAttribute('aria-selected', String(i === active));
    li.addEventListener('mousedown', (e) => { e.preventDefault(); choosePlace(r); });
    return li;
  }));
  ul.hidden = results.length === 0;
}

async function search(text) {
  if (text.trim().length < 2) { results = []; showResults(); return; }
  try {
    const p = new URLSearchParams({ name: text.trim(), count: 6, language: 'en', format: 'json' });
    const data = await getJSON(`https://geocoding-api.open-meteo.com/v1/search?${p}`);
    if ($('q').value.trim() !== text.trim()) return; // stale
    results = (data.results ?? []).map((r) => ({ name: r.name, admin: r.admin1, country: r.country_code, lat: r.latitude, lon: r.longitude }));
    active = -1;
    showResults();
  } catch (e) { console.warn('geocoding failed', e); }
}

$('q').addEventListener('input', (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => search(e.target.value), 250);
});
$('q').addEventListener('keydown', (e) => {
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    if (!results.length) return;
    e.preventDefault();
    active = (active + (e.key === 'ArrowDown' ? 1 : -1) + results.length) % results.length;
    showResults();
  } else if (e.key === 'Escape') hideResults();
});
$('q').addEventListener('blur', hideResults);
$('search').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!results.length) await search($('q').value);
  const pick = results[Math.max(active, 0)];
  if (pick) choosePlace(pick);
});

$('locate').addEventListener('click', () => {
  if (!navigator.geolocation) return status('This browser has no location support.', { error: true });
  navigator.geolocation.getCurrentPosition(
    (pos) => choosePlace({ name: 'Your location', admin: '', lat: +pos.coords.latitude.toFixed(4), lon: +pos.coords.longitude.toFixed(4) }),
    () => status('Location permission was denied. Try searching for a city instead.', { error: true }),
    { timeout: 10000 },
  );
});

document.querySelectorAll('.units button').forEach((b) => b.addEventListener('click', () => setUnit(b.dataset.unit)));
document.querySelectorAll('.units button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.unit === unit)));

// ---------- self test: run the model on a fixture made by PyTorch and compare ----------
async function selfTest() {
  const out = $('selftest');
  out.hidden = false;
  out.textContent = 'self-test: loading model…';
  const session = await loadModel();
  const fx = await getJSON('./test/fixture.json');
  const t0 = performance.now();
  const q = await forecast(session, Float32Array.from(fx.context.flat()));
  const ms = Math.round(performance.now() - t0);
  let worst = 0;
  for (let i = 0; i < fx.quantiles.length; i++) worst = Math.max(worst, Math.abs(q[i] - fx.quantiles[i]));
  let worstRain = 0;
  for (let h = 0; h < STEPS; h++) worstRain = Math.max(worstRain, Math.abs(exceedProb(q, V_PRECIP, h, RAIN_MM) - fx.rain[h]));
  const lv = LEVELS.every((l, i) => Math.abs(l - fx.levels[i]) < 1e-9);
  const med = Array.from({ length: 5 }, (_, h) => qAt(q, V_TEMP, h, Q_MED).toFixed(2));
  out.textContent = `self-test: outputs ${q.length} (expected ${fx.quantiles.length}), levels match: ${lv}\n` +
    `max |browser - pytorch| over all outputs = ${worst.toExponential(2)}\n` +
    `max |browser rain prob - python rain prob| = ${worstRain.toExponential(2)}\n` +
    `temp median +1..5h: ${med.join(', ')}\ninference ${ms} ms, threads ${ort.env.wasm.numThreads}`;
  window.__selftest = { worst, worstRain, ms, n: q.length, lv };
}

if (new URLSearchParams(location.search).has('selftest')) selfTest().catch((e) => { $('selftest').textContent = `self-test failed: ${e.message}`; console.error(e); });
load();
