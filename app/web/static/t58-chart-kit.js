/*
 * T58 Chart Kit -- one configurable price chart for Interactive Replay AND Live Monitor (Lightweight Charts v4).
 *
 *   const kit = T58ChartKit.create({ el, drawToolbarEl, controlsEl, legend: true, title: 'ES', drawKey: 'replay:abc' });
 *   kit.setBars(bars);            // full (re)load -- bars are {time, open, high, low, close, volume}
 *   kit.updateBar(bar);           // live tick / replay step: replaces the last bar when time matches, else appends
 *   kit.setMarkers(markers);      kit.setPriceLines('trade', [{price, color, title, style}]);
 *   kit.setAnnotations([...]);    // trade segments / zones drawn under the user's own drawings
 *   kit.fit();  kit.screenshot();  kit.setTitle('ES 5m');  kit.setDrawKey('live:ES');
 *
 * What the user can change without touching code (all remembered in localStorage, shared by both pages so the chart
 * looks and behaves the same everywhere):
 *   chart type   candles / hollow / OHLC bars / line / area / baseline / Heikin-Ashi
 *   indicators   SMA EMA WMA Bollinger VWAP Donchian Supertrend | RSI MACD Stochastic ATR | volume (+ volume MA)
 *                -- any number of each, with editable periods and colours
 *   look         theme preset, candle colours, grid, crosshair mode, watermark, price-axis mode (normal / log / % / indexed),
 *                auto-scale, last-price line, high/low lines, volume, time format, bar spacing
 *   drawing      the full T58ChartDrawingTools set (trend lines, fib, channels, ruler, long/short R:R, text ...)
 */
(function (global) {
  "use strict";
  const LS_KEY = "t58-chart-kit-v1";
  const STYLE_ID = "t58-kit-style";

  // ------------------------------------------------------------------------------------------------ presets
  const THEMES = {
    t58:   { label: "T58 (follow app)", auto: true },
    tvdark:{ label: "Midnight",   bg: "#131722", text: "#b2b5be", grid: "rgba(42,46,57,.9)", border: "#2a2e39" },
    dark:  { label: "Dark",       bg: "#0d1117", text: "#9aa4b2", grid: "rgba(255,255,255,.05)", border: "#242b36" },
    light: { label: "Light",      bg: "#ffffff", text: "#434651", grid: "rgba(42,46,57,.08)", border: "#d6d9e0" },
    contrast: { label: "High contrast", bg: "#000000", text: "#f2f2f2", grid: "rgba(255,255,255,.14)", border: "#555" },
  };
  const CANDLE_PRESETS = {
    t58:   { label: "Green / Red", up: "#35e0b0", down: "#ff6b6b" },
    tv:    { label: "Teal / Red (TradingView)", up: "#26a69a", down: "#ef5350" },
    blue:  { label: "Blue / Orange (colour-blind safe)", up: "#2f81f7", down: "#ff9f1c" },
    mono:  { label: "White / Grey", up: "#e5e7eb", down: "#6b7280" },
  };
  const DEFAULTS = {
    type: "candles", theme: "t58", candles: "t58", upColor: null, downColor: null,
    grid: "both", crosshair: "normal", priceMode: "normal", autoScale: true, lastPrice: true, hiLo: false, volume: true,
    watermark: true, timeFormat: "local", barSpacing: 8, rightOffset: 6, legend: true,
    indicators: [],
  };
  const PALETTE = ["#f0b429", "#22d3ee", "#8b7cff", "#ff6b6b", "#35e0b0", "#60a5fa", "#f472b6", "#a3e635"];

  const IND_DEFS = {
    sma:        { label: "SMA", group: "Overlays", params: { period: 20 } },
    ema:        { label: "EMA", group: "Overlays", params: { period: 20 } },
    wma:        { label: "WMA", group: "Overlays", params: { period: 20 } },
    bb:         { label: "Bollinger Bands", group: "Overlays", params: { period: 20, mult: 2 } },
    vwap:       { label: "VWAP (daily reset)", group: "Overlays", params: {} },
    donchian:   { label: "Donchian Channel", group: "Overlays", params: { period: 20 } },
    supertrend: { label: "Supertrend", group: "Overlays", params: { period: 10, mult: 3 } },
    rsi:        { label: "RSI", group: "Oscillators (own pane)", params: { period: 14 }, pane: true },
    macd:       { label: "MACD", group: "Oscillators (own pane)", params: { fast: 12, slow: 26, signal: 9 }, pane: true },
    stoch:      { label: "Stochastic", group: "Oscillators (own pane)", params: { k: 14, d: 3 }, pane: true },
    atr:        { label: "ATR", group: "Oscillators (own pane)", params: { period: 14 }, pane: true },
  };

  // ------------------------------------------------------------------------------------------------ maths
  function sma(v, n) {
    const out = new Array(v.length).fill(undefined); let s = 0;
    for (let i = 0; i < v.length; i++) { s += v[i]; if (i >= n) s -= v[i - n]; if (i >= n - 1) out[i] = s / n; }
    return out;
  }
  function ema(v, n) {
    const out = new Array(v.length).fill(undefined), k = 2 / (n + 1); let prev;
    for (let i = 0; i < v.length; i++) {
      if (i < n - 1) continue;
      if (i === n - 1) { let s = 0; for (let j = 0; j < n; j++) s += v[j]; prev = s / n; }
      else prev = v[i] * k + prev * (1 - k);
      out[i] = prev;
    }
    return out;
  }
  function wma(v, n) {
    const out = new Array(v.length).fill(undefined), den = n * (n + 1) / 2;
    for (let i = n - 1; i < v.length; i++) { let s = 0; for (let j = 0; j < n; j++) s += v[i - n + 1 + j] * (j + 1); out[i] = s / den; }
    return out;
  }
  function rma(v, n) {            // Wilder smoothing
    const out = new Array(v.length).fill(undefined); let prev;
    for (let i = 0; i < v.length; i++) {
      if (v[i] === undefined) continue;
      if (prev === undefined) { if (i >= n - 1) { let s = 0, c = 0; for (let j = i - n + 1; j <= i; j++) { if (v[j] !== undefined) { s += v[j]; c++; } } if (c === n) prev = s / n; else continue; } else continue; }
      else prev = (prev * (n - 1) + v[i]) / n;
      out[i] = prev;
    }
    return out;
  }
  function stdev(v, n) {
    const out = new Array(v.length).fill(undefined);
    for (let i = n - 1; i < v.length; i++) {
      let m = 0; for (let j = i - n + 1; j <= i; j++) m += v[j]; m /= n;
      let q = 0; for (let j = i - n + 1; j <= i; j++) q += (v[j] - m) ** 2;
      out[i] = Math.sqrt(q / n);
    }
    return out;
  }
  function trueRange(b) {
    return b.map((x, i) => i === 0 ? x.high - x.low : Math.max(x.high - x.low, Math.abs(x.high - b[i - 1].close), Math.abs(x.low - b[i - 1].close)));
  }
  function dayKey(t) { return Math.floor(t / 86400); }

  // Each calc returns lines: [{name, values[], color, kind:'line'|'hist'|'area', width, dash, base?}], plus optional levels for panes
  function calc(inst, bars) {
    const c = bars.map((b) => b.close), p = inst.params, col = inst.colors;
    switch (inst.type) {
      case "sma": return { lines: [{ name: "SMA " + p.period, values: sma(c, p.period), color: col[0] }] };
      case "ema": return { lines: [{ name: "EMA " + p.period, values: ema(c, p.period), color: col[0] }] };
      case "wma": return { lines: [{ name: "WMA " + p.period, values: wma(c, p.period), color: col[0] }] };
      case "bb": {
        const m = sma(c, p.period), s = stdev(c, p.period);
        return { lines: [
          { name: "BB mid", values: m, color: col[0] },
          { name: "BB up", values: m.map((x, i) => x === undefined ? undefined : x + p.mult * s[i]), color: col[1] || col[0], dash: 2 },
          { name: "BB low", values: m.map((x, i) => x === undefined ? undefined : x - p.mult * s[i]), color: col[1] || col[0], dash: 2 },
        ] };
      }
      case "vwap": {
        let pv = 0, vv = 0, day = null;
        return { lines: [{ name: "VWAP", color: col[0], dash: 2, values: bars.map((b) => {
          const d = dayKey(b.time); if (d !== day) { day = d; pv = 0; vv = 0; }
          const tp = (b.high + b.low + b.close) / 3, v = b.volume || 0; pv += tp * v; vv += v;
          return vv > 0 ? pv / vv : b.close;
        }) }] };
      }
      case "donchian": {
        const n = p.period, hi = new Array(bars.length).fill(undefined), lo = hi.slice(), mid = hi.slice();
        for (let i = n - 1; i < bars.length; i++) {
          let h = -Infinity, l = Infinity; for (let j = i - n + 1; j <= i; j++) { if (bars[j].high > h) h = bars[j].high; if (bars[j].low < l) l = bars[j].low; }
          hi[i] = h; lo[i] = l; mid[i] = (h + l) / 2;
        }
        return { lines: [{ name: "DC up", values: hi, color: col[0] }, { name: "DC low", values: lo, color: col[0] }, { name: "DC mid", values: mid, color: col[0], dash: 2 }] };
      }
      case "supertrend": {
        const atr = rma(trueRange(bars), p.period), n = bars.length;
        const up = new Array(n).fill(undefined), dn = up.slice();
        let fu, fl, dir = 1;
        for (let i = 0; i < n; i++) {
          if (atr[i] === undefined) continue;
          const hl2 = (bars[i].high + bars[i].low) / 2;
          let bu = hl2 + p.mult * atr[i], bl = hl2 - p.mult * atr[i];
          if (fu === undefined) { fu = bu; fl = bl; }
          else {
            bu = (bu < fu || bars[i - 1].close > fu) ? bu : fu;
            bl = (bl > fl || bars[i - 1].close < fl) ? bl : fl;
            if (dir === 1 && bars[i].close < fl) dir = -1; else if (dir === -1 && bars[i].close > fu) dir = 1;
            fu = bu; fl = bl;
          }
          if (dir === 1) up[i] = fl; else dn[i] = fu;
        }
        return { lines: [{ name: "ST up", values: up, color: "#35e0b0", width: 2, gaps: true }, { name: "ST down", values: dn, color: "#ff6b6b", width: 2, gaps: true }] };
      }
      case "rsi": {
        const n = p.period, g = [], l = [];
        for (let i = 0; i < c.length; i++) { const d = i ? c[i] - c[i - 1] : 0; g.push(i ? Math.max(d, 0) : undefined); l.push(i ? Math.max(-d, 0) : undefined); }
        const ag = rma(g, n), al = rma(l, n);
        return { levels: [30, 50, 70], range: [0, 100], lines: [{ name: "RSI " + n, color: col[0], values: ag.map((x, i) => x === undefined ? undefined : al[i] === 0 ? 100 : 100 - 100 / (1 + x / al[i])) }] };
      }
      case "macd": {
        const f = ema(c, p.fast), s = ema(c, p.slow);
        const macd = f.map((x, i) => x === undefined || s[i] === undefined ? undefined : x - s[i]);
        const clean = macd.map((x) => x === undefined ? 0 : x), first = macd.findIndex((x) => x !== undefined);
        const sig = first < 0 ? macd.map(() => undefined) : new Array(first).fill(undefined).concat(ema(clean.slice(first), p.signal));
        const hist = macd.map((x, i) => x === undefined || sig[i] === undefined ? undefined : x - sig[i]);
        return { levels: [0], lines: [
          { name: "Hist", values: hist, color: "#60a5fa", kind: "hist" },
          { name: "MACD", values: macd, color: col[0] }, { name: "Signal", values: sig, color: col[1] || "#ff6b6b" } ] };
      }
      case "stoch": {
        const k = new Array(bars.length).fill(undefined);
        for (let i = p.k - 1; i < bars.length; i++) {
          let h = -Infinity, lo = Infinity; for (let j = i - p.k + 1; j <= i; j++) { if (bars[j].high > h) h = bars[j].high; if (bars[j].low < lo) lo = bars[j].low; }
          k[i] = h === lo ? 50 : (bars[i].close - lo) / (h - lo) * 100;
        }
        const kc = k.map((x) => x === undefined ? 0 : x), first = p.k - 1;
        const d = bars.length > first ? new Array(first).fill(undefined).concat(sma(kc.slice(first), p.d)) : k.map(() => undefined);
        return { levels: [20, 80], range: [0, 100], lines: [{ name: "%K", values: k, color: col[0] }, { name: "%D", values: d, color: col[1] || "#ff6b6b" }] };
      }
      case "atr": return { lines: [{ name: "ATR " + p.period, values: rma(trueRange(bars), p.period), color: col[0] }] };
      default: return { lines: [] };
    }
  }

  function heikin(bars) {
    const out = []; let po, pc;
    for (let i = 0; i < bars.length; i++) {
      const b = bars[i], c = (b.open + b.high + b.low + b.close) / 4, o = i === 0 ? (b.open + b.close) / 2 : (po + pc) / 2;
      out.push({ time: b.time, open: o, high: Math.max(b.high, o, c), low: Math.min(b.low, o, c), close: c }); po = o; pc = c;
    }
    return out;
  }

  // ------------------------------------------------------------------------------------------------ storage
  function loadPrefs() {
    let o = {};
    try { o = JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { o = {}; }
    const p = Object.assign({}, DEFAULTS, o);
    p.indicators = Array.isArray(o.indicators) ? o.indicators : [];
    return p;
  }
  function savePrefs(p) { try { localStorage.setItem(LS_KEY, JSON.stringify(p)); } catch (e) { /* ignore */ } }
  // Several charts on one page share these settings; write only the keys that changed so one chart never clobbers another's.
  function savePart(p, keys) {
    let cur = {};
    try { cur = JSON.parse(localStorage.getItem(LS_KEY) || "{}") || {}; } catch (e) { cur = {}; }
    keys.forEach((k) => { cur[k] = p[k]; });
    try { localStorage.setItem(LS_KEY, JSON.stringify(cur)); } catch (e) { /* ignore */ }
  }

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style"); s.id = STYLE_ID;
    s.textContent = [
      ".t58k-bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}",
      ".t58k-btn{background:var(--panel-3,#10131a);color:var(--text,#e5e7eb);border:1px solid var(--border-light,rgba(255,255,255,.18));border-radius:7px;padding:5px 10px;font-size:12px;font-weight:600;cursor:pointer}",
      ".t58k-btn:hover{border-color:var(--teal,#35e0b0);color:var(--teal,#35e0b0)}",
      ".t58k-btn.on{border-color:var(--teal,#35e0b0);background:rgba(var(--teal-rgb,53,224,176),.16);color:var(--teal,#35e0b0)}",
      ".t58k-pop-wrap{position:relative;display:inline-block}",
      ".t58k-pop{position:absolute;top:100%;left:0;margin-top:5px;z-index:80;min-width:260px;max-width:340px;max-height:70vh;overflow:auto;background:var(--panel-2,#161a22);border:1px solid var(--border-light,rgba(255,255,255,.2));border-radius:10px;padding:10px;box-shadow:0 10px 30px rgba(0,0,0,.45);display:none;font-size:12px;color:var(--text,#e5e7eb)}",
      ".t58k-pop.open{display:block}",
      ".t58k-pop h4{margin:10px 0 5px;font-size:10.5px;letter-spacing:.06em;text-transform:uppercase;color:var(--text-muted,#9aa4b2)}",
      ".t58k-pop h4:first-child{margin-top:0}",
      ".t58k-row{display:flex;align-items:center;gap:8px;margin:4px 0;justify-content:space-between}",
      ".t58k-row label{color:var(--text-muted,#9aa4b2)}",
      ".t58k-pop select,.t58k-pop input[type=number],.t58k-pop input[type=text]{background:var(--panel-3,#10131a);color:var(--text,#e5e7eb);border:1px solid var(--border-light,rgba(255,255,255,.2));border-radius:5px;padding:3px 5px;font-size:12px}",
      ".t58k-pop input[type=number]{width:58px}",
      ".t58k-pop input[type=color]{width:22px;height:22px;border:0;padding:0;background:none;cursor:pointer}",
      ".t58k-add{display:block;width:100%;text-align:left;background:transparent;border:0;color:var(--text,#e5e7eb);padding:5px 6px;border-radius:5px;cursor:pointer;font-size:12px}",
      ".t58k-add:hover{background:rgba(var(--teal-rgb,53,224,176),.14)}",
      ".t58k-ind{border:1px solid var(--border,rgba(255,255,255,.12));border-radius:8px;padding:6px 8px;margin:5px 0}",
      ".t58k-ind .nm{font-weight:700}",
      ".t58k-x{background:transparent;border:0;color:#ff6b6b;cursor:pointer;font-size:14px;padding:0 3px}",
      ".t58k-legend{position:absolute;left:10px;top:8px;z-index:7;pointer-events:none;font-size:11.5px;line-height:1.5;font-variant-numeric:tabular-nums;color:var(--text,#e5e7eb);text-shadow:0 0 4px rgba(0,0,0,.35)}",
      ".t58k-legend .ttl{font-size:14px;font-weight:700}",
      ".t58k-legend .mut{color:var(--text-muted,#9aa4b2)}",
      ".t58k-legend .up{color:var(--teal,#35e0b0)} .t58k-legend .dn{color:#ff6b6b}",
      ".t58k-pane-lbl{position:absolute;left:10px;z-index:6;pointer-events:none;font-size:10.5px;color:var(--text-muted,#9aa4b2)}",
      ".t58k-full{position:fixed!important;inset:0;z-index:2000;background:var(--bg,#0b0e14);padding:10px}",
    ].join("\n");
    document.head.appendChild(s);
  }

  // ------------------------------------------------------------------------------------------------ the kit
  function create(cfg) {
    injectStyle();
    const LW = global.LightweightCharts;
    if (!LW) throw new Error("Lightweight Charts is not loaded");
    const prefs = loadPrefs();
    const el = cfg.el;
    const rootCss = getComputedStyle(document.documentElement);
    const cssv = (n, f) => (rootCss.getPropertyValue(n).trim() || f);
    if (!el.style.position || el.style.position === "static") el.style.position = "relative";

    let bars = [];
    let title = cfg.title || "";
    let main = null, volume = null, volMa = null;
    let markers = [];
    const priceLineSets = {};   // name -> [{price,color,title,style}]
    const livePriceLines = {};  // name -> [IPriceLine]
    let hiLoLines = [];
    let annotations = [];
    let nextId = 1;
    let instances = prefs.indicators.map(hydrate).filter(Boolean);
    let pending = null;        // earliest dirty index for the next flush
    let rafId = 0;
    let drawTools = null;
    const paneLabels = [];

    function hydrate(i) {
      if (!i || !IND_DEFS[i.type]) return null;
      return { id: "i" + (nextId++), type: i.type, params: Object.assign({}, IND_DEFS[i.type].params, i.params || {}), colors: i.colors && i.colors.length ? i.colors : [PALETTE[0]], visible: i.visible !== false, series: [], calc: null };
    }
    function persist() {
      prefs.indicators = instances.map((i) => ({ type: i.type, params: i.params, colors: i.colors, visible: i.visible }));
      savePart(prefs, ["indicators"]);
    }

    // ---- colours
    function palette() {
      const t = THEMES[prefs.theme] || THEMES.t58;
      const cp = CANDLE_PRESETS[prefs.candles] || CANDLE_PRESETS.t58;
      const light = document.documentElement.getAttribute("data-theme") === "light";
      const base = t.auto ? (light ? THEMES.light : {
        bg: "transparent", text: cssv("--text-muted", "#9aa4b2"), grid: "rgba(127,127,127,.12)", border: cssv("--border", "#2a2f3a"),
      }) : t;
      return { bg: base.bg, text: base.text, grid: base.grid, border: base.border, up: prefs.upColor || cp.up, down: prefs.downColor || cp.down };
    }
    function alpha(hex, a) {
      const m = /^#([0-9a-f]{6})$/i.exec(hex || ""); if (!m) return hex;
      const n = parseInt(m[1], 16); return "rgba(" + (n >> 16) + "," + ((n >> 8) & 255) + "," + (n & 255) + "," + a + ")";
    }

    // ---- chart
    const chart = LW.createChart(el, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: "#9aa4b2", fontSize: 12 },
      timeScale: { timeVisible: true, secondsVisible: false, rightOffset: prefs.rightOffset, barSpacing: prefs.barSpacing },
      crosshair: { mode: LW.CrosshairMode.Normal },
      rightPriceScale: { scaleMargins: { top: 0.06, bottom: 0.04 } },
      watermark: { visible: false },
    });

    // ---- legend
    const legendEl = document.createElement("div");
    legendEl.className = "t58k-legend";
    el.appendChild(legendEl);

    // ---- annotation canvas (trade segments / zones) -- sits under the user's drawings
    const ann = document.createElement("canvas");
    ann.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;z-index:4";
    el.appendChild(ann);
    const actx = ann.getContext("2d");
    function sizeAnn() {
      const r = el.getBoundingClientRect(), d = window.devicePixelRatio || 1;
      ann.width = Math.max(1, Math.round(r.width * d)); ann.height = Math.max(1, Math.round(r.height * d));
      ann.style.width = r.width + "px"; ann.style.height = r.height + "px"; actx.setTransform(d, 0, 0, d, 0, 0);
      drawAnn();
    }
    function drawAnn() {
      const w = ann.width / (window.devicePixelRatio || 1), h = ann.height / (window.devicePixelRatio || 1);
      actx.clearRect(0, 0, w, h);
      if (!annotations.length || !main) return;
      let pw = w - 56; try { pw = chart.timeScale().width() || pw; } catch (e) { /* ignore */ }
      actx.save(); actx.beginPath(); actx.rect(0, 0, pw, h - 26); actx.clip();
      const X = (t) => { const x = chart.timeScale().timeToCoordinate(t); return x == null ? NaN : x; };
      const Y = (p) => { const y = main.priceToCoordinate(p); return y == null ? NaN : y; };
      annotations.forEach((a) => {
        actx.setLineDash(a.dash || []);
        actx.strokeStyle = a.color || "#9aa4b2"; actx.fillStyle = a.color || "#9aa4b2"; actx.lineWidth = a.width || 1.2;
        if (a.kind === "segment") {
          const x0 = X(a.t0), x1 = X(a.t1), y0 = Y(a.p0), y1 = Y(a.p1);
          if ([x0, x1, y0, y1].some(Number.isNaN)) return;
          actx.beginPath(); actx.moveTo(x0, y0); actx.lineTo(x1, y1); actx.stroke();
        } else if (a.kind === "zone") {
          let x0 = X(a.t0), x1 = a.t1 == null ? pw : X(a.t1);
          const y0 = Y(a.p0), y1 = Y(a.p1);
          if (Number.isNaN(x0)) x0 = 0; if (Number.isNaN(x1)) x1 = pw;
          if ([y0, y1].some(Number.isNaN)) return;
          actx.globalAlpha = a.fill == null ? 0.12 : a.fill; actx.fillRect(x0, Math.min(y0, y1), x1 - x0, Math.abs(y1 - y0)); actx.globalAlpha = 1;
        }
      });
      actx.restore();
    }
    let annRo = null;
    if (typeof ResizeObserver !== "undefined") { annRo = new ResizeObserver(sizeAnn); annRo.observe(el); }
    chart.timeScale().subscribeVisibleLogicalRangeChange(() => requestAnimationFrame(drawAnn));

    // ---- series construction
    function candleOpts() {
      const c = palette();
      return { upColor: c.up, downColor: c.down, borderVisible: false, wickUpColor: c.up, wickDownColor: c.down };
    }
    function makeMain() {
      const c = palette(), t = prefs.type;
      if (main) { try { chart.removeSeries(main); } catch (e) { /* ignore */ } main = null; }
      const common = { lastValueVisible: prefs.lastPrice, priceLineVisible: prefs.lastPrice };
      if (t === "hollow") main = chart.addCandlestickSeries(Object.assign({ upColor: "rgba(0,0,0,0)", downColor: c.down, borderUpColor: c.up, borderDownColor: c.down, wickUpColor: c.up, wickDownColor: c.down, borderVisible: true }, common));
      else if (t === "bars") main = chart.addBarSeries(Object.assign({ upColor: c.up, downColor: c.down, thinBars: false }, common));
      else if (t === "line") main = chart.addLineSeries(Object.assign({ color: c.up, lineWidth: 2 }, common));
      else if (t === "area") main = chart.addAreaSeries(Object.assign({ lineColor: c.up, topColor: alpha(c.up, 0.35), bottomColor: alpha(c.up, 0.02), lineWidth: 2 }, common));
      else if (t === "baseline") main = chart.addBaselineSeries(Object.assign({ topLineColor: c.up, topFillColor1: alpha(c.up, 0.3), topFillColor2: alpha(c.up, 0.02), bottomLineColor: c.down, bottomFillColor1: alpha(c.down, 0.02), bottomFillColor2: alpha(c.down, 0.3), lineWidth: 2 }, common));
      else main = chart.addCandlestickSeries(Object.assign(candleOpts(), common));
      if (drawTools) drawTools.setSeries(main);
      return main;
    }
    function mainData(list) {
      const src = prefs.type === "heikin" ? heikin(list) : list;
      if (prefs.type === "line" || prefs.type === "area" || prefs.type === "baseline") return src.map((b) => ({ time: b.time, value: b.close }));
      return src.map((b) => ({ time: b.time, open: b.open, high: b.high, low: b.low, close: b.close }));
    }
    function pointOf(b) {
      return prefs.type === "line" || prefs.type === "area" || prefs.type === "baseline" ? { time: b.time, value: b.close } : { time: b.time, open: b.open, high: b.high, low: b.low, close: b.close };
    }
    if (prefs.type === "heikin") { /* uses candlestick series */ }

    function paneInstances() { return instances.filter((i) => IND_DEFS[i.type].pane && i.visible); }
    function applyLayout() {
      const panes = paneInstances(), n = panes.length, h = n ? Math.min(0.22, 0.62 / n) : 0;
      const mainBottom = n ? n * h + 0.03 : 0.04;
      chart.priceScale("right").applyOptions({
        scaleMargins: { top: 0.06, bottom: mainBottom }, mode: ({ normal: 0, log: 1, percent: 2, indexed: 3 })[prefs.priceMode] || 0, autoScale: prefs.autoScale,
      });
      if (volume) {
        const region = 1 - mainBottom;
        volume.priceScale().applyOptions({ scaleMargins: { top: Math.max(0.5, 1 - mainBottom - region * 0.2), bottom: mainBottom } });
      }
      paneLabels.forEach((l) => l.remove()); paneLabels.length = 0;
      panes.forEach((inst, k) => {
        const top = 1 - (n - k) * h, bottom = (n - k - 1) * h;
        inst.series.forEach((s) => { s.priceScale().applyOptions({ scaleMargins: { top: top + 0.01, bottom: bottom + 0.01 }, visible: true, borderVisible: false }); });
      });
      positionPaneLabels();
    }
    function positionPaneLabels() {
      paneLabels.forEach((l) => l.remove()); paneLabels.length = 0;
      const panes = paneInstances(), n = panes.length; if (!n) return;
      const H = el.clientHeight - 26, h = Math.min(0.22, 0.62 / n);
      panes.forEach((inst, k) => {
        const d = document.createElement("div"); d.className = "t58k-pane-lbl";
        d.style.top = (H * (1 - (n - k) * h) + 3) + "px";
        d.textContent = indLabel(inst); el.appendChild(d); paneLabels.push(d);
      });
    }
    function indLabel(i) {
      const v = Object.keys(i.params).map((k) => i.params[k]).join(",");
      return IND_DEFS[i.type].label.split(" (")[0] + (v ? " (" + v + ")" : "");
    }

    function buildIndSeries(inst) {
      inst.series.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* ignore */ } });
      inst.series = [];
      if (!inst.visible) return;
      inst.calc = calc(inst, bars);
      const def = IND_DEFS[inst.type];
      inst.calc.lines.forEach((ln, idx) => {
        const base = def.pane ? { priceScaleId: "ind-" + inst.id, lastValueVisible: false, priceLineVisible: false } : { lastValueVisible: false, priceLineVisible: false };
        let s;
        if (ln.kind === "hist") s = chart.addHistogramSeries(Object.assign({}, base, { color: ln.color }));
        else s = chart.addLineSeries(Object.assign({}, base, { color: ln.color || inst.colors[0], lineWidth: ln.width || 1.5, lineStyle: ln.dash ? LW.LineStyle.Dashed : LW.LineStyle.Solid, crosshairMarkerVisible: false }));
        if (def.pane && idx === 0 && inst.calc.levels) {
          inst.calc.levels.forEach((lv) => s.createPriceLine({ price: lv, color: "rgba(127,127,127,.45)", lineWidth: 1, lineStyle: LW.LineStyle.Dotted, axisLabelVisible: false }));
        }
        inst.series.push(s);
      });
      setIndData(inst, 0);
    }
    function lineData(inst, ln, from) {
      const out = [];
      for (let i = from; i < bars.length; i++) {
        const v = ln.values[i];
        if (v === undefined || Number.isNaN(v)) { if (ln.gaps) out.push({ time: bars[i].time }); continue; }
        const pt = { time: bars[i].time, value: v };
        if (ln.kind === "hist") pt.color = v >= 0 ? alpha("#35e0b0", 0.6) : alpha("#ff6b6b", 0.6);
        out.push(pt);
      }
      return out;
    }
    function setIndData(inst, from) {
      if (!inst.visible || !inst.calc) return;
      inst.calc.lines.forEach((ln, k) => {
        const s = inst.series[k]; if (!s) return;
        const data = lineData(inst, ln, 0);
        if (from === 0) s.setData(data);
        else lineData(inst, ln, from).forEach((pt) => s.update(pt));
      });
    }
    function rebuildIndicators() { instances.forEach(buildIndSeries); applyLayout(); updateLegend(null); }

    // ---- volume
    function ensureVolume() {
      if (volume) { try { chart.removeSeries(volume); } catch (e) { /* ignore */ } volume = null; }
      if (!prefs.volume) return;
      const c = palette();
      volume = chart.addHistogramSeries({ priceFormat: { type: "volume" }, priceScaleId: "t58-vol", lastValueVisible: false, priceLineVisible: false, color: alpha(c.down, 0.4) });
      volume.setData(bars.map((b) => volPoint(b, c)));
    }
    function volPoint(b, c) { return { time: b.time, value: b.volume || 0, color: alpha(b.close >= b.open ? c.up : c.down, 0.4) }; }

    // ---- appearance
    function applyAppearance() {
      const c = palette();
      chart.applyOptions({
        layout: { background: { color: c.bg }, textColor: c.text },
        grid: { vertLines: { visible: prefs.grid === "both" || prefs.grid === "v", color: c.grid }, horzLines: { visible: prefs.grid === "both" || prefs.grid === "h", color: c.grid } },
        rightPriceScale: { borderColor: c.border },
        timeScale: { borderColor: c.border, rightOffset: prefs.rightOffset, barSpacing: prefs.barSpacing },
        crosshair: { mode: prefs.crosshair === "magnet" ? LW.CrosshairMode.Magnet : LW.CrosshairMode.Normal, vertLine: { color: c.text, style: LW.LineStyle.Dashed, labelBackgroundColor: "#2a2e39" }, horzLine: { color: c.text, style: LW.LineStyle.Dashed, labelBackgroundColor: "#2a2e39" } },
        localization: { timeFormatter: (t) => fmtTime(t, true) },
        watermark: { visible: !!(prefs.watermark && title), text: title, color: alpha(c.text.startsWith("#") ? c.text : "#9aa4b2", 0.09), fontSize: 46, horzAlign: "center", vertAlign: "center" },
      });
      chart.timeScale().applyOptions({ tickMarkFormatter: (t, type) => tickLabel(t, type) });
      el.style.background = c.bg === "transparent" ? "" : c.bg;
    }
    function tzDate(t) { return new Date(t * 1000); }
    function fmtTime(t, full) {
      const d = tzDate(t), utc = prefs.timeFormat === "utc";
      const g = (a, b) => (utc ? d[a]() : d[b]());
      const p = (n) => String(n).padStart(2, "0");
      return g("getUTCFullYear", "getFullYear") + "-" + p(g("getUTCMonth", "getMonth") + 1) + "-" + p(g("getUTCDate", "getDate")) + " " + p(g("getUTCHours", "getHours")) + ":" + p(g("getUTCMinutes", "getMinutes"));
    }
    function tickLabel(t, type) {
      const d = tzDate(t), utc = prefs.timeFormat === "utc";
      const g = (a, b) => (utc ? d[a]() : d[b]());
      const p = (n) => String(n).padStart(2, "0");
      const mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      if (type === 0) return String(g("getUTCFullYear", "getFullYear"));
      if (type === 1) return mon[g("getUTCMonth", "getMonth")];
      if (type === 2) return String(g("getUTCDate", "getDate"));
      return p(g("getUTCHours", "getHours")) + ":" + p(g("getUTCMinutes", "getMinutes"));
    }

    // ---- legend (OHLC + indicator values at the crosshair)
    function fmtN(v) { if (v == null || Number.isNaN(v)) return "-"; const a = Math.abs(v); return v.toFixed(a >= 1000 ? 2 : a >= 100 ? 2 : a >= 10 ? 3 : 5); }
    function idxOfTime(t) {
      let lo = 0, hi = bars.length - 1;
      while (lo <= hi) { const m = (lo + hi) >> 1; if (bars[m].time === t) return m; if (bars[m].time < t) lo = m + 1; else hi = m - 1; }
      return -1;
    }
    function updateLegend(idx) {
      if (!prefs.legend || cfg.legend === false) { legendEl.style.display = "none"; return; }
      legendEl.style.display = "";
      if (idx == null || idx < 0 || idx >= bars.length) idx = bars.length - 1;
      const b = bars[idx];
      let html = "<div><span class='ttl'>" + esc(title || "") + "</span></div>";
      if (b) {
        const prev = idx > 0 ? bars[idx - 1].close : null, ch = prev ? b.close - prev : 0, pc = prev ? ch / prev * 100 : 0;
        html += "<div><span class='mut'>O</span> " + fmtN(b.open) + " <span class='mut'>H</span> " + fmtN(b.high) + " <span class='mut'>L</span> " + fmtN(b.low) + " <span class='mut'>C</span> " + fmtN(b.close)
          + " <span class='" + (ch >= 0 ? "up" : "dn") + "'>" + (ch >= 0 ? "+" : "") + fmtN(ch) + " (" + (pc >= 0 ? "+" : "") + pc.toFixed(2) + "%)</span></div>";
        html += "<div class='mut'>" + fmtTime(b.time) + (b.volume ? " &middot; vol " + Math.round(b.volume).toLocaleString() : "") + "</div>";
      }
      instances.forEach((inst) => {
        if (!inst.visible || !inst.calc) return;
        const vals = inst.calc.lines.filter((l) => l.kind !== "hist").map((l) => "<span style='color:" + l.color + "'>" + fmtN(l.values[idx]) + "</span>").join(" ");
        html += "<div><span class='mut'>" + esc(indLabel(inst)) + "</span> " + vals + "</div>";
      });
      legendEl.innerHTML = html;
    }
    function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]); }
    chart.subscribeCrosshairMove((param) => {
      if (!param || param.time == null) { updateLegend(null); return; }
      updateLegend(idxOfTime(param.time));
    });

    // ---- data flow
    function setBars(list) {
      bars = Array.isArray(list) ? list : [];
      if (!main) makeMain();
      main.setData(mainData(bars));
      main.setMarkers(markers);
      ensureVolume();
      rebuildIndicators();
      reapplyPriceLines(); refreshHiLo();
      updateLegend(null); drawAnn();
      if (drawTools) drawTools.redraw();
    }
    function updateBar(bar) {
      if (!main) { setBars([bar]); return; }
      const last = bars.length ? bars[bars.length - 1] : null;
      if (last && bar.time === last.time) { bars[bars.length - 1] = bar; pending = Math.min(pending == null ? bars.length - 1 : pending, bars.length - 1); }
      else if (!last || bar.time > last.time) { bars.push(bar); pending = Math.min(pending == null ? bars.length - 1 : pending, bars.length - 1); }
      else return;    // older than the chart's last bar: ignore (a re-seed is setBars)
      if (!rafId) rafId = requestAnimationFrame(flush);
    }
    function flush() {
      rafId = 0;
      if (pending == null || !main) return;
      const from = pending; pending = null;
      if (prefs.type === "heikin") {
        const ha = heikin(bars); for (let i = from; i < ha.length; i++) main.update(ha[i]);
      } else for (let i = from; i < bars.length; i++) main.update(pointOf(bars[i]));
      if (volume) { const c = palette(); for (let i = from; i < bars.length; i++) volume.update(volPoint(bars[i], c)); }
      instances.forEach((inst) => { if (inst.visible) { inst.calc = calc(inst, bars); setIndData(inst, from); } });
      refreshLastPriceLine(); refreshHiLo(); updateLegend(null); drawAnn();
      if (drawTools) drawTools.redraw();
    }
    function refreshLastPriceLine() { /* native last-value line is handled by the series options */ }
    function refreshHiLo() {
      hiLoLines.forEach((l) => { try { main.removePriceLine(l); } catch (e) { /* ignore */ } }); hiLoLines = [];
      if (!prefs.hiLo || !bars.length || !main) return;
      let h = -Infinity, l = Infinity; bars.forEach((b) => { if (b.high > h) h = b.high; if (b.low < l) l = b.low; });
      const c = palette();
      hiLoLines.push(main.createPriceLine({ price: h, color: alpha(c.up, 0.7), lineWidth: 1, lineStyle: LW.LineStyle.Dotted, axisLabelVisible: true, title: "HIGH" }));
      hiLoLines.push(main.createPriceLine({ price: l, color: alpha(c.down, 0.7), lineWidth: 1, lineStyle: LW.LineStyle.Dotted, axisLabelVisible: true, title: "LOW" }));
    }
    function setMarkers(m) { markers = m || []; if (main) main.setMarkers(markers); }
    function reapplyPriceLines() { Object.keys(priceLineSets).forEach((k) => setPriceLines(k, priceLineSets[k])); }
    function setPriceLines(name, list) {
      priceLineSets[name] = list || [];
      (livePriceLines[name] || []).forEach((l) => { try { main && main.removePriceLine(l); } catch (e) { /* ignore */ } });
      livePriceLines[name] = [];
      if (!main) return;
      priceLineSets[name].forEach((pl) => {
        livePriceLines[name].push(main.createPriceLine({ price: pl.price, color: pl.color || "#9aa4b2", lineWidth: pl.width || 1,
          lineStyle: ({ solid: 0, dotted: 1, dashed: 2 })[pl.style || "dashed"], axisLabelVisible: pl.label !== false, title: pl.title || "" }));
      });
    }
    function setAnnotations(list) { annotations = list || []; drawAnn(); }

    // ---- drawing tools
    makeMain();
    applyAppearance();
    sizeAnn();
    if (global.T58ChartDrawingTools && cfg.drawToolbarEl !== null) {
      drawTools = global.T58ChartDrawingTools.attach(chart, main, el, cfg.drawToolbarEl || null, {
        storageKey: cfg.drawKey ? "t58-draw:" + cfg.drawKey : null, getBars: () => bars,
      });
    }

    // ---- public changes
    function setType(t) {
      prefs.type = t; savePart(prefs, ["type"]);
      makeMain(); if (bars.length) { main.setData(mainData(bars)); main.setMarkers(markers); }
      reapplyPriceLines(); refreshHiLo(); applyLayout(); drawAnn();
    }
    function setPref(k, v, rebuild) {
      prefs[k] = v; savePart(prefs, k === "candles" ? ["candles", "upColor", "downColor"] : [k]); applyAppearance();
      if (rebuild === "series") { makeMain(); if (bars.length) { main.setData(mainData(bars)); main.setMarkers(markers); } reapplyPriceLines(); refreshHiLo(); ensureVolume(); instances.forEach(buildIndSeries); }
      if (rebuild === "volume") ensureVolume();
      applyLayout(); updateLegend(null); drawAnn();
    }
    function addIndicator(type) {
      const def = IND_DEFS[type]; if (!def) return null;
      const inst = hydrate({ type, params: def.params, colors: [PALETTE[(instances.length) % PALETTE.length], PALETTE[(instances.length + 3) % PALETTE.length]] });
      instances.push(inst); persist(); buildIndSeries(inst); applyLayout(); updateLegend(null); return inst;
    }
    function removeIndicator(id) {
      const i = instances.findIndex((x) => x.id === id); if (i < 0) return;
      instances[i].series.forEach((s) => { try { chart.removeSeries(s); } catch (e) { /* ignore */ } });
      instances.splice(i, 1); persist(); applyLayout(); updateLegend(null);
    }
    function fit() { chart.timeScale().fitContent(); }
    function screenshot() {
      try {
        const cv = chart.takeScreenshot();
        const out = document.createElement("canvas"); out.width = cv.width; out.height = cv.height;
        const g = out.getContext("2d"); g.drawImage(cv, 0, 0);
        const r = el.getBoundingClientRect(), k = cv.width / r.width;
        g.drawImage(ann, 0, 0, ann.width, ann.height, 0, 0, cv.width, cv.height);
        const dc = el.querySelectorAll("canvas");
        dc.forEach((c) => { if (c !== ann && c.style.zIndex === "5") g.drawImage(c, 0, 0, c.width, c.height, 0, 0, cv.width, cv.height); });
        const a = document.createElement("a"); a.download = (title || "chart").replace(/[^\w.-]+/g, "_") + "_" + Date.now() + ".png"; a.href = out.toDataURL("image/png"); a.click();
      } catch (e) { try { window.alert("Screenshot is not supported by this build of the chart library."); } catch (e2) { /* ignore */ } }
    }

    // ---- controls UI
    function buildControls(host) {
      if (!host) return;
      host.classList.add("t58k-bar"); host.innerHTML = "";
      const mkBtn = (txt, ttl, fn) => { const b = document.createElement("button"); b.type = "button"; b.className = "t58k-btn"; b.innerHTML = txt; b.title = ttl || ""; b.addEventListener("click", fn); return b; };
      const pops = [];
      const closeAll = () => pops.forEach((p) => p.classList.remove("open"));
      document.addEventListener("click", (e) => { if (!host.contains(e.target)) closeAll(); });
      function popover(label, ttl, render) {
        const w = document.createElement("span"); w.className = "t58k-pop-wrap";
        const pop = document.createElement("div"); pop.className = "t58k-pop"; pops.push(pop);
        const b = mkBtn(label, ttl, (e) => { e.stopPropagation(); const o = pop.classList.contains("open"); closeAll(); if (!o) { render(pop); pop.classList.add("open"); } });
        pop.addEventListener("click", (e) => e.stopPropagation());
        w.appendChild(b); w.appendChild(pop); host.appendChild(w); return { pop, render };
      }
      const sel = (opts, cur, fn) => {
        const s = document.createElement("select");
        opts.forEach(([v, l]) => { const o = document.createElement("option"); o.value = v; o.textContent = l; if (String(v) === String(cur)) o.selected = true; s.appendChild(o); });
        s.addEventListener("change", () => fn(s.value)); return s;
      };
      const row = (label, control) => { const r = document.createElement("div"); r.className = "t58k-row"; const l = document.createElement("label"); l.textContent = label; r.appendChild(l); r.appendChild(control); return r; };
      const check = (cur, fn) => { const c = document.createElement("input"); c.type = "checkbox"; c.checked = !!cur; c.addEventListener("change", () => fn(c.checked)); return c; };

      // chart type
      const typeSel = sel([["candles", "Candles"], ["hollow", "Hollow candles"], ["bars", "OHLC bars"], ["line", "Line"], ["area", "Area"], ["baseline", "Baseline"], ["heikin", "Heikin-Ashi"]], prefs.type, setType);
      typeSel.className = "t58k-btn"; typeSel.title = "Chart type"; host.appendChild(typeSel);

      // indicators
      const ind = popover("ƒx Indicators", "Add / edit indicators", function render(pop) {
        pop.innerHTML = "";
        const h = (t) => { const e = document.createElement("h4"); e.textContent = t; pop.appendChild(e); };
        h("Active");
        if (!instances.length) { const p = document.createElement("div"); p.style.opacity = ".7"; p.textContent = "None yet -- pick one below."; pop.appendChild(p); }
        instances.forEach((inst) => {
          const box = document.createElement("div"); box.className = "t58k-ind";
          const top = document.createElement("div"); top.className = "t58k-row";
          const nm = document.createElement("span"); nm.className = "nm"; nm.textContent = IND_DEFS[inst.type].label.split(" (")[0];
          const right = document.createElement("span");
          const eye = document.createElement("button"); eye.type = "button"; eye.className = "t58k-x"; eye.style.color = "inherit"; eye.textContent = inst.visible ? "\u{1F441}" : "∅"; eye.title = "Show / hide";
          eye.addEventListener("click", () => { inst.visible = !inst.visible; persist(); buildIndSeries(inst); applyLayout(); updateLegend(null); render(pop); });
          const x = document.createElement("button"); x.type = "button"; x.className = "t58k-x"; x.textContent = "✕"; x.title = "Remove";
          x.addEventListener("click", () => { removeIndicator(inst.id); render(pop); });
          right.appendChild(eye); right.appendChild(x); top.appendChild(nm); top.appendChild(right); box.appendChild(top);
          const pr = document.createElement("div"); pr.className = "t58k-row"; pr.style.justifyContent = "flex-start"; pr.style.flexWrap = "wrap";
          Object.keys(inst.params).forEach((k) => {
            const lab = document.createElement("label"); lab.textContent = k;
            const inp = document.createElement("input"); inp.type = "number"; inp.value = inst.params[k]; inp.step = k === "mult" ? "0.1" : "1"; inp.min = k === "mult" ? "0.1" : "1";
            inp.addEventListener("change", () => { const v = parseFloat(inp.value); if (v > 0) { inst.params[k] = k === "mult" ? v : Math.round(v); persist(); buildIndSeries(inst); applyLayout(); updateLegend(null); } });
            pr.appendChild(lab); pr.appendChild(inp);
          });
          inst.colors.forEach((cl, ci) => {
            const c = document.createElement("input"); c.type = "color"; c.value = /^#[0-9a-f]{6}$/i.test(cl) ? cl : "#22d3ee"; c.title = "Colour " + (ci + 1);
            c.addEventListener("change", () => { inst.colors[ci] = c.value; persist(); buildIndSeries(inst); updateLegend(null); }); pr.appendChild(c);
          });
          box.appendChild(pr); pop.appendChild(box);
        });
        let g = "";
        Object.keys(IND_DEFS).forEach((k) => {
          const d = IND_DEFS[k]; if (d.group !== g) { g = d.group; h("Add: " + g); }
          const b = document.createElement("button"); b.type = "button"; b.className = "t58k-add"; b.textContent = "+ " + d.label;
          b.addEventListener("click", () => { addIndicator(k); render(pop); }); pop.appendChild(b);
        });
        h("Volume");
        pop.appendChild(row("Show volume", check(prefs.volume, (v) => setPref("volume", v, "volume"))));
      });

      // settings
      popover("⚙ Chart settings", "Theme, colours, grid, scales", function render(pop) {
        pop.innerHTML = "";
        const h = (t) => { const e = document.createElement("h4"); e.textContent = t; pop.appendChild(e); };
        h("Appearance");
        pop.appendChild(row("Theme", sel(Object.keys(THEMES).map((k) => [k, THEMES[k].label]), prefs.theme, (v) => setPref("theme", v, "series"))));
        pop.appendChild(row("Candle colours", sel(Object.keys(CANDLE_PRESETS).map((k) => [k, CANDLE_PRESETS[k].label]), prefs.candles, (v) => { prefs.upColor = null; prefs.downColor = null; setPref("candles", v, "series"); render(pop); })));
        const up = document.createElement("input"); up.type = "color"; up.value = palette().up; up.addEventListener("change", () => setPref("upColor", up.value, "series"));
        const dn = document.createElement("input"); dn.type = "color"; dn.value = palette().down; dn.addEventListener("change", () => setPref("downColor", dn.value, "series"));
        const cc = document.createElement("span"); cc.appendChild(up); cc.appendChild(dn); pop.appendChild(row("Custom up / down", cc));
        pop.appendChild(row("Grid", sel([["both", "Both"], ["h", "Horizontal"], ["v", "Vertical"], ["none", "None"]], prefs.grid, (v) => setPref("grid", v))));
        pop.appendChild(row("Crosshair", sel([["normal", "Free"], ["magnet", "Snap to candle"]], prefs.crosshair, (v) => setPref("crosshair", v))));
        pop.appendChild(row("Symbol watermark", check(prefs.watermark, (v) => setPref("watermark", v))));
        pop.appendChild(row("Legend (OHLC + values)", check(prefs.legend, (v) => setPref("legend", v))));
        h("Scales");
        pop.appendChild(row("Price axis", sel([["normal", "Normal"], ["log", "Logarithmic"], ["percent", "Percentage"], ["indexed", "Indexed to 100"]], prefs.priceMode, (v) => setPref("priceMode", v))));
        pop.appendChild(row("Auto-fit price", check(prefs.autoScale, (v) => setPref("autoScale", v))));
        pop.appendChild(row("Time zone", sel([["local", "Local"], ["utc", "UTC"]], prefs.timeFormat, (v) => { setPref("timeFormat", v); chart.timeScale().applyOptions({}); })));
        const bs = document.createElement("input"); bs.type = "number"; bs.min = 2; bs.max = 40; bs.value = prefs.barSpacing; bs.addEventListener("change", () => setPref("barSpacing", Math.max(2, Math.min(40, parseFloat(bs.value) || 8))));
        pop.appendChild(row("Bar spacing (px)", bs));
        const ro2 = document.createElement("input"); ro2.type = "number"; ro2.min = 0; ro2.max = 100; ro2.value = prefs.rightOffset; ro2.addEventListener("change", () => setPref("rightOffset", Math.max(0, Math.min(100, parseFloat(ro2.value) || 0))));
        pop.appendChild(row("Right margin (bars)", ro2));
        h("Lines");
        pop.appendChild(row("Last-price line", check(prefs.lastPrice, (v) => setPref("lastPrice", v, "series"))));
        pop.appendChild(row("High / low of loaded data", check(prefs.hiLo, (v) => { setPref("hiLo", v); refreshHiLo(); })));
        h("");
        const reset = mkBtn("Reset all chart settings", "", () => {
          Object.keys(DEFAULTS).forEach((k) => { prefs[k] = Array.isArray(DEFAULTS[k]) ? [] : DEFAULTS[k]; });
          instances.slice().forEach((i) => removeIndicator(i.id)); savePrefs(prefs);
          setPref("type", "candles", "series"); render(pop);
        });
        pop.appendChild(reset);
      });

      host.appendChild(mkBtn("⤢ Fit", "Fit all data in view", fit));
      host.appendChild(mkBtn("\u{1F4F7}", "Save chart as PNG", screenshot));
      host.appendChild(mkBtn("⛶", "Full screen chart", () => { el.parentElement && el.parentElement.classList.toggle("t58k-full"); }));
      return host;
    }

    const api = {
      chart, get series() { return main; }, get bars() { return bars; }, get prefs() { return prefs; }, get drawTools() { return drawTools; },
      flushNow() { if (rafId) { cancelAnimationFrame(rafId); flush(); } }, setBars, updateBar, setMarkers, setPriceLines, setAnnotations, fit, screenshot, buildControls,
      setTitle(t) { title = t || ""; applyAppearance(); updateLegend(null); },
      setDrawKey(k) { if (drawTools) drawTools.setStorageKey(k ? "t58-draw:" + k : null); },
      addIndicator, removeIndicator, setType,
      destroy() { try { annRo && annRo.disconnect(); chart.remove(); drawTools && drawTools.destroy(); } catch (e) { /* ignore */ } },
    };
    if (cfg.controlsEl) buildControls(cfg.controlsEl);
    if (cfg.title) api.setTitle(cfg.title);
    return api;
  }

  global.T58ChartKit = { create, IND_DEFS, THEMES };
})(window);
