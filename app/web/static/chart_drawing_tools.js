/*
 * T58 Chart Drawing Tools -- a dependency-free canvas overlay for TradingView's open-source Lightweight Charts (v4).
 * Shared by Interactive Replay and Live Monitor (via t58-chart-kit.js).
 *
 * Tools: cursor, horizontal line, vertical line, ray, trend line, extended line, arrow, parallel channel, rectangle,
 * ellipse, brush/path, fib retracement, fib extension, measure (ruler), text note, long / short position (R:R).
 *
 * What makes it usable rather than a toy:
 *   - every drawing lives in (bar-index, price) space, so it stays glued to the candles through pan / zoom, and can be
 *     drawn into the empty space to the right of the last bar (position tools need that);
 *   - click-drag OR click-click placement; Esc cancels; one-shot tools return to the cursor (hold Ctrl / "keep drawing"
 *     in the toolbar to stay in a tool);
 *   - select a drawing by clicking it, drag its handles to reshape it or its body to move it, Delete removes it;
 *   - a floating property bar (colour, width, line style, fill, lock, clone, delete) appears next to the selection;
 *   - magnet snaps to the nearest open/high/low/close of the bar under the cursor;
 *   - undo / redo (Ctrl+Z / Ctrl+Y), hide all, lock all, clear all;
 *   - drawings persist per storage key in localStorage (best-effort) and re-attach by candle time after a reload.
 *
 * API (backward compatible with the old 4-argument call):
 *   const tools = T58ChartDrawingTools.attach(chart, series, containerEl, toolbarEl, opts);
 *   tools.setTool(id|null)   tools.clear()   tools.redraw()   tools.setSeries(newSeries)
 *   tools.setStorageKey(key) tools.undo()    tools.redo()     tools.setMagnet(bool) tools.setHidden(bool)
 *   tools.getDrawings()      tools.setDrawings(list)          tools.destroy()
 *   opts: { storageKey, getBars: () => [{time,open,high,low,close}], onChange: fn, priceDecimals: n }
 */
(function (global) {
  "use strict";

  const CURSOR = "cursor";
  const GROUPS = [
    { id: "lines", label: "Lines", icon: "╱", tools: [
      { id: "trend", label: "Trend line", icon: "╱", pts: 2 },
      { id: "ray", label: "Ray", icon: "↦", pts: 2 },
      { id: "extended", label: "Extended line", icon: "↔", pts: 2 },
      { id: "hline", label: "Horizontal line", icon: "―", pts: 1 },
      { id: "vline", label: "Vertical line", icon: "│", pts: 1 },
      { id: "arrow", label: "Arrow", icon: "↗", pts: 2 },
      { id: "channel", label: "Parallel channel", icon: "☰", pts: 3 },
    ] },
    { id: "fib", label: "Fib & measure", icon: "F", tools: [
      { id: "fib", label: "Fib retracement", icon: "F", pts: 2 },
      { id: "fibext", label: "Fib extension", icon: "Fx", pts: 3 },
      { id: "measure", label: "Measure (ruler)", icon: "↔↕", pts: 2 },
    ] },
    { id: "shapes", label: "Shapes", icon: "▭", tools: [
      { id: "box", label: "Rectangle", icon: "▭", pts: 2 },
      { id: "ellipse", label: "Ellipse", icon: "◯", pts: 2 },
      { id: "path", label: "Brush / path", icon: "〰", pts: 0 },
      { id: "text", label: "Text note", icon: "T", pts: 1 },
    ] },
    { id: "trade", label: "Trade planning", icon: "⇅", tools: [
      { id: "long", label: "Long position", icon: "↑", pts: 2 },
      { id: "short", label: "Short position", icon: "↓", pts: 2 },
    ] },
  ];
  const TOOL_INDEX = {};
  GROUPS.forEach((g) => g.tools.forEach((t) => { TOOL_INDEX[t.id] = t; }));
  const DEFAULT_COLORS = { hline: "#35e0b0", vline: "#9aa4b2", ray: "#22d3ee", trend: "#22d3ee", extended: "#22d3ee", arrow: "#f0b429",
    channel: "#8b7cff", box: "#8b7cff", ellipse: "#8b7cff", long: "#35e0b0", short: "#ff6b6b", fib: "#f0b429", fibext: "#f0b429",
    path: "#e5e7eb", measure: "#60a5fa", text: "#e5e7eb" };
  const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
  const FIB_EXT_LEVELS = [0, 0.618, 1, 1.272, 1.618, 2.0, 2.618];
  const SWATCHES = ["#35e0b0", "#22d3ee", "#60a5fa", "#8b7cff", "#f0b429", "#ff6b6b", "#e5e7eb", "#9aa4b2"];
  const FILL_TYPES = new Set(["box", "ellipse", "channel", "measure"]);
  const HANDLE_R = 5, HIT_PX = 7, MAX_UNDO = 100;
  const STYLE_ID = "t58-draw-style";

  function injectStyle() {
    if (document.getElementById(STYLE_ID)) return;
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = [
      ".t58-draw-toolbar{display:flex;flex-wrap:wrap;align-items:center;gap:4px}",
      ".t58-draw-sep{width:1px;align-self:stretch;background:var(--border,rgba(255,255,255,.12));margin:0 3px}",
      ".t58-draw-grp{position:relative}",
      ".t58-draw-menu{position:absolute;top:100%;left:0;margin-top:4px;min-width:178px;background:var(--panel-2,#161a22);border:1px solid var(--border-light,rgba(255,255,255,.18));border-radius:8px;padding:4px;z-index:60;box-shadow:0 8px 24px rgba(0,0,0,.35);display:none}",
      ".t58-draw-menu.open{display:block}",
      ".t58-draw-menu button{display:flex;gap:8px;align-items:center;width:100%;text-align:left;background:transparent;border:0;color:var(--text,#e5e7eb);padding:6px 8px;border-radius:6px;font-size:12px;cursor:pointer}",
      ".t58-draw-menu button:hover{background:rgba(var(--teal-rgb,53,224,176),.14)}",
      ".t58-draw-menu button .ic{width:20px;text-align:center;font-weight:700;color:var(--teal,#35e0b0)}",
      ".t58-draw-btn.toggle.on{border-color:var(--teal,#35e0b0);background:rgba(var(--teal-rgb,53,224,176),.18);color:var(--teal,#35e0b0)}",
      ".t58-draw-props{position:absolute;z-index:40;display:none;gap:6px;align-items:center;background:var(--panel-2,#161a22);border:1px solid var(--border-light,rgba(255,255,255,.2));border-radius:8px;padding:5px 7px;box-shadow:0 6px 20px rgba(0,0,0,.4);font-size:11px;color:var(--text,#e5e7eb)}",
      ".t58-draw-props.show{display:flex}",
      ".t58-draw-props .sw{width:16px;height:16px;border-radius:50%;border:2px solid transparent;cursor:pointer;padding:0}",
      ".t58-draw-props .sw.sel{border-color:#fff}",
      ".t58-draw-props select,.t58-draw-props input[type=text]{background:var(--panel-3,#10131a);color:var(--text,#e5e7eb);border:1px solid var(--border-light,rgba(255,255,255,.2));border-radius:5px;font-size:11px;padding:2px 4px}",
      ".t58-draw-props input[type=color]{width:20px;height:20px;border:0;padding:0;background:none;cursor:pointer}",
      ".t58-draw-props button{background:var(--panel-3,#10131a);color:var(--text,#e5e7eb);border:1px solid var(--border-light,rgba(255,255,255,.2));border-radius:5px;font-size:11px;padding:3px 7px;cursor:pointer}",
      ".t58-draw-props button.danger{color:#ff6b6b}",
      ".t58-draw-props button.on{border-color:var(--teal,#35e0b0);color:var(--teal,#35e0b0)}",
    ].join("\n");
    document.head.appendChild(s);
  }

  function uid() { return "d" + Math.random().toString(36).slice(2, 9); }
  function clone(o) { return JSON.parse(JSON.stringify(o)); }

  function attach(chart, series, containerEl, toolbarEl, opts) {
    opts = opts || {};
    injectStyle();
    let mainSeries = series;
    let activeTool = null;           // null == cursor
    let keepDrawing = false;
    let magnet = false;
    let hidden = false;
    let lockedAll = false;
    let drawings = [];               // {id,type,pts:[{l,t,p}],style:{color,width,dash,fill},locked,text}
    let selectedId = null;
    let draft = null;                // drawing being placed
    let draftStage = 0;              // points committed so far (click-click mode)
    let dragState = null;            // {mode:'handle'|'move'|'place', ...}
    let hoverPos = null;
    let undoStack = [], redoStack = [];
    let storageKey = opts.storageKey || null;
    let destroyed = false;
    let rafPending = false;

    if (!containerEl.style.position || containerEl.style.position === "static") containerEl.style.position = "relative";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "position:absolute;left:0;top:0;pointer-events:none;z-index:5";
    containerEl.appendChild(canvas);
    const ctx = canvas.getContext("2d");
    const propsEl = document.createElement("div");
    propsEl.className = "t58-draw-props";
    containerEl.appendChild(propsEl);

    // ---------------------------------------------------------------- geometry / conversion
    const cssW = () => canvas.width / (window.devicePixelRatio || 1);
    const cssH = () => canvas.height / (window.devicePixelRatio || 1);
    function plotW() {
      try { const w = chart.timeScale().width(); if (w > 0) return w; } catch (e) { /* older build */ }
      return cssW() - 56;
    }
    function plotH() {
      try { if (typeof chart.paneSize === "function") { const s = chart.paneSize(); if (s && s.height > 0) return s.height; } } catch (e) { /* ignore */ }
      return cssH() - 26;
    }
    function barsNow() {
      try { if (opts.getBars) return opts.getBars() || []; } catch (e) { /* ignore */ }
      try { if (mainSeries && typeof mainSeries.data === "function") return mainSeries.data() || []; } catch (e) { /* ignore */ }
      return [];
    }
    function xOf(l) {
      const x = chart.timeScale().logicalToCoordinate(l);
      return x == null ? NaN : x;
    }
    function yOf(p) {
      const y = mainSeries.priceToCoordinate(p);
      return y == null ? NaN : y;
    }
    function lOfX(x) { const l = chart.timeScale().coordinateToLogical(x); return l == null ? NaN : l; }
    function pOfY(y) { const p = mainSeries.coordinateToPrice(y); return p == null ? NaN : p; }
    function timeAtLogical(l) {
      const bars = barsNow(), i = Math.round(l);
      return i >= 0 && i < bars.length ? bars[i].time : null;
    }
    function decimals(p) {
      if (typeof opts.priceDecimals === "number") return opts.priceDecimals;
      const a = Math.abs(p);
      return a >= 1000 ? 2 : a >= 100 ? 2 : a >= 10 ? 3 : 5;
    }
    function fmtP(p) { return Number(p).toFixed(decimals(p)); }

    function resize() {
      const r = containerEl.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(r.width * dpr));
      canvas.height = Math.max(1, Math.round(r.height * dpr));
      canvas.style.width = r.width + "px";
      canvas.style.height = r.height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      redraw();
    }
    let ro = null;
    if (typeof ResizeObserver !== "undefined") { ro = new ResizeObserver(resize); ro.observe(containerEl); }
    else window.addEventListener("resize", resize);

    // ---------------------------------------------------------------- point helpers
    function evtPoint(evt) {
      const r = canvas.getBoundingClientRect();
      const x = evt.clientX - r.left, y = evt.clientY - r.top;
      let l = lOfX(x), p = pOfY(y);
      if (magnet && !Number.isNaN(l) && !Number.isNaN(p)) {
        const bars = barsNow(), i = Math.round(l), b = bars[i];
        if (b) {
          const cands = [b.open, b.high, b.low, b.close];
          let best = null, bd = 1e9;
          cands.forEach((c) => { const d = Math.abs(yOf(c) - y); if (d < bd) { bd = d; best = c; } });
          if (best != null && bd < 22) { p = best; l = i; }
        }
      }
      return { x, y, l, p, t: Number.isNaN(l) ? null : timeAtLogical(l) };
    }
    const toPt = (e) => ({ l: e.l, t: e.t, p: e.p });
    function screenPts(d) { return d.pts.map((q) => ({ x: xOf(q.l), y: yOf(q.p) })); }
    const bad = (a) => a.some((q) => Number.isNaN(q.x) || Number.isNaN(q.y));

    // ---------------------------------------------------------------- undo / persistence
    function snapshot() {
      undoStack.push(JSON.stringify(drawings));
      if (undoStack.length > MAX_UNDO) undoStack.shift();
      redoStack = [];
    }
    function restore(json) {
      drawings = JSON.parse(json);
      if (!drawings.some((d) => d.id === selectedId)) selectedId = null;
      afterChange();
    }
    function undo() { if (!undoStack.length) return; redoStack.push(JSON.stringify(drawings)); restore(undoStack.pop()); }
    function redo() { if (!redoStack.length) return; undoStack.push(JSON.stringify(drawings)); restore(redoStack.pop()); }
    function afterChange() {
      save(); redraw(); updateProps(); updateToolbarState();
      if (typeof opts.onChange === "function") { try { opts.onChange(drawings); } catch (e) { /* ignore */ } }
    }
    function save() {
      if (!storageKey) return;
      try {
        const keep = drawings.filter((d) => d.type !== "measure");
        localStorage.setItem(storageKey, JSON.stringify({ v: 2, hidden, items: keep }));
      } catch (e) { /* storage unavailable */ }
    }
    function resolveSaved(items) {
      // Re-anchor by candle time: logical indices change when a different history window is loaded.
      const bars = barsNow();
      const idxOf = {};
      bars.forEach((b, i) => { idxOf[b.time] = i; });
      return items.map((d) => {
        d.pts.forEach((q) => { if (q.t != null && idxOf[q.t] != null) q.l = idxOf[q.t]; });
        return d;
      });
    }
    function load() {
      if (!storageKey) return;
      try {
        const raw = localStorage.getItem(storageKey);
        if (!raw) { drawings = []; return; }
        const o = JSON.parse(raw);
        drawings = resolveSaved(Array.isArray(o.items) ? o.items : []);
        hidden = !!o.hidden;
      } catch (e) { drawings = []; }
    }

    // ---------------------------------------------------------------- drawing
    function lineDash(d, isDraft) {
      if (isDraft) return [4, 3];
      return d.style.dash === "dashed" ? [7, 4] : d.style.dash === "dotted" ? [2, 3] : [];
    }
    function extendLine(a, b, toRight, toLeft) {
      const dx = b.x - a.x, dy = b.y - a.y, W = cssW();
      if (Math.abs(dx) < 1e-6) return [a, b];
      const k = dy / dx;
      const x0 = toLeft ? 0 : a.x, x1 = toRight ? W : b.x;
      return [{ x: x0, y: a.y + (x0 - a.x) * k }, { x: x1, y: a.y + (x1 - a.x) * k }];
    }
    function tag(text, x, y, color, align) {
      ctx.save();
      ctx.font = "11px ui-monospace, Menlo, monospace";
      const w = ctx.measureText(text).width + 8;
      const bx = align === "right" ? x - w : x;
      ctx.fillStyle = "rgba(10,12,18,.78)"; ctx.fillRect(bx, y - 8, w, 16);
      ctx.fillStyle = color; ctx.textBaseline = "middle"; ctx.fillText(text, bx + 4, y);
      ctx.restore();
    }

    function drawOne(d, isDraft) {
      const pts = screenPts(d);
      if (bad(pts) && d.type !== "path") return;
      const c = d.style.color, sel = d.id === selectedId && !isDraft;
      ctx.save();
      ctx.beginPath(); ctx.rect(0, 0, plotW(), plotH()); ctx.clip();
      ctx.lineWidth = d.style.width || 1.5;
      ctx.strokeStyle = c; ctx.fillStyle = c;
      ctx.setLineDash(lineDash(d, isDraft));
      ctx.font = "11px ui-monospace, Menlo, monospace";
      const T = d.type, W = cssW();
      if (T === "hline") {
        ctx.beginPath(); ctx.moveTo(0, pts[0].y); ctx.lineTo(W, pts[0].y); ctx.stroke();
        ctx.setLineDash([]); tag(fmtP(d.pts[0].p), plotW() - 4, pts[0].y, c, "right");
      } else if (T === "vline") {
        ctx.beginPath(); ctx.moveTo(pts[0].x, 0); ctx.lineTo(pts[0].x, cssH()); ctx.stroke();
      } else if ((T === "trend" || T === "ray" || T === "extended" || T === "arrow") && pts.length >= 2) {
        const seg = T === "ray" ? extendLine(pts[0], pts[1], true, false) : T === "extended" ? extendLine(pts[0], pts[1], true, true) : [pts[0], pts[1]];
        ctx.beginPath(); ctx.moveTo(seg[0].x, seg[0].y); ctx.lineTo(seg[1].x, seg[1].y); ctx.stroke();
        if (T === "arrow") {
          const a = Math.atan2(pts[1].y - pts[0].y, pts[1].x - pts[0].x), s = 11 + (d.style.width || 1.5) * 2;
          ctx.setLineDash([]); ctx.beginPath(); ctx.moveTo(pts[1].x, pts[1].y);
          ctx.lineTo(pts[1].x - s * Math.cos(a - 0.4), pts[1].y - s * Math.sin(a - 0.4));
          ctx.lineTo(pts[1].x - s * Math.cos(a + 0.4), pts[1].y - s * Math.sin(a + 0.4));
          ctx.closePath(); ctx.fill();
        }
      } else if (T === "path" && pts.length >= 1) {
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
      } else if (T === "box" && pts.length >= 2) {
        const x = Math.min(pts[0].x, pts[1].x), y = Math.min(pts[0].y, pts[1].y), w = Math.abs(pts[1].x - pts[0].x), h = Math.abs(pts[1].y - pts[0].y);
        ctx.globalAlpha = d.style.fill == null ? 0.12 : d.style.fill; ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1; ctx.strokeRect(x, y, w, h);
      } else if (T === "ellipse" && pts.length >= 2) {
        const cx = (pts[0].x + pts[1].x) / 2, cy = (pts[0].y + pts[1].y) / 2;
        ctx.beginPath(); ctx.ellipse(cx, cy, Math.abs(pts[1].x - pts[0].x) / 2, Math.abs(pts[1].y - pts[0].y) / 2, 0, 0, Math.PI * 2);
        ctx.globalAlpha = d.style.fill == null ? 0.1 : d.style.fill; ctx.fill(); ctx.globalAlpha = 1; ctx.stroke();
      } else if (T === "channel" && pts.length >= 2) {
        const p3 = pts[2] || { x: pts[1].x, y: pts[1].y + 40 };
        const off = p3.y - (pts[0].y + (p3.x - pts[0].x) * ((pts[1].y - pts[0].y) / ((pts[1].x - pts[0].x) || 1e-6)));
        const a2 = { x: pts[0].x, y: pts[0].y + off }, b2 = { x: pts[1].x, y: pts[1].y + off };
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(a2.x, a2.y); ctx.lineTo(b2.x, b2.y); ctx.stroke();
        ctx.setLineDash([3, 4]); ctx.globalAlpha = 0.6;
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y + off / 2); ctx.lineTo(pts[1].x, pts[1].y + off / 2); ctx.stroke();
        ctx.globalAlpha = d.style.fill == null ? 0.08 : d.style.fill; ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.lineTo(b2.x, b2.y); ctx.lineTo(a2.x, a2.y); ctx.closePath(); ctx.fill();
      } else if (T === "fib" && pts.length >= 2) {
        drawFib(d, pts, FIB_LEVELS, false);
      } else if (T === "fibext" && pts.length >= 3) {
        drawFib(d, pts, FIB_EXT_LEVELS, true);
      } else if (T === "fibext" && pts.length === 2) {
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke();
      } else if (T === "measure" && pts.length >= 2) {
        drawMeasure(d, pts);
      } else if (T === "text") {
        ctx.setLineDash([]); ctx.font = "13px system-ui, sans-serif";
        const txt = d.text || "Text";
        const w = ctx.measureText(txt).width + 12;
        ctx.fillStyle = "rgba(10,12,18,.72)"; ctx.fillRect(pts[0].x, pts[0].y - 11, w, 22);
        ctx.strokeStyle = c; ctx.lineWidth = 1; ctx.strokeRect(pts[0].x, pts[0].y - 11, w, 22);
        ctx.fillStyle = c; ctx.textBaseline = "middle"; ctx.fillText(txt, pts[0].x + 6, pts[0].y);
      } else if ((T === "long" || T === "short") && pts.length >= 2) {
        drawPosition(d, pts);
      }
      ctx.restore();
      if (sel) drawHandles(d);
    }

    function posGeom(d, pts) {
      const isLong = d.type === "long";
      const p0 = d.pts[0].p, p1 = d.pts[1].p;
      const p2 = d.pts[2] ? d.pts[2].p : p0 - (p1 - p0);
      const x0 = Math.min(pts[0].x, pts[1].x), x1 = Math.max(Math.max(pts[0].x, pts[1].x), x0 + 70);
      return { isLong, p0, p1, p2, x0, x1, ye: yOf(p0), yt: yOf(p1), ys: yOf(p2) };
    }
    function drawPosition(d, pts) {
      const g = posGeom(d, pts);
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = "#35e0b0"; ctx.fillRect(g.x0, Math.min(g.ye, g.yt), g.x1 - g.x0, Math.abs(g.ye - g.yt));
      ctx.fillStyle = "#ff6b6b"; ctx.fillRect(g.x0, Math.min(g.ye, g.ys), g.x1 - g.x0, Math.abs(g.ye - g.ys));
      ctx.globalAlpha = 1;
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = "#e5e7eb"; ctx.beginPath(); ctx.moveTo(g.x0, g.ye); ctx.lineTo(g.x1, g.ye); ctx.stroke();
      ctx.strokeStyle = "#35e0b0"; ctx.beginPath(); ctx.moveTo(g.x0, g.yt); ctx.lineTo(g.x1, g.yt); ctx.stroke();
      ctx.strokeStyle = "#ff6b6b"; ctx.beginPath(); ctx.moveTo(g.x0, g.ys); ctx.lineTo(g.x1, g.ys); ctx.stroke();
      const reward = Math.abs(g.p1 - g.p0), risk = Math.abs(g.p0 - g.p2);
      const rr = risk > 0 ? reward / risk : 0;
      const pct = (v) => (g.p0 ? (v / g.p0 * 100).toFixed(2) + "%" : "");
      tag((g.isLong ? "LONG " : "SHORT ") + fmtP(g.p0) + "   R:R " + rr.toFixed(2), g.x0 + 2, g.ye, "#e5e7eb");
      tag("target " + fmtP(g.p1) + " (" + pct(reward) + ")", g.x0 + 2, g.yt + (g.yt < g.ye ? -10 : 10), "#35e0b0");
      tag("stop " + fmtP(g.p2) + " (" + pct(risk) + ")", g.x0 + 2, g.ys + (g.ys < g.ye ? -10 : 10), "#ff6b6b");
    }

    function drawFib(d, pts, levels, isExt) {
      const a = d.pts[0].p, b = d.pts[1].p, c3 = d.pts[2] ? d.pts[2].p : null;
      const xl = Math.min(pts[0].x, pts[1].x), xr = Math.max(pts[0].x, pts[1].x, pts[2] ? pts[2].x : -1e9) + 60;
      ctx.setLineDash([]);
      ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke(); ctx.globalAlpha = 1;
      let prevY = null;
      levels.forEach((lvl, i) => {
        const pr = isExt ? (c3 != null ? c3 : b) + (b - a) * lvl : a + (b - a) * lvl;
        const y = yOf(pr);
        if (Number.isNaN(y)) return;
        ctx.globalAlpha = (lvl === 0 || lvl === 1) ? 0.95 : 0.6;
        ctx.beginPath(); ctx.moveTo(xl, y); ctx.lineTo(xr, y); ctx.stroke();
        if (prevY != null && i % 2 === 0) { ctx.globalAlpha = 0.06; ctx.fillRect(xl, Math.min(y, prevY), xr - xl, Math.abs(y - prevY)); }
        ctx.globalAlpha = 1;
        tag(lvl.toFixed(3) + "  " + fmtP(pr), xl + 2, y - 8, d.style.color);
        prevY = y;
      });
    }

    function drawMeasure(d, pts) {
      const up = d.pts[1].p >= d.pts[0].p;
      const col = up ? "#35e0b0" : "#ff6b6b";
      const x = Math.min(pts[0].x, pts[1].x), y = Math.min(pts[0].y, pts[1].y), w = Math.abs(pts[1].x - pts[0].x), h = Math.abs(pts[1].y - pts[0].y);
      ctx.globalAlpha = 0.14; ctx.fillStyle = col; ctx.fillRect(x, y, w, h); ctx.globalAlpha = 1;
      ctx.strokeStyle = col; ctx.setLineDash([4, 3]); ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
      const dp = d.pts[1].p - d.pts[0].p, pct = d.pts[0].p ? dp / d.pts[0].p * 100 : 0;
      const nb = Math.round(Math.abs(d.pts[1].l - d.pts[0].l));
      let dur = "";
      const t0 = d.pts[0].t, t1 = d.pts[1].t;
      if (t0 != null && t1 != null) {
        const s = Math.abs(t1 - t0); dur = s >= 86400 ? (s / 86400).toFixed(1) + "d" : s >= 3600 ? (s / 3600).toFixed(1) + "h" : Math.round(s / 60) + "m";
      }
      const label = (dp >= 0 ? "+" : "") + fmtP(dp) + " (" + (pct >= 0 ? "+" : "") + pct.toFixed(2) + "%)  " + nb + " bars" + (dur ? " · " + dur : "");
      tag(label, x + w / 2 - 70, up ? y - 14 : y + h + 14, col);
    }

    function handlePoints(d) {
      const pts = screenPts(d);
      if (bad(pts)) return [];
      const out = pts.map((q, i) => ({ x: q.x, y: q.y, i, kind: "pt" }));
      if (d.type === "box" || d.type === "ellipse" || d.type === "measure") {
        out.push({ x: pts[0].x, y: pts[1].y, kind: "corner", ci: "a" });
        out.push({ x: pts[1].x, y: pts[0].y, kind: "corner", ci: "b" });
      }
      if (d.type === "channel" && pts.length === 2) { /* third handle appears after placement */ }
      if ((d.type === "long" || d.type === "short") && pts.length >= 2) {
        const g = posGeom(d, pts);
        out.length = 0;
        out.push({ x: g.x0, y: g.ye, i: 0, kind: "pt" });
        out.push({ x: g.x1, y: g.yt, i: 1, kind: "pt" });
        out.push({ x: g.x1, y: g.ys, i: 2, kind: "pos-stop" });
      }
      return out;
    }
    function drawHandles(d) {
      ctx.save(); ctx.setLineDash([]);
      handlePoints(d).forEach((h) => {
        ctx.beginPath(); ctx.arc(h.x, h.y, HANDLE_R, 0, Math.PI * 2);
        ctx.fillStyle = "#0b0e14"; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = d.style.color; ctx.stroke();
      });
      ctx.restore();
    }

    function redraw() {
      if (destroyed) return;
      ctx.clearRect(0, 0, cssW(), cssH());
      if (hidden) return;
      for (const d of drawings) drawOne(d, false);
      if (draft) drawOne(draft, true);
    }
    function scheduleRedraw() {
      if (rafPending) return;
      rafPending = true;
      requestAnimationFrame(() => { rafPending = false; redraw(); positionProps(); });
    }

    // ---------------------------------------------------------------- hit testing
    function distSeg(px, py, a, b) {
      const dx = b.x - a.x, dy = b.y - a.y, L = dx * dx + dy * dy;
      let t = L ? ((px - a.x) * dx + (py - a.y) * dy) / L : 0;
      t = Math.max(0, Math.min(1, t));
      return Math.hypot(px - (a.x + t * dx), py - (a.y + t * dy));
    }
    function hitHandle(d, x, y) {
      const hs = handlePoints(d);
      for (const h of hs) if (Math.hypot(h.x - x, h.y - y) <= HANDLE_R + 4) return h;
      return null;
    }
    function hitDrawing(d, x, y) {
      const pts = screenPts(d);
      if (bad(pts) && d.type !== "path") return false;
      const T = d.type;
      if (T === "hline") return Math.abs(y - pts[0].y) <= HIT_PX;
      if (T === "vline") return Math.abs(x - pts[0].x) <= HIT_PX;
      if (T === "trend" || T === "arrow") return distSeg(x, y, pts[0], pts[1]) <= HIT_PX;
      if (T === "ray" || T === "extended") { const s = extendLine(pts[0], pts[1], true, T === "extended"); return distSeg(x, y, s[0], s[1]) <= HIT_PX; }
      if (T === "path") { for (let i = 1; i < pts.length; i++) if (distSeg(x, y, pts[i - 1], pts[i]) <= HIT_PX) return true; return false; }
      if (T === "box" || T === "measure") {
        const xa = Math.min(pts[0].x, pts[1].x), xb = Math.max(pts[0].x, pts[1].x), ya = Math.min(pts[0].y, pts[1].y), yb = Math.max(pts[0].y, pts[1].y);
        return x >= xa - 3 && x <= xb + 3 && y >= ya - 3 && y <= yb + 3;
      }
      if (T === "ellipse") {
        const cx = (pts[0].x + pts[1].x) / 2, cy = (pts[0].y + pts[1].y) / 2, rx = Math.abs(pts[1].x - pts[0].x) / 2 + 3, ry = Math.abs(pts[1].y - pts[0].y) / 2 + 3;
        return rx > 0 && ry > 0 && ((x - cx) ** 2) / (rx * rx) + ((y - cy) ** 2) / (ry * ry) <= 1;
      }
      if (T === "channel") {
        if (pts.length < 2) return false;
        const p3 = pts[2] || { x: pts[1].x, y: pts[1].y + 40 };
        const off = p3.y - (pts[0].y + (p3.x - pts[0].x) * ((pts[1].y - pts[0].y) / ((pts[1].x - pts[0].x) || 1e-6)));
        return distSeg(x, y, pts[0], pts[1]) <= HIT_PX || distSeg(x, y, { x: pts[0].x, y: pts[0].y + off }, { x: pts[1].x, y: pts[1].y + off }) <= HIT_PX;
      }
      if (T === "fib" || T === "fibext") {
        const xl = Math.min(pts[0].x, pts[1].x), xr = Math.max(pts[0].x, pts[1].x, pts[2] ? pts[2].x : -1e9) + 60;
        if (x < xl - 4 || x > xr + 4) return false;
        const lv = T === "fib" ? FIB_LEVELS : FIB_EXT_LEVELS, a = d.pts[0].p, b = d.pts[1].p, c3 = d.pts[2] ? d.pts[2].p : b;
        return lv.some((l) => Math.abs(yOf(T === "fib" ? a + (b - a) * l : c3 + (b - a) * l) - y) <= HIT_PX) || distSeg(x, y, pts[0], pts[1]) <= HIT_PX;
      }
      if (T === "text") return x >= pts[0].x - 2 && x <= pts[0].x + 160 && Math.abs(y - pts[0].y) <= 13;
      if (T === "long" || T === "short") {
        const g = posGeom(d, pts);
        const ya = Math.min(g.ye, g.yt, g.ys), yb = Math.max(g.ye, g.yt, g.ys);
        return x >= g.x0 - 2 && x <= g.x1 + 2 && y >= ya - 2 && y <= yb + 2;
      }
      return false;
    }
    function pickAt(x, y) {
      for (let i = drawings.length - 1; i >= 0; i--) if (hitDrawing(drawings[i], x, y)) return drawings[i];
      return null;
    }

    // ---------------------------------------------------------------- creation
    function newDrawing(type, pts) {
      return { id: uid(), type, pts, locked: false, text: null,
        style: { color: DEFAULT_COLORS[type] || "#22d3ee", width: type === "long" || type === "short" ? 1.5 : 1.5, dash: "solid", fill: null } };
    }
    function neededPts(type) { return type === "path" ? 99 : (TOOL_INDEX[type] ? TOOL_INDEX[type].pts : 2); }
    function commitDraft() {
      if (!draft) return;
      const d = draft; draft = null; draftStage = 0; dragState = null;
      let ok = true;
      if (d.type === "path") ok = d.pts.length >= 2;
      else if (d.type === "channel" && d.pts.length < 3) ok = false;
      else if (d.type === "fibext" && d.pts.length < 3) ok = false;
      if ((d.type === "long" || d.type === "short") && d.pts.length === 2) {
        // stop defaults to the mirror of the target; becomes an independent, draggable third point
        d.pts.push({ l: d.pts[1].l, t: d.pts[1].t, p: d.pts[0].p - (d.pts[1].p - d.pts[0].p) });
      }
      if (ok) {
        snapshot(); drawings.push(d); selectedId = d.id;
        if (d.type === "text") editText(d);
      }
      if (!keepDrawing) setTool(null);
      afterChange();
    }
    function cancelDraft() { draft = null; draftStage = 0; dragState = null; redraw(); }

    function onCanvasDown(evt) {
      if (!activeTool || evt.button !== 0) return;
      const e = evtPoint(evt);
      if (Number.isNaN(e.l) || Number.isNaN(e.p)) return;
      evt.preventDefault(); evt.stopPropagation();
      const need = neededPts(activeTool);
      if (activeTool === "path") {
        if (!draft) draft = newDrawing("path", []);
        draft.pts.push(toPt(e)); redraw(); return;
      }
      if (!draft) {
        draft = newDrawing(activeTool, [toPt(e)]);
        draftStage = 1;
        if (need === 1) { commitDraft(); return; }
        draft.pts.push(toPt(e));        // rubber-band end point follows the pointer
        dragState = { mode: "place", sx: e.x, sy: e.y, moved: false };
        return;
      }
      // click-click mode: this click fixes the point currently following the pointer
      draft.pts[draftStage] = toPt(e);
      draftStage += 1;
      if (draftStage >= need) { commitDraft(); return; }
      draft.pts.push(toPt(e));
      redraw();
    }
    function onCanvasMove(evt) {
      const e = evtPoint(evt); hoverPos = e;
      if (!draft || Number.isNaN(e.l) || Number.isNaN(e.p)) return;
      if (activeTool === "path") { return; }
      const idx = draft.pts.length - 1;
      draft.pts[idx] = toPt(e);
      if (dragState && dragState.mode === "place" && Math.hypot(e.x - dragState.sx, e.y - dragState.sy) > 4) dragState.moved = true;
      scheduleRedraw();
    }
    function onCanvasUp(evt) {
      if (!draft || !dragState || dragState.mode !== "place") return;
      const e = evtPoint(evt);
      if (dragState.moved) {              // click-drag-release placement of the 2nd point
        draft.pts[1] = toPt(e);
        draftStage = 2; dragState = null;
        if (draftStage >= neededPts(draft.type)) commitDraft();
        else { draft.pts.push(toPt(e)); redraw(); }
      } else { dragState = null; }        // plain click: stay in click-click mode, 2nd click will place
    }
    function onCanvasDbl() { if (draft && draft.type === "path") commitDraft(); }

    // ---------------------------------------------------------------- selection / editing (cursor mode)
    function onContainerDown(evt) {
      if (activeTool || evt.button !== 0 || hidden) return;
      if (evt.target === propsEl || propsEl.contains(evt.target)) return;
      const r = canvas.getBoundingClientRect(), x = evt.clientX - r.left, y = evt.clientY - r.top;
      if (x > plotW() || y > plotH()) return;
      const sel = drawings.find((d) => d.id === selectedId);
      if (sel && !sel.locked && !lockedAll) {
        const h = hitHandle(sel, x, y);
        if (h) { startEdit(evt, sel, { mode: "handle", h }); return; }
      }
      const hit = pickAt(x, y);
      if (hit) {
        selectedId = hit.id; redraw(); updateProps();
        if (!hit.locked && !lockedAll) startEdit(evt, hit, { mode: "move" });
        else { evt.stopImmediatePropagation(); evt.preventDefault(); }
        return;
      }
      if (selectedId) { dropTemporary(); selectedId = null; redraw(); updateProps(); }
    }
    function dropTemporary() {
      const i = drawings.findIndex((d) => d.id === selectedId && d.type === "measure");
      if (i >= 0) drawings.splice(i, 1);
    }
    function startEdit(evt, d, st) {
      evt.stopImmediatePropagation(); evt.preventDefault();
      const e = evtPoint(evt);
      dragState = Object.assign({ d, before: JSON.stringify(drawings), sx: e.x, sy: e.y, l0: e.l, p0: e.p, orig: clone(d.pts), moved: false }, st);
      window.addEventListener("mousemove", onWinMove, true);
      window.addEventListener("mouseup", onWinUp, true);
    }
    function onWinMove(evt) {
      if (!dragState || dragState.mode === "place") return;
      const e = evtPoint(evt);
      if (Number.isNaN(e.l) || Number.isNaN(e.p)) return;
      const s = dragState, d = s.d;
      if (Math.hypot(e.x - s.sx, e.y - s.sy) > 2) s.moved = true;
      if (s.mode === "move") {
        const dl = e.l - s.l0, dp = e.p - s.p0;
        d.pts = s.orig.map((q) => { const l = q.l + dl; return { l, t: timeAtLogical(l), p: q.p + dp }; });
      } else if (s.mode === "handle") {
        const h = s.h;
        if (h.kind === "pt" || h.kind === "pos-stop") {
          const i = h.i;
          if ((d.type === "long" || d.type === "short") && i === 1) {
            d.pts[1] = { l: e.l, t: e.t, p: e.p };
          } else if ((d.type === "long" || d.type === "short") && i === 2) {
            d.pts[2] = { l: d.pts[1].l, t: d.pts[1].t, p: e.p };
          } else d.pts[i] = { l: e.l, t: e.t, p: e.p };
        } else if (h.kind === "corner") {
          if (h.ci === "a") { d.pts[0] = { l: d.pts[0].l, t: d.pts[0].t, p: d.pts[0].p }; d.pts[0].l = e.l; d.pts[0].t = e.t; d.pts[1].p = e.p; }
          else { d.pts[1].l = e.l; d.pts[1].t = e.t; d.pts[0].p = e.p; }
        }
      }
      scheduleRedraw();
    }
    function onWinUp() {
      window.removeEventListener("mousemove", onWinMove, true);
      window.removeEventListener("mouseup", onWinUp, true);
      if (dragState && dragState.moved) {
        undoStack.push(dragState.before); if (undoStack.length > MAX_UNDO) undoStack.shift(); redoStack = [];
        afterChange();
      }
      dragState = null;
    }
    function onContainerHover(evt) {
      if (activeTool || dragState || hidden) return;
      const r = canvas.getBoundingClientRect(), x = evt.clientX - r.left, y = evt.clientY - r.top;
      const over = x <= plotW() && y <= plotH() && pickAt(x, y);
      containerEl.style.cursor = over ? "pointer" : "";
    }

    function editText(d) {
      let v;
      try { v = window.prompt("Text note", d.text || ""); } catch (e) { v = null; }
      if (v != null) { snapshot(); d.text = v; afterChange(); }
    }
    function onDblClickContainer(evt) {
      if (activeTool) return;
      const r = canvas.getBoundingClientRect(), hit = pickAt(evt.clientX - r.left, evt.clientY - r.top);
      if (hit && hit.type === "text" && !hit.locked) editText(hit);
    }

    // ---------------------------------------------------------------- property bar
    function selected() { return drawings.find((d) => d.id === selectedId) || null; }
    function updateProps() {
      const d = selected();
      if (!d || hidden) { propsEl.classList.remove("show"); return; }
      propsEl.innerHTML = "";
      const mk = (tagName, cls, txt) => { const n = document.createElement(tagName); if (cls) n.className = cls; if (txt != null) n.textContent = txt; return n; };
      const apply = (fn) => { snapshot(); fn(d); afterChange(); };
      SWATCHES.forEach((c) => {
        const b = mk("button", "sw" + (d.style.color === c ? " sel" : "")); b.type = "button"; b.style.background = c; b.title = c;
        b.addEventListener("click", () => apply((x) => { x.style.color = c; }));
        propsEl.appendChild(b);
      });
      const col = mk("input"); col.type = "color"; col.value = /^#[0-9a-f]{6}$/i.test(d.style.color) ? d.style.color : "#22d3ee"; col.title = "Custom colour";
      col.addEventListener("change", () => apply((x) => { x.style.color = col.value; }));
      propsEl.appendChild(col);
      const w = mk("select"); [1, 1.5, 2, 3, 4].forEach((n) => { const o = mk("option", null, n + "px"); o.value = n; if (Number(d.style.width) === n) o.selected = true; w.appendChild(o); });
      w.title = "Line width"; w.addEventListener("change", () => apply((x) => { x.style.width = Number(w.value); }));
      propsEl.appendChild(w);
      const ls = mk("select"); ["solid", "dashed", "dotted"].forEach((n) => { const o = mk("option", null, n); o.value = n; if (d.style.dash === n) o.selected = true; ls.appendChild(o); });
      ls.title = "Line style"; ls.addEventListener("change", () => apply((x) => { x.style.dash = ls.value; }));
      propsEl.appendChild(ls);
      if (FILL_TYPES.has(d.type)) {
        const f = mk("input"); f.type = "range"; f.min = 0; f.max = 60; f.value = Math.round((d.style.fill == null ? 0.12 : d.style.fill) * 100); f.title = "Fill opacity"; f.style.width = "56px";
        f.addEventListener("change", () => apply((x) => { x.style.fill = Number(f.value) / 100; }));
        propsEl.appendChild(f);
      }
      if (d.type === "text") { const eb = mk("button", null, "Edit"); eb.type = "button"; eb.addEventListener("click", () => editText(d)); propsEl.appendChild(eb); }
      const lk = mk("button", d.locked ? "on" : "", d.locked ? "Locked" : "Lock"); lk.type = "button"; lk.title = "Lock this drawing in place";
      lk.addEventListener("click", () => apply((x) => { x.locked = !x.locked; }));
      propsEl.appendChild(lk);
      const cl = mk("button", null, "Clone"); cl.type = "button"; cl.title = "Duplicate (Ctrl+D)"; cl.addEventListener("click", () => cloneSelected());
      propsEl.appendChild(cl);
      const del = mk("button", "danger", "Delete"); del.type = "button"; del.title = "Delete (Del)"; del.addEventListener("click", () => deleteSelected());
      propsEl.appendChild(del);
      propsEl.classList.add("show");
      positionProps();
    }
    function positionProps() {
      const d = selected();
      if (!d || !propsEl.classList.contains("show")) return;
      const pts = screenPts(d).filter((q) => !Number.isNaN(q.x) && !Number.isNaN(q.y));
      if (!pts.length) { propsEl.style.visibility = "hidden"; return; }
      propsEl.style.visibility = "visible";
      const minY = Math.min.apply(null, pts.map((q) => q.y)), avgX = pts.reduce((s, q) => s + q.x, 0) / pts.length;
      const bw = propsEl.offsetWidth || 340;
      propsEl.style.left = Math.max(4, Math.min(plotW() - bw - 4, avgX - bw / 2)) + "px";
      propsEl.style.top = Math.max(4, minY - 42) + "px";
    }
    function deleteSelected() {
      if (!selectedId) return;
      snapshot(); drawings = drawings.filter((d) => d.id !== selectedId); selectedId = null; afterChange();
    }
    function cloneSelected() {
      const d = selected(); if (!d) return;
      snapshot();
      const c = clone(d); c.id = uid(); c.locked = false;
      c.pts.forEach((q) => { q.l += 3; });
      drawings.push(c); selectedId = c.id; afterChange();
    }

    // ---------------------------------------------------------------- keyboard
    function onKey(e) {
      const tag = (e.target && e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || tag === "select" || (e.target && e.target.isContentEditable)) return;
      if (!containerEl.isConnected) return;
      const mod = e.ctrlKey || e.metaKey;
      if (e.key === "Escape") {
        if (draft && draft.type === "path") { commitDraft(); return; }
        if (draft) { cancelDraft(); return; }
        if (activeTool) { setTool(null); return; }
        if (selectedId) { dropTemporary(); selectedId = null; redraw(); updateProps(); }
        return;
      }
      if (e.key === "Enter" && draft && draft.type === "path") { commitDraft(); return; }
      if ((e.key === "Delete" || e.key === "Backspace") && selectedId) { const d = selected(); if (d && !d.locked) { e.preventDefault(); deleteSelected(); } return; }
      if (mod && (e.key === "z" || e.key === "Z")) { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
      if (mod && (e.key === "y" || e.key === "Y")) { e.preventDefault(); redo(); return; }
      if (mod && (e.key === "d" || e.key === "D") && selectedId) { e.preventDefault(); cloneSelected(); return; }
      if (e.altKey && !mod) {
        const map = { t: "trend", h: "hline", v: "vline", f: "fib", r: "box", m: "measure", l: "long", s: "short" };
        const k = map[(e.key || "").toLowerCase()];
        if (k) { e.preventDefault(); setTool(k); }
      }
    }

    // ---------------------------------------------------------------- tool state / toolbar
    function setTool(id) {
      if (id === CURSOR) id = null;
      if (draft && draft.type === "path" && id !== "path") commitDraft();
      else if (draft && id !== activeTool) cancelDraft();
      activeTool = id;
      if (id) { dropTemporary(); selectedId = null; updateProps(); }
      canvas.style.pointerEvents = id ? "auto" : "none";
      canvas.style.cursor = id ? "crosshair" : "default";
      updateToolbarState();
    }
    let toolbarRefs = null;
    function updateToolbarState() {
      if (!toolbarRefs) return;
      const cur = activeTool || CURSOR;
      toolbarRefs.cursorBtn.classList.toggle("active", cur === CURSOR);
      toolbarRefs.groupBtns.forEach((gb) => {
        const g = gb.group, on = g.tools.some((t) => t.id === cur);
        gb.btn.classList.toggle("active", on);
        if (on) gb.icon.textContent = TOOL_INDEX[cur].icon; else gb.icon.textContent = gb.lastIcon;
      });
      toolbarRefs.magnetBtn.classList.toggle("on", magnet);
      toolbarRefs.keepBtn.classList.toggle("on", keepDrawing);
      toolbarRefs.hideBtn.classList.toggle("on", hidden);
      toolbarRefs.lockBtn.classList.toggle("on", lockedAll);
      toolbarRefs.undoBtn.disabled = !undoStack.length;
      toolbarRefs.redoBtn.disabled = !redoStack.length;
      toolbarRefs.countEl.textContent = drawings.length ? String(drawings.length) : "";
    }
    function buildToolbar() {
      if (!toolbarEl) return;
      toolbarEl.classList.add("t58-draw-toolbar");
      const btn = (cls, html, title, fn) => {
        const b = document.createElement("button"); b.type = "button"; b.className = "t58-draw-btn " + cls; b.innerHTML = html; b.title = title;
        b.addEventListener("click", fn); return b;
      };
      const cursorBtn = btn("t58-draw-cursor", "↖", "Cursor: pan, zoom, select drawings (Esc)", () => setTool(null));
      toolbarEl.appendChild(cursorBtn);
      const groupBtns = [];
      const menus = [];
      const closeMenus = () => menus.forEach((m) => m.classList.remove("open"));
      GROUPS.forEach((g) => {
        const wrap = document.createElement("div"); wrap.className = "t58-draw-grp";
        const b = btn("", "<span class='gi'>" + g.icon + "</span> ▾", g.label, (ev) => {
          ev.stopPropagation();
          const isOpen = menu.classList.contains("open"); closeMenus(); if (!isOpen) menu.classList.add("open");
        });
        const menu = document.createElement("div"); menu.className = "t58-draw-menu"; menus.push(menu);
        g.tools.forEach((t) => {
          const it = document.createElement("button"); it.type = "button";
          it.innerHTML = "<span class='ic'>" + t.icon + "</span>" + t.label;
          it.addEventListener("click", (ev) => { ev.stopPropagation(); closeMenus(); setTool(t.id); });
          menu.appendChild(it);
        });
        wrap.appendChild(b); wrap.appendChild(menu); toolbarEl.appendChild(wrap);
        groupBtns.push({ group: g, btn: b, icon: b.querySelector(".gi"), lastIcon: g.icon });
      });
      document.addEventListener("click", closeMenus);
      const sep = () => { const s = document.createElement("span"); s.className = "t58-draw-sep"; toolbarEl.appendChild(s); };
      sep();
      const magnetBtn = btn("toggle", "⦿", "Magnet: snap to candle open/high/low/close", () => setMagnet(!magnet));
      const keepBtn = btn("toggle", "↻", "Keep drawing: stay in the tool after each shape", () => { keepDrawing = !keepDrawing; updateToolbarState(); });
      const undoBtn = btn("", "↶", "Undo (Ctrl+Z)", undo);
      const redoBtn = btn("", "↷", "Redo (Ctrl+Y)", redo);
      const hideBtn = btn("toggle", "\u{1F441}", "Hide / show all drawings", () => setHidden(!hidden));
      const lockBtn = btn("toggle", "\u{1F512}", "Lock all drawings (no accidental moves)", () => { lockedAll = !lockedAll; updateToolbarState(); });
      [magnetBtn, keepBtn, undoBtn, redoBtn, hideBtn, lockBtn].forEach((b) => toolbarEl.appendChild(b));
      const clearBtn = btn("t58-draw-clear", "✕ Clear", "Remove every drawing on this chart", () => {
        if (!drawings.length) return;
        let ok = true;
        try { ok = drawings.length < 3 || window.confirm("Remove all " + drawings.length + " drawings from this chart?"); } catch (e) { ok = true; }
        if (ok) clearAll();
      });
      toolbarEl.appendChild(clearBtn);
      const countEl = document.createElement("span"); countEl.style.cssText = "font-size:10px;opacity:.6;margin-left:2px"; toolbarEl.appendChild(countEl);
      toolbarRefs = { cursorBtn, groupBtns, magnetBtn, keepBtn, undoBtn, redoBtn, hideBtn, lockBtn, countEl };
      updateToolbarState();
    }

    function setMagnet(v) { magnet = !!v; updateToolbarState(); }
    function setHidden(v) { hidden = !!v; if (hidden) { selectedId = null; } save(); redraw(); updateProps(); updateToolbarState(); }
    function clearAll() {
      if (drawings.length) snapshot();
      drawings = []; draft = null; dragState = null; selectedId = null; afterChange();
    }
    function setSeries(s) { mainSeries = s; redraw(); }
    function setStorageKey(k) {
      storageKey = k || null; selectedId = null; undoStack = []; redoStack = [];
      load(); redraw(); updateProps(); updateToolbarState();
    }
    function setDrawings(list) { drawings = Array.isArray(list) ? list : []; afterChange(); }
    function destroy() {
      destroyed = true;
      try { ro && ro.disconnect(); } catch (e) { /* ignore */ }
      document.removeEventListener("keydown", onKey);
      containerEl.removeEventListener("mousedown", onContainerDown, true);
      containerEl.removeEventListener("mousemove", onContainerHover);
      containerEl.removeEventListener("dblclick", onDblClickContainer);
      try { canvas.remove(); propsEl.remove(); } catch (e) { /* ignore */ }
    }

    canvas.addEventListener("mousedown", onCanvasDown);
    canvas.addEventListener("mousemove", onCanvasMove);
    canvas.addEventListener("mouseup", onCanvasUp);
    canvas.addEventListener("dblclick", onCanvasDbl);
    containerEl.addEventListener("mousedown", onContainerDown, true);
    containerEl.addEventListener("mousemove", onContainerHover);
    containerEl.addEventListener("dblclick", onDblClickContainer);
    document.addEventListener("keydown", onKey);
    chart.timeScale().subscribeVisibleLogicalRangeChange(scheduleRedraw);
    // The price axis can rescale without the time range changing (autoscale on new data): redraw on any crosshair-free tick too.
    try { chart.subscribeCrosshairMove(() => { if (draft) scheduleRedraw(); }); } catch (e) { /* ignore */ }

    load();
    buildToolbar();
    resize();

    return {
      setTool, clear: clearAll, redraw, setSeries, setStorageKey, undo, redo, setMagnet, setHidden,
      getDrawings: () => clone(drawings), setDrawings, destroy,
      deleteSelected, get activeTool() { return activeTool; },
    };
  }

  global.T58ChartDrawingTools = { attach, TOOLS: GROUPS };
})(window);
