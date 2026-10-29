/* T58 dashboard visuals. No external libraries -- the desktop build runs
   fully offline, so nothing here may depend on a CDN. */

const T58_COLORS = {
  teal: "#35e0b0",
  violet: "#8b7cff",
  coral: "#ff7b7b",
  amber: "#f0b429",
  grid: "rgba(130, 235, 220, 0.14)",
};

// One cool ramp (teal / violet / cyan / blue) plus gold, coral and a soft green: same family as theme.css.
const CLUSTER_PALETTE = ["#35e0b0", "#8b7cff", "#4cc9f0", "#f0b429", "#ff7b7b", "#7ee0a0", "#6aa8ff"];

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function ensureGlowFilter(svg, id, color, stdDev) {
  let defs = svg.querySelector("defs");
  if (!defs) {
    defs = svgEl("defs", {});
    svg.insertBefore(defs, svg.firstChild);
  }
  if (svg.querySelector(`#${id}`)) return;
  const filter = svgEl("filter", { id, x: "-60%", y: "-60%", width: "220%", height: "220%" });
  filter.appendChild(svgEl("feGaussianBlur", { stdDeviation: stdDev, result: "blur" }));
  const merge = svgEl("feMerge", {});
  merge.appendChild(svgEl("feMergeNode", { in: "blur" }));
  merge.appendChild(svgEl("feMergeNode", { in: "SourceGraphic" }));
  filter.appendChild(merge);
  defs.appendChild(filter);
}

/* ---- Equity comparison chart ----
   Every curve is drawn as RETURN FROM ITS OWN STARTING BALANCE (%), not raw dollars. Raw dollars put a $200k account
   far above a $50k one and squash everything else flat; returns put every strategy on the same footing, so the curves
   are directly comparable and fill the chart.
   Click any legend chip (or line) to add/remove it from the comparison -- select as many as you like. With a selection,
   the axis zooms to just those curves and the rest fade into the background. Selection survives the dashboard's
   periodic repaints. */
const t58EquitySel = new Set();
const EQUITY_COLORS = ["#35e0b0", "#8b7cff", "#4cc9f0", "#f0b429", "#ff7b7b", "#7ee0a0", "#6aa8ff", "#e879f9", "#fb923c", "#a3e635"];

function eqNiceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * p;
}

function eqTicks(lo, hi, target) {
  const step = eqNiceStep((hi - lo) / Math.max(target, 2));
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-6 ? 0 : v);
  return { ticks: out, step };
}

function eqSmoothPath(pts) {
  if (pts.length < 3) return "M" + pts.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join("L");
  let d = `M${pts[0][0].toFixed(1)},${pts[0][1].toFixed(1)}`;
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i][0] + pts[i + 1][0]) / 2, my = (pts[i][1] + pts[i + 1][1]) / 2;
    d += `Q${pts[i][0].toFixed(1)},${pts[i][1].toFixed(1)} ${mx.toFixed(1)},${my.toFixed(1)}`;
  }
  const l = pts[pts.length - 1];
  return d + `L${l[0].toFixed(1)},${l[1].toFixed(1)}`;
}

function eqThin(vals, max) {
  if (vals.length <= max) return vals.slice();
  const out = [];
  const step = (vals.length - 1) / (max - 1);
  for (let i = 0; i < max; i++) out.push(vals[Math.round(i * step)]);
  return out;
}

function renderEquityChart(container, series) {
  container.__eqSeries = series;
  container.innerHTML = "";
  if (!series || !series.length) {
    container.innerHTML = '<div class="t58-empty">No completed runs yet -- run a strategy to populate this chart.</div>';
    return;
  }
  // forget selections for strategies that no longer exist
  Array.from(t58EquitySel).forEach(n => { if (!series.some(s => s.name === n)) t58EquitySel.delete(n); });

  // Stable colour per strategy (by name order, so a repaint or a re-sort never changes a strategy's colour)
  const byName = series.map(s => s.name).sort();
  const colorOf = name => EQUITY_COLORS[byName.indexOf(name) % EQUITY_COLORS.length];

  // Return from each curve's own starting balance. If a curve has no usable starting balance, fall back to dollars for all.
  const pctMode = series.every(s => { const st = (typeof s.start === "number" && s.start > 0) ? s.start : s.values[0]; return st > 0; });
  const curves = series.map(s => {
    const start = (typeof s.start === "number" && s.start > 0) ? s.start : s.values[0];
    const raw = pctMode ? s.values.map(v => (v / start - 1) * 100) : s.values.slice();
    return { s, color: colorOf(s.name), r: eqThin(raw, 110), final: raw.length ? raw[raw.length - 1] : 0 };
  });
  const fmt = v => pctMode
    ? ((v > 0 ? "+" : v < 0 ? "−" : "") + (Math.abs(v) >= 100 ? Math.abs(v).toFixed(0) : Math.abs(v) >= 10 ? Math.abs(v).toFixed(1) : Math.abs(v).toFixed(2)) + "%")
    : ((v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 }));
  const axisFmt = v => pctMode ? (v === 0 ? "0%" : (v > 0 ? "+" : "−") + Math.abs(v) + "%") : (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString();

  const hasSel = t58EquitySel.size > 0;
  const visible = hasSel ? curves.filter(c => t58EquitySel.has(c.s.name)) : curves;

  const W = Math.max(280, Math.round(container.clientWidth || 640));
  const H = Math.round(Math.min(340, Math.max(230, W * 0.44)));
  const padL = pctMode ? 48 : 58, padR = hasSel ? Math.min(118, Math.round(W * 0.2)) : 14, padT = 12, padB = 24;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  let lo = 0, hi = 0;
  visible.forEach(c => c.r.forEach(v => { if (v < lo) lo = v; if (v > hi) hi = v; }));
  if (hi - lo < 1e-9) { hi = lo + 1; }
  const span = hi - lo; lo -= span * 0.06; hi += span * 0.08;
  const tk = eqTicks(lo, hi, H > 280 ? 6 : 5);
  const x = (i, n) => padL + (n <= 1 ? 0 : (i / (n - 1))) * plotW;
  const y = v => padT + (1 - (v - lo) / (hi - lo)) * plotH;

  const wrap = document.createElement("div");
  wrap.style.cssText = "position:relative;";
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%", height: H, role: "img", "aria-label": "Equity curves as return from starting balance" });
  svg.style.display = "block";
  ensureGlowFilter(svg, "t58-glow-line", null, 2.2);
  const clipId = "t58-eq-clip";
  const defs = svg.querySelector("defs");
  const clip = svgEl("clipPath", { id: clipId });
  clip.appendChild(svgEl("rect", { x: padL, y: padT - 2, width: plotW, height: plotH + 4 }));
  defs.appendChild(clip);

  // grid + axis labels
  tk.ticks.forEach(v => {
    const yy = y(v);
    if (yy < padT - 1 || yy > padT + plotH + 1) return;
    svg.appendChild(svgEl("line", { x1: padL, x2: padL + plotW, y1: yy, y2: yy, stroke: v === 0 ? "rgba(200,230,240,.38)" : T58_COLORS.grid, "stroke-dasharray": v === 0 ? "4 4" : "", "stroke-width": 1 }));
    const t = svgEl("text", { x: padL - 7, y: yy + 3.5, "text-anchor": "end", "font-size": 10.5 });
    t.style.fill = "var(--text-dim)"; t.textContent = axisFmt(v);
    svg.appendChild(t);
  });
  [0, 0.25, 0.5, 0.75, 1].forEach(f => {
    const t = svgEl("text", { x: padL + f * plotW, y: H - 7, "text-anchor": f === 0 ? "start" : f === 1 ? "end" : "middle", "font-size": 10 });
    t.style.fill = "var(--text-dim)"; t.textContent = f === 0 ? "start" : f === 1 ? "latest" : Math.round(f * 100) + "%";
    svg.appendChild(t);
  });

  const g = svgEl("g", { "clip-path": `url(#${clipId})` });
  svg.appendChild(g);
  const order = curves.slice().sort((a, b) => (t58EquitySel.has(a.s.name) ? 1 : 0) - (t58EquitySel.has(b.s.name) ? 1 : 0));  // selected on top
  order.forEach(c => {
    const hot = hasSel && t58EquitySel.has(c.s.name);
    const faded = hasSel && !hot;
    const pts = c.r.map((v, i) => [x(i, c.r.length), y(v)]);
    const d = eqSmoothPath(pts);
    if (hasSel && hot && t58EquitySel.size === 1) {
      const area = svgEl("path", { d: d + `L${pts[pts.length - 1][0].toFixed(1)},${y(0).toFixed(1)}L${pts[0][0].toFixed(1)},${y(0).toFixed(1)}Z`, fill: c.color, opacity: 0.1 });
      g.appendChild(area);
    }
    if (!faded) g.appendChild(svgEl("path", { d, fill: "none", stroke: c.color, "stroke-width": 3, opacity: hot ? 0.45 : 0.28, filter: "url(#t58-glow-line)", "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const line = svgEl("path", { d, fill: "none", stroke: c.color, "stroke-width": hot ? 2.4 : 1.7, opacity: faded ? 0.14 : 1, "stroke-linejoin": "round", "stroke-linecap": "round" });
    g.appendChild(line);
    const hit = svgEl("path", { d, fill: "none", stroke: "transparent", "stroke-width": 12, style: "cursor:pointer;pointer-events:stroke;" });
    const tip = svgEl("title", {}); tip.textContent = `${c.s.name} ${fmt(c.final)} -- click to ${t58EquitySel.has(c.s.name) ? "remove from" : "add to"} the comparison`;
    hit.appendChild(tip);
    hit.addEventListener("click", e => { e.stopPropagation(); toggle(c.s.name); });
    g.appendChild(hit);
  });

  // end-of-line labels for the selected curves (nudged apart so they never overlap)
  if (hasSel) {
    const ends = visible.map(c => ({ c, yy: y(c.r[c.r.length - 1]) })).sort((a, b) => a.yy - b.yy);
    for (let i = 1; i < ends.length; i++) if (ends[i].yy - ends[i - 1].yy < 13) ends[i].yy = ends[i - 1].yy + 13;
    ends.forEach(e => {
      svg.appendChild(svgEl("circle", { cx: padL + plotW, cy: y(e.c.r[e.c.r.length - 1]), r: 3.2, fill: e.c.color }));
      const t = svgEl("text", { x: padL + plotW + 8, y: e.yy + 3.5, "font-size": 10.5, "font-weight": 600 });
      t.setAttribute("fill", e.c.color);
      const nm = e.c.s.name.length > 11 ? e.c.s.name.slice(0, 10) + "…" : e.c.s.name;
      t.textContent = `${nm} ${fmt(e.c.final)}`;
      svg.appendChild(t);
    });
  }

  // hover readout
  const cross = svgEl("line", { y1: padT, y2: padT + plotH, stroke: "rgba(200,230,240,.35)", "stroke-width": 1, visibility: "hidden", "pointer-events": "none" });
  svg.appendChild(cross);
  const tipBox = document.createElement("div");
  tipBox.style.cssText = "position:absolute;pointer-events:none;display:none;z-index:3;background:var(--panel-3,#0f1722);border:1px solid var(--border,#243044);border-radius:10px;padding:7px 10px;font-size:11.5px;line-height:1.5;box-shadow:0 8px 24px rgba(0,0,0,.35);white-space:nowrap;";
  const overlay = svgEl("rect", { x: padL, y: padT, width: plotW, height: plotH, fill: "transparent" });
  overlay.style.cursor = "default";
  overlay.addEventListener("click", () => { if (t58EquitySel.size) { t58EquitySel.clear(); redraw(); } });
  svg.addEventListener("pointermove", ev => {
    const box = svg.getBoundingClientRect();
    const px = (ev.clientX - box.left) * (W / box.width);
    if (px < padL - 2 || px > padL + plotW + 2) { cross.setAttribute("visibility", "hidden"); tipBox.style.display = "none"; return; }
    const f = Math.min(1, Math.max(0, (px - padL) / plotW));
    cross.setAttribute("x1", padL + f * plotW); cross.setAttribute("x2", padL + f * plotW); cross.setAttribute("visibility", "visible");
    const rows = visible.map(c => ({ c, v: c.r[Math.round(f * (c.r.length - 1))] })).sort((a, b) => b.v - a.v).slice(0, 8);
    tipBox.innerHTML = `<div style="color:var(--text-dim);margin-bottom:2px;">${Math.round(f * 100)}% through the run</div>` + rows.map(r =>
      `<div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${r.c.color};margin-right:6px;"></span>${r.c.s.name.replace(/</g, "&lt;")} <b style="float:right;margin-left:14px;">${fmt(r.v)}</b></div>`).join("");
    tipBox.style.display = "block";
    const left = (padL + f * plotW) * (box.width / W);
    tipBox.style.left = Math.min(Math.max(4, left + 12), Math.max(4, box.width - tipBox.offsetWidth - 4)) + "px";
    tipBox.style.top = "10px";
  });
  svg.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tipBox.style.display = "none"; });
  svg.appendChild(overlay);
  // lines sit under the overlay for hover, so re-raise the hit paths for clicking
  Array.from(g.querySelectorAll("path[stroke=transparent]")).forEach(p => svg.appendChild(p));
  svg.querySelectorAll("path[stroke=transparent]").forEach(p => p.setAttribute("clip-path", `url(#${clipId})`));

  wrap.appendChild(svg);
  wrap.appendChild(tipBox);
  container.appendChild(wrap);

  // legend: click to add / remove (multi-select)
  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:6px 8px;margin-top:8px;";
  curves.slice().sort((a, b) => b.final - a.final).forEach(c => {
    const on = !hasSel || t58EquitySel.has(c.s.name);
    const hot = hasSel && t58EquitySel.has(c.s.name);
    const item = document.createElement("button");
    item.type = "button";
    item.setAttribute("aria-pressed", hot ? "true" : "false");
    item.style.cssText = `display:flex;align-items:center;gap:6px;font-size:11px;color:${hot ? "var(--text)" : "var(--text-muted)"};background:${hot ? "rgba(255,255,255,.05)" : "none"};border:1px solid ${hot ? c.color : "var(--border,transparent)"};border-radius:999px;padding:2px 9px;cursor:pointer;font-family:inherit;opacity:${on ? 1 : 0.45};`;
    item.innerHTML = `<span style="width:9px;height:9px;border-radius:50%;background:${c.color};box-shadow:0 0 6px ${c.color};display:inline-block;"></span>`;
    item.appendChild(document.createTextNode(c.s.name + " "));
    const fin = document.createElement("span");
    fin.style.cssText = `font-variant-numeric:tabular-nums;color:${c.final >= 0 ? "var(--teal)" : "var(--coral)"};`;
    fin.textContent = fmt(c.final);
    item.appendChild(fin);
    item.title = `Click to ${hot ? "remove from" : "add to"} the comparison`;
    item.addEventListener("click", e => { e.stopPropagation(); toggle(c.s.name); });
    legend.appendChild(item);
  });
  const hint = document.createElement("button");
  hint.type = "button";
  hint.style.cssText = "margin-left:auto;font-size:11px;color:var(--text-dim);background:none;border:none;cursor:pointer;font-family:inherit;text-decoration:underline dotted;";
  hint.textContent = hasSel ? "Show all" : (series.length > 1 ? "Click strategies to compare" : "");
  hint.addEventListener("click", e => { e.stopPropagation(); if (hasSel) { t58EquitySel.clear(); redraw(); } });
  if (hint.textContent) legend.appendChild(hint);
  container.appendChild(legend);

  function toggle(name) { if (t58EquitySel.has(name)) t58EquitySel.delete(name); else t58EquitySel.add(name); redraw(); }
  function redraw() { renderEquityChart(container, container.__eqSeries); }

  // keep the chart crisp when the card is resized (one observer per container)
  if (!container.__eqRO && typeof ResizeObserver !== "undefined") {
    let lastW = container.clientWidth, timer = 0;
    container.__eqRO = new ResizeObserver(() => {
      if (Math.abs(container.clientWidth - lastW) < 8) return;
      clearTimeout(timer);
      timer = setTimeout(() => { lastW = container.clientWidth; if (container.__eqSeries) renderEquityChart(container, container.__eqSeries); }, 120);
    });
    container.__eqRO.observe(container);
  }
}

/* ---- Weekday x hour heatmap ---- */
function renderHeatmap(container, grid) {
  container.innerHTML = "";
  let maxAbs = 0.0001;
  grid.forEach(row => row.forEach(v => { maxAbs = Math.max(maxAbs, Math.abs(v)); }));

  const days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const table = document.createElement("div");
  table.style.cssText = "display:grid;grid-template-columns:32px repeat(24,1fr);gap:2px;font-size:9px;";

  table.appendChild(document.createElement("div"));
  for (let h = 0; h < 24; h += 1) {
    const label = document.createElement("div");
    label.style.cssText = "color:var(--text-dim);text-align:center;";
    label.textContent = h % 3 === 0 ? h : "";
    table.appendChild(label);
  }

  grid.forEach((row, d) => {
    const dayLabel = document.createElement("div");
    dayLabel.style.cssText = "color:var(--text-muted);display:flex;align-items:center;";
    dayLabel.textContent = days[d];
    table.appendChild(dayLabel);
    row.forEach(v => {
      const cell = document.createElement("div");
      const intensity = Math.min(Math.abs(v) / maxAbs, 1);
      const color = v >= 0 ? T58_COLORS.teal : T58_COLORS.coral;
      cell.style.cssText = `aspect-ratio:1;border-radius:2px;background:${color};opacity:${(0.08 + intensity * 0.85).toFixed(2)};`;
      cell.title = `${days[d]} ${String(row.indexOf(v)).padStart(2, "0")}:00 -> ${v.toFixed(0)}`;
      table.appendChild(cell);
    });
  });

  container.appendChild(table);
}

/* ---- Strategy universe graph (clustered by instrument) ---- */
function renderUniverseGraph(container, graph) {
  container.innerHTML = "";
  if (!graph || !graph.nodes || !graph.nodes.length) {
    container.innerHTML = '<div class="t58-empty">No strategies run yet -- this fills in once you run a few.</div>';
    return;
  }
  const width = 560, height = 260;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "260" });
  ensureGlowFilter(svg, "t58-glow-node", null, 3);

  const nClusters = Math.max(graph.instruments.length, 1);
  const clusterCenters = graph.instruments.map((_, i) => {
    const angle = (i / nClusters) * Math.PI * 2;
    const r = nClusters > 1 ? Math.min(width, height) * 0.30 : 0;
    return { x: width / 2 + r * Math.cos(angle), y: height / 2 + r * Math.sin(angle) };
  });

  const pos = graph.nodes.map((n, i) => {
    const c = clusterCenters[n.cluster] || { x: width / 2, y: height / 2 };
    const angle = (i / graph.nodes.length) * Math.PI * 2;
    return { x: c.x + Math.cos(angle) * 26, y: c.y + Math.sin(angle) * 26, vx: 0, vy: 0 };
  });

  for (let iter = 0; iter < 60; iter += 1) {
    for (let i = 0; i < pos.length; i += 1) {
      let fx = 0, fy = 0;
      for (let j = 0; j < pos.length; j += 1) {
        if (i === j) continue;
        const dx = pos[i].x - pos[j].x, dy = pos[i].y - pos[j].y;
        const d2 = Math.max(dx * dx + dy * dy, 20);
        fx += (dx / d2) * 400;
        fy += (dy / d2) * 400;
      }
      const c = clusterCenters[graph.nodes[i].cluster] || { x: width / 2, y: height / 2 };
      fx += (c.x - pos[i].x) * 0.02;
      fy += (c.y - pos[i].y) * 0.02;
      pos[i].vx = (pos[i].vx + fx) * 0.35;
      pos[i].vy = (pos[i].vy + fy) * 0.35;
    }
    graph.edges.forEach(e => {
      const a = pos[e.source], b = pos[e.target];
      const dx = b.x - a.x, dy = b.y - a.y;
      const pull = e.weight * 0.02;
      a.vx += dx * pull; a.vy += dy * pull;
      b.vx -= dx * pull; b.vy -= dy * pull;
    });
    pos.forEach(p => {
      p.x = Math.min(width - 16, Math.max(16, p.x + p.vx));
      p.y = Math.min(height - 16, Math.max(16, p.y + p.vy));
    });
  }

  graph.edges.forEach(e => {
    const a = pos[e.source], b = pos[e.target];
    svg.appendChild(svgEl("line", {
      x1: a.x.toFixed(1), y1: a.y.toFixed(1), x2: b.x.toFixed(1), y2: b.y.toFixed(1),
      stroke: T58_COLORS.violet, "stroke-width": (0.5 + e.weight * 2).toFixed(2), opacity: (0.15 + e.weight * 0.35).toFixed(2),
    }));
  });

  graph.nodes.forEach((n, i) => {
    const color = CLUSTER_PALETTE[n.cluster % CLUSTER_PALETTE.length];
    const r = 5 + Math.min(Math.max(n.sharpe, 0), 3) * 2.2;
    const g = svgEl("g", {});
    g.appendChild(svgEl("circle", { cx: pos[i].x, cy: pos[i].y, r, fill: color, filter: "url(#t58-glow-node)", opacity: 0.55 }));
    const dot = svgEl("circle", { cx: pos[i].x, cy: pos[i].y, r: r * 0.65, fill: color, stroke: n.passed ? T58_COLORS.teal : T58_COLORS.coral, "stroke-width": 1.2 });
    const title = svgEl("title", {});
    title.textContent = `${n.name}  (${n.instrument})  sharpe ${n.sharpe.toFixed(2)}  ${n.passed ? "pass" : "fail"}`;
    g.appendChild(dot);
    g.appendChild(title);
    svg.appendChild(g);
  });

  container.appendChild(svg);

  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:12px;margin-top:6px;";
  graph.instruments.forEach((name, i) => {
    const item = document.createElement("div");
    item.style.cssText = "display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text-muted);";
    const color = CLUSTER_PALETTE[i % CLUSTER_PALETTE.length];
    item.innerHTML = `<span style="width:9px;height:9px;border-radius:50%;background:${color};display:inline-block;"></span>${name}`;
    legend.appendChild(item);
  });
  container.appendChild(legend);
}

window.T58Dashboard = { renderEquityChart, renderHeatmap, renderUniverseGraph };
