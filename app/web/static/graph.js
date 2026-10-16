/* T58 dashboard visuals. No external libraries -- the desktop build runs
   fully offline, so nothing here may depend on a CDN. */

const T58_COLORS = {
  teal: "#35e0b0",
  violet: "#8b7cff",
  coral: "#ff6f6f",
  amber: "#f0b429",
  grid: "#1c2230",
};

const CLUSTER_PALETTE = ["#35e0b0", "#8b7cff", "#ff6f6f", "#f0b429", "#4fb0ff", "#e07be0", "#7bd4e0"];

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
   Click a line (or its legend entry) to spotlight that strategy: it is drawn
   thick and bright with its name labelled at the end, the others fade. Click
   it again (or the empty chart) to clear. The selection is remembered across
   the dashboard's periodic repaints. */
let t58EquitySelected = null;

function renderEquityChart(container, series) {
  container.innerHTML = "";
  if (!series || !series.length) {
    container.innerHTML = '<div class="t58-empty">No completed runs yet -- run a strategy to populate this chart.</div>';
    return;
  }
  if (t58EquitySelected !== null && !series.some(s => s.name === t58EquitySelected)) t58EquitySelected = null;

  const width = 560, height = 200, pad = 28;
  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height: "220" });
  ensureGlowFilter(svg, "t58-glow-line", null, 2.2);

  let lo = Infinity, hi = -Infinity, maxLen = 0;
  series.forEach(s => {
    s.values.forEach(v => { lo = Math.min(lo, v); hi = Math.max(hi, v); });
    maxLen = Math.max(maxLen, s.values.length);
  });
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  const range = (hi - lo) || 1;

  const x = i => pad + (i / Math.max(maxLen - 1, 1)) * (width - pad * 2);
  const y = v => height - pad - ((v - lo) / range) * (height - pad * 2);

  svg.appendChild(svgEl("line", { x1: pad, y1: height - pad, x2: width - pad, y2: height - pad, stroke: T58_COLORS.grid }));

  const parts = [];   // {name, glow, line, hit, legend, color}
  const select = name => {
    t58EquitySelected = (t58EquitySelected === name) ? null : name;
    apply();
  };

  series.forEach(s => {
    const color = s.passed ? T58_COLORS.teal : T58_COLORS.coral;
    const points = s.values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
    const glow = svgEl("polyline", { points, fill: "none", stroke: color, "stroke-width": 2.5, opacity: 0.5, filter: "url(#t58-glow-line)" });
    const line = svgEl("polyline", { points, fill: "none", stroke: color, "stroke-width": 1.6 });
    const hit = svgEl("polyline", { points, fill: "none", stroke: "transparent", "stroke-width": 12, style: "cursor:pointer;pointer-events:stroke;" });
    const tip = svgEl("title", {});
    tip.textContent = `${s.name} (${s.passed ? "passed" : "failed"}) -- click to highlight`;
    hit.appendChild(tip);
    hit.addEventListener("click", e => { e.stopPropagation(); select(s.name); });
    svg.appendChild(glow);
    svg.appendChild(line);
    svg.appendChild(hit);
    const last = s.values.length ? s.values[s.values.length - 1] : null;
    parts.push({ s, color, glow, line, hit, end: last === null ? null : [x(s.values.length - 1), y(last)] });
  });

  const label = svgEl("text", { "font-size": 10, "font-weight": 600, "pointer-events": "none", "text-anchor": "end" });
  svg.appendChild(label);
  svg.addEventListener("click", () => { if (t58EquitySelected !== null) { t58EquitySelected = null; apply(); } });

  container.appendChild(svg);

  const legend = document.createElement("div");
  legend.style.cssText = "display:flex;flex-wrap:wrap;gap:6px 8px;margin-top:8px;";
  parts.forEach(p => {
    const item = document.createElement("button");
    item.type = "button";
    item.style.cssText = "display:flex;align-items:center;gap:6px;font-size:11px;color:var(--text-muted);background:none;border:1px solid transparent;border-radius:999px;padding:2px 8px;cursor:pointer;font-family:inherit;";
    item.innerHTML = `<span style="width:9px;height:9px;border-radius:50%;background:${p.color};box-shadow:0 0 6px ${p.color};display:inline-block;"></span>`;
    item.appendChild(document.createTextNode(p.s.name));
    item.title = "Click to highlight this strategy";
    item.addEventListener("click", e => { e.stopPropagation(); select(p.s.name); });
    legend.appendChild(item);
    p.legend = item;
  });
  container.appendChild(legend);

  function apply() {
    const sel = t58EquitySelected;
    label.textContent = "";
    parts.forEach(p => {
      const on = sel === null || p.s.name === sel;
      const hot = sel !== null && p.s.name === sel;
      p.line.setAttribute("stroke-width", hot ? 3.4 : 1.6);
      p.line.setAttribute("opacity", on ? 1 : 0.15);
      p.glow.setAttribute("opacity", hot ? 0.9 : (on ? 0.5 : 0.04));
      p.legend.style.opacity = on ? 1 : 0.4;
      p.legend.style.borderColor = hot ? p.color : "transparent";
      p.legend.style.color = hot ? "var(--text)" : "var(--text-muted)";
      if (hot) {
        svg.appendChild(p.glow); svg.appendChild(p.line); svg.appendChild(p.hit);   // draw on top
        if (p.end) {
          label.setAttribute("x", Math.min(width - 4, p.end[0]));
          label.setAttribute("y", Math.max(12, p.end[1] - 8));
          label.setAttribute("fill", p.color);
          label.textContent = p.s.name;
          svg.appendChild(label);
        }
      }
    });
  }
  apply();
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
