/*
 * T58 Chart Drawing Tools -- a small, dependency-free canvas overlay for
 * TradingView's open-source Lightweight Charts (the same library both
 * Live Market and Interactive Replay already load). Deliberately simple,
 * per the product request this was built for: seven tools (horizontal
 * line, trend line, box, long position, short position, fib retracement,
 * path), no persistence, no undo stack -- just enough to mark up a chart
 * while looking at it.
 *
 * Every drawing is stored in PRICE/TIME space (not pixel space), so it
 * stays anchored to the right candles/price level through panning and
 * zooming -- redrawn on every visible-range change via the chart's own
 * timeScale() coordinate conversion.
 *
 * Usage:
 *   const tools = T58ChartDrawingTools.attach(chart, series, containerEl, toolbarEl);
 *   tools.setTool('trend');   // or null to disable drawing and let the
 *                             // chart pan/zoom normally again
 *   tools.clear();
 *
 * `containerEl` is the chart's own wrapping element (must be position:
 * relative or static -- this sets position:relative itself if neither).
 * `toolbarEl`, if given, gets one button per tool plus a clear button
 * auto-built into it; omit it to build your own buttons and call
 * `setTool(id)` yourself.
 */
(function (global) {
  "use strict";

  const TOOL_DEFS = [
    { id: "hline", label: "H-Line", icon: "\u2015" },
    { id: "trend", label: "Trend Line", icon: "/" },
    { id: "box", label: "Box", icon: "\u25ad" },
    { id: "long", label: "Long", icon: "\u2191" },
    { id: "short", label: "Short", icon: "\u2193" },
    { id: "fib", label: "Fib Retracement", icon: "F" },
    { id: "path", label: "Path", icon: "\u3030" },
  ];
  const FIB_LEVELS = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1];
  const TOOL_COLORS = { hline: "#35e0b0", trend: "#22d3ee", box: "#8b7cff", long: "#35e0b0", short: "#ff6b6b", fib: "#f0b429", path: "#e5e7eb" };

  function attach(chart, series, containerEl, toolbarEl) {
    let activeTool = null;
    let drawings = []; // {type, points: [{time, price}], color}
    let pathDraft = null;   // multi-click path being built (finished with Enter/Escape/double-click)
    let dragStart = null;   // {time, price} for a click-drag tool (trend/box/long/short/fib)
    let dragDraft = null;

    if (!containerEl.style.position || containerEl.style.position === "static") {
      containerEl.style.position = "relative";
    }
    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.left = "0";
    canvas.style.top = "0";
    canvas.style.pointerEvents = "none"; // only intercepts clicks while a tool is active -- see setTool()
    canvas.style.zIndex = "5";
    containerEl.appendChild(canvas);
    const ctx = canvas.getContext("2d");

    function resize() {
      const rect = containerEl.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      redraw();
    }
    if (typeof ResizeObserver !== "undefined") {
      new ResizeObserver(resize).observe(containerEl);
    } else {
      window.addEventListener("resize", resize);
    }

    function xFor(time) { return chart.timeScale().timeToCoordinate(time); }
    function yFor(price) { return series.priceToCoordinate(price); }
    function timeFor(x) { return chart.timeScale().coordinateToTime(x); }
    function priceFor(y) { return series.coordinateToPrice(y); }

    function cssWidth() { return canvas.width / (window.devicePixelRatio || 1); }
    function cssHeight() { return canvas.height / (window.devicePixelRatio || 1); }

    function redraw() {
      ctx.clearRect(0, 0, cssWidth(), cssHeight());
      for (const d of drawings) drawOne(d, false);
      if (dragDraft) drawOne(dragDraft, true);
      if (pathDraft) drawOne(pathDraft, true);
    }

    function drawOne(d, isDraft) {
      const pts = d.points.map((p) => ({ x: xFor(p.time), y: yFor(p.price) }));
      if (pts.some((p) => p.x == null || p.y == null || Number.isNaN(p.x) || Number.isNaN(p.y))) return;
      ctx.save();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = d.color;
      ctx.setLineDash(isDraft ? [4, 3] : []);
      ctx.font = "10.5px ui-monospace, monospace";

      if (d.type === "hline") {
        ctx.beginPath(); ctx.moveTo(0, pts[0].y); ctx.lineTo(cssWidth(), pts[0].y); ctx.stroke();
        ctx.fillStyle = d.color;
        ctx.fillText(fmtPrice(d.points[0].price), cssWidth() - 78, pts[0].y - 4);
      } else if (d.type === "trend" && pts.length >= 2) {
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y); ctx.lineTo(pts[1].x, pts[1].y); ctx.stroke();
      } else if (d.type === "path" && pts.length >= 1) {
        ctx.beginPath(); ctx.moveTo(pts[0].x, pts[0].y);
        for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
        ctx.stroke();
        for (const p of pts) { ctx.beginPath(); ctx.arc(p.x, p.y, 2.5, 0, Math.PI * 2); ctx.fillStyle = d.color; ctx.fill(); }
      } else if (d.type === "box" && pts.length >= 2) {
        const x = Math.min(pts[0].x, pts[1].x), y = Math.min(pts[0].y, pts[1].y);
        const w = Math.abs(pts[1].x - pts[0].x), h = Math.abs(pts[1].y - pts[0].y);
        ctx.globalAlpha = 0.12; ctx.fillStyle = d.color; ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 1; ctx.strokeRect(x, y, w, h);
      } else if ((d.type === "long" || d.type === "short") && pts.length >= 2) {
        drawPositionTool(d, pts);
      } else if (d.type === "fib" && pts.length >= 2) {
        drawFib(d, pts);
      }
      ctx.restore();
    }

    function drawPositionTool(d, pts) {
      const isLong = d.type === "long";
      const entryY = pts[0].y, targetY = pts[1].y;
      const dist = targetY - entryY;
      const stopY = entryY - dist; // stop mirrored on the opposite side of entry from the target, same distance -- a simple 1:1-styled risk zone, not a re-derived R-multiple
      const x0 = Math.min(pts[0].x, pts[1].x) - 30, x1 = Math.max(pts[0].x, pts[1].x) + 90;
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = "#35e0b0"; ctx.fillRect(x0, Math.min(entryY, targetY), x1 - x0, Math.abs(entryY - targetY));
      ctx.fillStyle = "#ff6b6b"; ctx.fillRect(x0, Math.min(entryY, stopY), x1 - x0, Math.abs(entryY - stopY));
      ctx.globalAlpha = 1;
      ctx.strokeStyle = isLong ? "#35e0b0" : "#ff6b6b";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(x0, entryY); ctx.lineTo(x1, entryY); ctx.stroke();
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = "#35e0b0"; ctx.beginPath(); ctx.moveTo(x0, targetY); ctx.lineTo(x1, targetY); ctx.stroke();
      ctx.strokeStyle = "#ff6b6b"; ctx.beginPath(); ctx.moveTo(x0, stopY); ctx.lineTo(x1, stopY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#e5e7eb";
      ctx.fillText((isLong ? "LONG entry " : "SHORT entry ") + fmtPrice(d.points[0].price), x0 + 4, entryY - 4);
      ctx.fillStyle = "#35e0b0"; ctx.fillText("target " + fmtPrice(d.points[1].price), x0 + 4, targetY - 4);
      const stopPrice = priceFor(stopY);
      ctx.fillStyle = "#ff6b6b"; ctx.fillText("stop " + (stopPrice != null ? fmtPrice(stopPrice) : "-"), x0 + 4, stopY - 4);
    }

    function drawFib(d, pts) {
      const p0 = d.points[0].price, p1 = d.points[1].price;
      const x0 = Math.min(pts[0].x, pts[1].x) - 20, x1 = Math.max(pts[0].x, pts[1].x) + 90;
      for (const lvl of FIB_LEVELS) {
        const price = p0 + (p1 - p0) * lvl;
        const y = yFor(price);
        if (y == null) continue;
        ctx.globalAlpha = lvl === 0 || lvl === 1 ? 0.9 : 0.55;
        ctx.strokeStyle = "#f0b429";
        ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.stroke();
        ctx.globalAlpha = 1; ctx.fillStyle = "#f0b429";
        ctx.fillText(lvl.toFixed(3) + "  " + fmtPrice(price), x1 - 86, y - 3);
      }
    }

    function fmtPrice(p) {
      const n = Number(p);
      return Math.abs(n) >= 100 ? n.toFixed(2) : n.toFixed(5);
    }

    function toPoint(evt) {
      const rect = canvas.getBoundingClientRect();
      const x = evt.clientX - rect.left, y = evt.clientY - rect.top;
      return { x, y, time: timeFor(x), price: priceFor(y) };
    }

    const CLICK_TOOLS = new Set(["hline"]);
    const DRAG_TOOLS = new Set(["trend", "box", "long", "short", "fib"]);

    function onMouseDown(evt) {
      if (!activeTool) return;
      const p = toPoint(evt);
      if (p.time == null || p.price == null) return;
      evt.preventDefault();

      if (CLICK_TOOLS.has(activeTool)) {
        drawings.push({ type: activeTool, points: [{ time: p.time, price: p.price }], color: TOOL_COLORS[activeTool] });
        redraw();
        return;
      }
      if (activeTool === "path") {
        if (!pathDraft) pathDraft = { type: "path", points: [], color: TOOL_COLORS.path };
        pathDraft.points.push({ time: p.time, price: p.price });
        redraw();
        return;
      }
      if (DRAG_TOOLS.has(activeTool)) {
        dragStart = { time: p.time, price: p.price };
      }
    }

    function onMouseMove(evt) {
      if (!activeTool || !dragStart || !DRAG_TOOLS.has(activeTool)) return;
      const p = toPoint(evt);
      if (p.time == null || p.price == null) return;
      dragDraft = { type: activeTool, points: [dragStart, { time: p.time, price: p.price }], color: TOOL_COLORS[activeTool] };
      redraw();
    }

    function onMouseUp(evt) {
      if (!activeTool || !dragStart || !DRAG_TOOLS.has(activeTool)) return;
      const p = toPoint(evt);
      if (p.time != null && p.price != null && (p.time !== dragStart.time || p.price !== dragStart.price)) {
        drawings.push({ type: activeTool, points: [dragStart, { time: p.time, price: p.price }], color: TOOL_COLORS[activeTool] });
      }
      dragStart = null; dragDraft = null;
      redraw();
    }

    function onDoubleClick() { finishPath(); }
    function onKeyDown(e) {
      if ((e.key === "Escape" || e.key === "Enter") && pathDraft) finishPath();
    }
    function finishPath() {
      if (pathDraft && pathDraft.points.length >= 2) drawings.push(pathDraft);
      pathDraft = null;
      redraw();
    }

    canvas.addEventListener("mousedown", onMouseDown);
    canvas.addEventListener("mousemove", onMouseMove);
    canvas.addEventListener("mouseup", onMouseUp);
    canvas.addEventListener("dblclick", onDoubleClick);
    document.addEventListener("keydown", onKeyDown);
    chart.timeScale().subscribeVisibleLogicalRangeChange(redraw);

    function setTool(toolId) {
      if (activeTool === "path" && toolId !== "path") finishPath();
      activeTool = toolId;
      // Pointer events only capture the canvas while a tool is selected --
      // otherwise it must stay pass-through so the chart's own pan/zoom/
      // crosshair keeps working exactly as before this existed.
      canvas.style.pointerEvents = toolId ? "auto" : "none";
      canvas.style.cursor = toolId ? "crosshair" : "default";
    }

    function clearAll() {
      drawings = []; pathDraft = null; dragDraft = null; dragStart = null;
      redraw();
    }

    resize();

    if (toolbarEl) {
      TOOL_DEFS.forEach((t) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "t58-draw-btn";
        btn.dataset.tool = t.id;
        btn.title = t.label;
        btn.textContent = t.icon;
        btn.addEventListener("click", () => {
          const wasActive = btn.classList.contains("active");
          toolbarEl.querySelectorAll(".t58-draw-btn").forEach((b) => b.classList.remove("active"));
          if (wasActive) {
            setTool(null);
          } else {
            btn.classList.add("active");
            setTool(t.id);
          }
        });
        toolbarEl.appendChild(btn);
      });
      const clearBtn = document.createElement("button");
      clearBtn.type = "button";
      clearBtn.className = "t58-draw-btn t58-draw-clear";
      clearBtn.title = "Clear all drawings";
      clearBtn.textContent = "\u2715 Clear";
      clearBtn.addEventListener("click", () => {
        clearAll();
        toolbarEl.querySelectorAll(".t58-draw-btn").forEach((b) => b.classList.remove("active"));
        setTool(null);
      });
      toolbarEl.appendChild(clearBtn);
    }

    return { setTool, clear: clearAll, redraw };
  }

  global.T58ChartDrawingTools = { attach };
})(window);
