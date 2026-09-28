/*
 * Run telemetry front end: phase banner, equity swarm, survivors strip and the
 * Universe Map. Server side: app/web/telemetry_routes.py.
 *
 * Plain JS, no dependencies. Each block activates only if its container is on
 * the page (data-t58-phase-banner / data-t58-swarm / data-t58-survivors /
 * data-t58-universe), so this one file is safe to load from any template --
 * and safe to load twice (guarded below), since several partials include it.
 *
 * The pure helpers (clipRange, nearestDot, formatMoney, ...) are exposed on
 * window.T58Telemetry so tests/test_telemetry_js.py can exercise them under
 * node without a browser.
 */
(function (root) {
  "use strict";
  if (root.T58Telemetry) return;

  var COLORS = { up: "53,224,176", down: "255,111,111" };
  var STAGE_COLORS = { rejected: "#5b6478", tested: "#6fa8ff", validated: "#f0b429", survivor: "#35e0b0" };
  var STAGE_ORDER = ["rejected", "tested", "validated", "survivor"];
  var STAGE_LABELS = { rejected: "Rejected", tested: "Passed filter", validated: "Validated", survivor: "Survivor" };
  var FAMILY_COLORS = ["#35e0b0", "#6fa8ff", "#f0b429", "#ff6f6f", "#b58cff", "#00d4ff", "#ff2bd6", "#b6ff3c",
                       "#ffb547", "#8a93a6", "#7bd88f", "#e07bb2"];

  // ------------------------------------------------------------------ helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function formatMoney(v) {
    if (v == null || !isFinite(v)) return "n/a";
    var sign = v < 0 ? "-" : "";
    return sign + "$" + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }

  function formatPct(v, digits) {
    if (v == null || !isFinite(v)) return "n/a";
    return v.toFixed(digits == null ? 1 : digits) + "%";
  }

  /** Robust y-range for the swarm: clips the top/bottom 2% so one outlier
   *  candidate can't flatten every other curve; always includes 0. */
  function clipRange(values) {
    var flat = [];
    for (var i = 0; i < values.length; i++) {
      var row = values[i];
      for (var j = 0; j < row.length; j++) if (isFinite(row[j])) flat.push(row[j]);
    }
    if (!flat.length) return { lo: -1, hi: 1 };
    flat.sort(function (a, b) { return a - b; });
    var lo = flat[Math.floor(0.02 * (flat.length - 1))];
    var hi = flat[Math.ceil(0.98 * (flat.length - 1))];
    lo = Math.min(lo, 0); hi = Math.max(hi, 0);
    if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
    var pad = (hi - lo) * 0.06;
    return { lo: lo - pad, hi: hi + pad };
  }

  /** Index of the dot closest to (x, y) within `radius` px, else -1. */
  function nearestDot(pts, x, y, radius) {
    var best = -1, bestD = radius * radius;
    for (var i = 0; i < pts.length; i++) {
      var dx = pts[i].px - x, dy = pts[i].py - y, d = dx * dx + dy * dy;
      if (d <= bestD) { bestD = d; best = i; }
    }
    return best;
  }

  function getJSON(url) {
    return fetch(url, { headers: { "Accept": "application/json" } }).then(function (r) {
      return r.json().catch(function () { return {}; });
    });
  }

  function setupCanvas(canvas, cssHeight) {
    var wrap = canvas.parentNode;
    var w = Math.max(200, wrap.clientWidth || 300);
    var dpr = root.devicePixelRatio || 1;
    canvas.style.height = cssHeight + "px";
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(cssHeight * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx: ctx, w: w, h: cssHeight };
  }

  function eventPoint(canvas, ev) {
    var rect = canvas.getBoundingClientRect();
    var t = ev.touches && ev.touches[0] ? ev.touches[0] : ev;
    return { x: t.clientX - rect.left, y: t.clientY - rect.top };
  }

  // -------------------------------------------------------------- phase banner
  var PHASES = ["Proving Ground", "Breeding", "Out-of-Sample", "Survivors"];

  function initBanner(el) {
    var url = el.getAttribute("data-progress-url");
    var key = el.getAttribute("data-progress-key");
    var text = el.querySelector("[data-role=text]");
    var pips = el.querySelector("[data-role=pips]");
    var stats = el.querySelector("[data-role=stats]");
    var timer = null;

    pips.innerHTML = PHASES.map(function (p, i) {
      return '<div class="t58-tm-pip" data-i="' + (i + 1) + '">' + (i + 1) + " " + esc(p) + "</div>";
    }).join("");

    function render(p) {
      el.hidden = false;
      el.classList.toggle("done", !!p.done);
      text.textContent = p.banner || "Starting…";
      var nodes = pips.children;
      for (var i = 0; i < nodes.length; i++) {
        var n = i + 1;
        nodes[i].className = "t58-tm-pip" + (n === p.phase_index && !p.done ? " active" : (n <= p.phase_index ? " reached" : ""));
      }
      var bits = [];
      if (p.candidates_total && p.candidates_total > p.candidates_done && !p.done) {
        bits.push("<span><b>" + p.candidates_done.toLocaleString() + "</b> of " + p.candidates_total.toLocaleString() + " candidates</span>");
      }
      if (p.survivors != null) bits.push("<span><b>" + p.survivors.toLocaleString() + "</b> survivors</span>");
      if (p.generation != null) bits.push("<span>generation <b>" + p.generation + "</b></span>");
      if (p.best_fitness != null) bits.push("<span>best fitness <b>" + esc(p.best_fitness) + "</b></span>");
      bits.push("<span>" + formatElapsed(p.elapsed_seconds) + " elapsed</span>");
      stats.innerHTML = bits.join("");
    }

    function tick() {
      getJSON(url).then(function (data) {
        var p = key ? data[key] : data.progress;
        if (p) render(p);
        // Keep polling while the run is live. For the shared Evolution status
        // endpoint (key set) the snapshot's own `done` decides; otherwise the
        // envelope's.
        var finished = p ? !!p.done : (data.found === false);
        // A job that exists but isn't tracked will never produce a snapshot.
        if (!p && !key && data.found === true) finished = true;
        if (finished) return;
        timer = setTimeout(tick, 2000);
      }).catch(function () { timer = setTimeout(tick, 4000); });
    }
    tick();
    return function stop() { if (timer) clearTimeout(timer); };
  }

  function formatElapsed(sec) {
    sec = Math.max(0, Math.round(sec || 0));
    var m = Math.floor(sec / 60), s = sec % 60;
    return m ? m + "m " + (s < 10 ? "0" : "") + s + "s" : s + "s";
  }

  // ------------------------------------------------------------------- swarm
  function drawSwarm(canvas, swarm) {
    var c = setupCanvas(canvas, 300), ctx = c.ctx, W = c.w, H = c.h;
    var padL = 44, padR = 10, padT = 14, padB = 22;
    var curves = swarm.curves || [];
    var vals = curves.map(function (cv) { return cv.values; });
    var rng = clipRange(vals), n = swarm.points || (vals[0] ? vals[0].length : 2);
    function X(i) { return padL + (W - padL - padR) * (i / Math.max(1, n - 1)); }
    function Y(v) { return padT + (H - padT - padB) * (1 - (v - rng.lo) / (rng.hi - rng.lo)); }

    ctx.clearRect(0, 0, W, H);
    // holdout shading + divider
    var sx = X(swarm.split_index);
    ctx.fillStyle = "rgba(255,255,255,0.045)";
    ctx.fillRect(sx, padT, W - padR - sx, H - padT - padB);
    // zero line + y labels
    ctx.strokeStyle = "rgba(255,255,255,0.18)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
    ctx.fillStyle = "#8a93a6"; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "right";
    [rng.lo, 0, rng.hi].forEach(function (v) { ctx.fillText(v.toFixed(0) + "%", padL - 6, Y(v) + 4); });

    // failing candidates first so survivors paint on top
    [false, true].forEach(function (holds) {
      curves.forEach(function (cv) {
        if (cv.holds_up !== holds) return;
        ctx.strokeStyle = "rgba(" + (holds ? COLORS.up : COLORS.down) + "," + (holds ? 0.42 : 0.22) + ")";
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (var i = 0; i < cv.values.length; i++) {
          var x = X(i), y = Y(Math.max(rng.lo, Math.min(rng.hi, cv.values[i])));
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
      });
    });

    // divider line + labels (drawn last so it is always visible)
    ctx.strokeStyle = "#f0b429"; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(sx, padT); ctx.lineTo(sx, H - padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = "#f0b429"; ctx.textAlign = "right"; ctx.fillText("in-sample", sx - 6, padT + 10);
    ctx.textAlign = "left"; ctx.fillText("holdout", sx + 6, padT + 10);
  }

  function initSwarm(el) {
    var url = el.getAttribute("data-swarm-url");
    var statusEl = el.querySelector("[data-role=status]");
    var noteEl = el.querySelector("[data-role=note]");
    var canvas = el.querySelector("[data-role=canvas]");
    var wrap = el.querySelector("[data-role=wrap]");
    var last = null, timer = null;

    var btn = document.createElement("button");
    btn.type = "button"; btn.className = "t58-tm-btn"; btn.textContent = "Refresh swarm"; btn.style.marginTop = "8px";
    el.appendChild(btn);
    btn.addEventListener("click", function () { if (timer) clearTimeout(timer); tick(true); });

    function show(msg, hideChart) {
      el.hidden = false; statusEl.textContent = msg;
      wrap.style.display = hideChart ? "none" : "";
    }

    function render(swarm) {
      last = swarm;
      var n = (swarm.curves || []).length;
      if (!n) { show("No candidate produced a drawable equity curve.", true); return; }
      show(swarm.holds_up_count + " of " + n + " candidates (" + formatPct(swarm.holds_up_pct, 0) +
           ") are profitable after the divider.", false);
      drawSwarm(canvas, swarm);
      var notes = ["Teal = still profitable on the holdout segment; red = not. Each curve is that candidate's " +
        "equity as % return, sampled to " + swarm.points + " points."];
      if (swarm.timed_out) notes.push("Drawing stopped at the time limit: showing " + swarm.computed + " of " + swarm.requested + " candidates.");
      if (swarm.failed) notes.push(swarm.failed + " candidate(s) couldn't be redrawn and are omitted.");
      notes.push("Search Lab scores candidates on the full history, so the holdout here is the segment the validation stages treat as unseen — not proof no candidate ever saw it.");
      noteEl.textContent = notes.join(" ");
    }

    function tick(refresh) {
      getJSON(url + (refresh ? (url.indexOf("?") < 0 ? "?" : "&") + "refresh=1" : "")).then(function (d) {
        if (d.status === "ready" && d.swarm) { render(d.swarm); return; }
        if (d.status === "running") {
          show("Redrawing candidates… " + (d.done || 0) + "/" + (d.total || "?"), true);
          timer = setTimeout(function () { tick(false); }, 1500); return;
        }
        if (d.status === "pending") { show(d.reason || "Waiting for the run to finish…", true); timer = setTimeout(function () { tick(false); }, 4000); return; }
        if (d.status === "error") { show("Couldn't build the swarm: " + (d.error || "unknown error"), true); return; }
        show(d.reason || "Swarm not available for this run.", true);
      }).catch(function () { timer = setTimeout(function () { tick(false); }, 5000); });
    }

    var resizeT = null;
    root.addEventListener("resize", function () {
      clearTimeout(resizeT);
      resizeT = setTimeout(function () { if (last && (last.curves || []).length) drawSwarm(canvas, last); }, 150);
    });
    tick(false);
  }

  // ---------------------------------------------------------------- survivors
  function initSurvivors(el) {
    var statusEl = el.querySelector("[data-role=status]");
    var funnel = el.querySelector("[data-role=funnel]");
    getJSON("/api/telemetry/survivors").then(function (d) {
      if (!d.stages) { statusEl.textContent = "Couldn't load the funnel."; return; }
      if (!d.total) { statusEl.textContent = "No strategies saved in the library yet."; funnel.innerHTML = ""; return; }
      statusEl.textContent = d.total + " strateg" + (d.total === 1 ? "y" : "ies") + " in the library" +
        (d.ready_verdict_count ? " · " + d.ready_verdict_count + " with a READY verdict" : "") + ".";
      funnel.innerHTML = d.stages.map(function (s) {
        return '<div class="t58-tm-fstage"><span>' + esc(s.title) + '</span><div class="t58-tm-fbar"><span style="width:' +
          Math.max(0, Math.min(100, s.pct_of_total)) + '%"></span></div><span class="t58-tm-fcount">' + s.count + "</span></div>";
      }).join("");
    }).catch(function () { statusEl.textContent = "Couldn't load the funnel."; });
  }

  // ----------------------------------------------------------------- universe
  function initUniverse(el) {
    var canvas = el.querySelector("[data-role=canvas]");
    var wrap = el.querySelector("[data-role=wrap]");
    var statusEl = el.querySelector("[data-role=status]");
    var legend = el.querySelector("[data-role=legend]");
    var srcBox = el.querySelector("[data-role=sources]");
    var tip = document.createElement("div");
    tip.className = "t58-tm-tip"; tip.hidden = true; wrap.appendChild(tip);

    var data = null, pts = [], hiddenStages = {}, hover = -1, familyColor = {};

    function sources() {
      var on = [];
      srcBox.querySelectorAll("input[type=checkbox]").forEach(function (cb) { if (cb.checked) on.push(cb.value); });
      return on;
    }

    function layout() {
      var c = setupCanvas(canvas, Math.round(Math.min(560, Math.max(320, (wrap.clientWidth || 320) * 0.9))));
      var W = c.w, H = c.h, m = 26, half = Math.min(W, H) / 2 - m;
      pts = [];
      (data.dots || []).forEach(function (d) {
        if (hiddenStages[d.stage]) return;
        pts.push({ d: d, px: W / 2 + d.x * half, py: H / 2 + d.y * half });
      });
      return c;
    }

    function draw() {
      if (!data) return;
      var c = layout(), ctx = c.ctx, W = c.w, H = c.h, half = Math.min(W, H) / 2 - 26;
      ctx.clearRect(0, 0, W, H);
      (data.clusters || []).forEach(function (cl) {
        var cx = W / 2 + cl.cx * half, cy = H / 2 + cl.cy * half, r = cl.r * half + 8;
        ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(cx, cy, r, 0, 6.2832); ctx.stroke();
      });
      // rejected first so survivors are never hidden underneath
      STAGE_ORDER.forEach(function (stage) {
        pts.forEach(function (p) {
          if (p.d.stage !== stage) return;
          var big = stage === "survivor" ? 4 : (stage === "validated" ? 3 : 2.2);
          ctx.fillStyle = STAGE_COLORS[stage];
          ctx.globalAlpha = stage === "rejected" ? 0.55 : 0.95;
          ctx.beginPath(); ctx.arc(p.px, p.py, big, 0, 6.2832); ctx.fill();
        });
      });
      ctx.globalAlpha = 1;
      ctx.font = "12px system-ui, sans-serif"; ctx.textAlign = "center";
      (data.clusters || []).forEach(function (cl) {
        var cx = W / 2 + cl.cx * half, cy = H / 2 + cl.cy * half, r = cl.r * half + 8;
        ctx.fillStyle = familyColor[cl.family] || "#8a93a6";
        var ly = cy - r - 5; if (ly < 12) ly = cy + r + 14;
        ctx.fillText(cl.label + " (" + cl.count + ")", cx, ly);
      });
      if (hover >= 0 && pts[hover]) {
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(pts[hover].px, pts[hover].py, 7, 0, 6.2832); ctx.stroke();
      }
    }

    function showTip(i) {
      hover = i;
      if (i < 0) { tip.hidden = true; draw(); return; }
      var d = pts[i].d;
      tip.innerHTML = "<b>" + esc(d.id) + "</b><br>" +
        '<span class="k">Family</span> ' + esc(d.family_label) + "<br>" +
        '<span class="k">Symbol</span> ' + esc(d.symbol) + " · " + esc(d.timeframe) + "<br>" +
        '<span class="k">Profit</span> <span class="' + (d.profit > 0 ? "t58-tm-pos" : (d.profit < 0 ? "t58-tm-neg" : "")) + '">' + formatMoney(d.profit) + "</span><br>" +
        '<span class="k">Max drawdown</span> ' + formatPct(d.drawdown_pct) + "<br>" +
        '<span class="k">Stage</span> ' + esc(STAGE_LABELS[d.stage] || d.stage) + ' <span class="k">(' + esc(d.source) + ")</span>";
      tip.hidden = false;
      var W = wrap.clientWidth, tw = tip.offsetWidth || 200;
      var left = Math.min(Math.max(4, pts[i].px + 12), W - tw - 4);
      tip.style.left = left + "px"; tip.style.top = Math.max(4, pts[i].py - 10) + "px";
      draw();
    }

    function onMove(ev) {
      if (!pts.length) return;
      var p = eventPoint(canvas, ev);
      var i = nearestDot(pts, p.x, p.y, ev.touches ? 16 : 9);
      if (i !== hover) showTip(i);
    }
    canvas.addEventListener("mousemove", onMove);
    canvas.addEventListener("mouseleave", function () { showTip(-1); });
    canvas.addEventListener("touchstart", onMove, { passive: true });
    canvas.addEventListener("touchmove", onMove, { passive: true });

    function buildLegend() {
      legend.innerHTML = "";
      STAGE_ORDER.forEach(function (stage) {
        var chip = document.createElement("span");
        chip.className = "t58-tm-chip" + (hiddenStages[stage] ? " off" : "");
        chip.innerHTML = '<span class="t58-tm-dot" style="background:' + STAGE_COLORS[stage] + '"></span>' + esc(STAGE_LABELS[stage]);
        chip.addEventListener("click", function () { hiddenStages[stage] = !hiddenStages[stage]; hover = -1; tip.hidden = true; buildLegend(); draw(); });
        legend.appendChild(chip);
      });
    }

    function load() {
      var on = sources();
      if (!on.length) { statusEl.textContent = "Pick at least one data source."; data = null; canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height); return; }
      statusEl.textContent = "Loading…";
      getJSON("/api/telemetry/universe?sources=" + encodeURIComponent(on.join(","))).then(function (d) {
        if (d.error) { statusEl.textContent = "Couldn't load the map: " + d.error; return; }
        data = d; hover = -1; tip.hidden = true;
        (d.clusters || []).forEach(function (cl, i) { familyColor[cl.family] = FAMILY_COLORS[i % FAMILY_COLORS.length]; });
        if (!d.total) { statusEl.textContent = "Nothing to show yet — run a Search Lab, Speed Run or Evolution Lab, or save a strategy."; }
        else {
          var surv = (d.clusters || []).reduce(function (a, cl) { return a + cl.survivors; }, 0);
          statusEl.textContent = d.shown.toLocaleString() + " strategies in " + d.clusters.length + " families · " + surv + " survivors" +
            (d.truncated ? " · showing the best " + d.shown.toLocaleString() + " of " + d.total.toLocaleString() : "") + ".";
        }
        buildLegend(); draw();
      }).catch(function () { statusEl.textContent = "Couldn't load the map."; });
    }

    srcBox.addEventListener("change", load);
    var rt = null;
    root.addEventListener("resize", function () { clearTimeout(rt); rt = setTimeout(draw, 150); });
    buildLegend(); load();
  }

  // -------------------------------------------------------------------- boot
  function boot() {
    document.querySelectorAll("[data-t58-phase-banner]").forEach(initBanner);
    document.querySelectorAll("[data-t58-swarm]").forEach(initSwarm);
    document.querySelectorAll("[data-t58-survivors]").forEach(initSurvivors);
    document.querySelectorAll("[data-t58-universe]").forEach(initUniverse);
  }

  root.T58Telemetry = { esc: esc, formatMoney: formatMoney, formatPct: formatPct, formatElapsed: formatElapsed,
                        clipRange: clipRange, nearestDot: nearestDot };
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
  }
})(typeof window !== "undefined" ? window : globalThis);
