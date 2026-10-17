/*
 * Evolution Lab live gallery. Server side: app/web/evolution_lab_routes.py.
 *
 * Design for "does not slow the rest of the app": one small JSON poll (an in-memory ring buffer on the server),
 * paused while the tab is hidden, at most MAX_CARDS canvases alive, each card drawn once per update (a 90-point
 * polyline), DOM writes batched in one requestAnimationFrame. All text goes through textContent -- nothing from the
 * server is ever interpreted as HTML.
 */
(function (root) {
  "use strict";
  var MAX_CARDS = 48, POLL_MS = 2500, state = { seq: 0, cards: {}, order: [], filter: "all", timer: null, busy: false };

  function $(id) { return document.getElementById(id); }
  function fmt(v, d) { return (v == null || !isFinite(v)) ? "n/a" : Number(v).toFixed(d == null ? 1 : d); }
  function money(v) { return (v == null || !isFinite(v)) ? "n/a" : (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 }); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  // Pure helper (unit-testable): colour + scaling for a curve.
  function curveGeometry(curve, w, h) {
    if (!curve || curve.length < 2) return null;
    var lo = Math.min.apply(null, curve), hi = Math.max.apply(null, curve);
    if (hi === lo) { hi = lo + 1; }
    var pts = curve.map(function (v, i) {
      return [i / (curve.length - 1) * (w - 2) + 1, h - 3 - (v - lo) / (hi - lo) * (h - 6)];
    });
    return { pts: pts, up: curve[curve.length - 1] >= curve[0], base: h - 3 - (curve[0] - lo) / (hi - lo) * (h - 6) };
  }

  function drawCurve(canvas, curve) {
    var dpr = root.devicePixelRatio || 1, w = canvas.clientWidth || 200, h = canvas.clientHeight || 64;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    var ctx = canvas.getContext("2d"); if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, w, h);
    var g = curveGeometry(curve, w, h); if (!g) return;
    var rgb = g.up ? "53,224,176" : "255,111,111";
    ctx.strokeStyle = "rgba(255,255,255,0.10)"; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(0, g.base); ctx.lineTo(w, g.base); ctx.stroke(); ctx.setLineDash([]);
    var grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, "rgba(" + rgb + ",0.30)"); grad.addColorStop(1, "rgba(" + rgb + ",0)");
    ctx.beginPath(); ctx.moveTo(g.pts[0][0], h);
    g.pts.forEach(function (p) { ctx.lineTo(p[0], p[1]); });
    ctx.lineTo(g.pts[g.pts.length - 1][0], h); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
    ctx.beginPath(); g.pts.forEach(function (p, i) { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); });
    ctx.strokeStyle = "rgb(" + rgb + ")"; ctx.lineWidth = 1.6; ctx.lineJoin = "round"; ctx.stroke();
  }

  function visible(item) {
    var f = state.filter;
    if (f === "all") return true;
    if (f === "elite") return !!item.elite;
    if (f === "evaluated") return item.stage === "evaluated";
    if (f === "control") return !!item.control;
    return true;
  }

  function buildCard(item) {
    var card = el("div", "elab-card fresh"), t = el("div", "t"), name = el("b", null, item.control ? "random control" : item.id);
    name.title = item.id; t.appendChild(name);
    var tag = el("span", "elab-tag " + (item.control ? "control" : (item.tribe || "")), item.control ? "control" : (item.tribe || (item.family || "")));
    t.appendChild(tag); card.appendChild(t);
    var cv = el("canvas"); card.appendChild(cv);
    var m = el("div", "m"); card.appendChild(m);
    card._cv = cv; card._m = m; card._tag = tag;
    return card;
  }

  function fillCard(card, item) {
    card.classList.toggle("elite", !!item.elite);
    card.classList.toggle("control", !!item.control);
    card.classList.toggle("rejected", item.stage === "prefilter" && item.passed === false);
    card._item = item;
    var m = card._m; while (m.firstChild) m.removeChild(m.firstChild);
    function stat(label, val) { var s = el("span", null, label + " "); s.appendChild(el("b", null, val)); m.appendChild(s); }
    stat("gen", String(item.gen));
    stat("net", money(item.net_profit));
    if (item.n_trades != null) stat("trades", String(item.n_trades));
    if (item.max_drawdown_pct != null) stat("DD", fmt(item.max_drawdown_pct) + "%");
    if (item.score != null) stat("score", fmt(item.score));
    if (item.pass_probability != null) stat("pass", fmt(item.pass_probability * 100, 0) + "%");
    if (item.payout_probability != null) stat("payout", fmt(item.payout_probability * 100, 0) + "%");
    if (item.decay15 && item.decay15.decayed) stat("decay", "yes");
    if (item.elite) stat("★", "elite");
    root.requestAnimationFrame(function () { drawCurve(card._cv, item.curve); });
  }

  function render(items) {
    var grid = $("elab-grid"); if (!grid) return;
    var empty = grid.querySelector(".elab-empty"); if (empty) grid.removeChild(empty);
    items.forEach(function (it) {
      var card = state.cards[it.id];
      if (!card) {
        card = buildCard(it); state.cards[it.id] = card; state.order.push(it.id);
        grid.insertBefore(card, grid.firstChild);
      } else {
        card.classList.remove("fresh"); void card.offsetWidth; card.classList.add("fresh");
        grid.insertBefore(card, grid.firstChild);
      }
      fillCard(card, it);
    });
    while (state.order.length > MAX_CARDS) {      // keep the DOM (and canvases) bounded
      var old = state.order.shift(), c = state.cards[old];
      if (c && c.parentNode) c.parentNode.removeChild(c);
      delete state.cards[old];
    }
    applyFilter();
  }

  function applyFilter() {
    state.order.forEach(function (id) { var c = state.cards[id]; if (c) c.style.display = visible(c._item) ? "" : "none"; });
  }

  function renderLab(lab) {
    if (!lab) return;
    var chips = $("elab-chips"); while (chips.firstChild) chips.removeChild(chips.firstChild);
    var c1 = el("span", "elab-chip", "generation " + lab.generation); chips.appendChild(c1);
    var c2 = el("span", "elab-chip " + (lab.tribes_active ? "on" : ""), lab.tribes_active
      ? "tribes apart until gen " + lab.tribe_merge_generation : "tribes merged"); chips.appendChild(c2);
    var box = $("elab-controls"), c = lab.controls || {};
    while (box.firstChild) box.removeChild(box.firstChild);
    var head = el("div"); head.appendChild(el("b", null, "Does the search beat chance? "));
    head.appendChild(document.createTextNode(c.verdict || "")); box.appendChild(head);
    if (c.evolved_tested != null) {
      var bars = el("div", "elab-bars");
      [["Evolved", c.evolved_pass_rate, c.evolved_passed, c.evolved_tested, "var(--teal)"],
       ["Random control", c.control_pass_rate, c.control_passed, c.control_tested, "var(--text-dim)"]].forEach(function (r) {
        var row = el("div", "elab-bar"); row.appendChild(el("span", null, r[0]));
        var bar = el("i"), fill = el("em"); fill.style.width = Math.min(100, (r[1] || 0) * 100) + "%"; fill.style.background = r[4];
        bar.appendChild(fill); row.appendChild(bar); row.appendChild(el("span", null, r[2] + "/" + r[3]));
        bars.appendChild(row);
      });
      box.appendChild(bars);
    }
  }

  function poll() {
    if (state.busy || document.hidden) return;
    state.busy = true;
    fetch("/api/evolution/lab/feed?since=" + state.seq, { headers: { "Accept": "application/json" } })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        var card = $("elab"); if (!card) return;
        if (d.reset) { state.seq = 0; state.cards = {}; state.order = []; $("elab-grid").textContent = ""; }
        if (d.lab) card.hidden = false;
        if (d.items && d.items.length) { render(d.items); }
        state.seq = d.seq || state.seq;
        renderLab(d.lab);
      })
      .catch(function () { /* transient: next poll retries */ })
      .then(function () { state.busy = false; });
  }

  function finalists() {
    var out = $("elab-fin-out"); out.textContent = "Computing… (runs in the background; this can take a minute)";
    (function loop() {
      fetch("/api/evolution/lab/finalists").then(function (r) { return r.json(); }).then(function (d) {
        if (d.status === "computing") { root.setTimeout(loop, 3000); return; }
        out.textContent = "";
        if (d.error) { out.appendChild(el("p", "help", "Could not compute: " + d.error)); }
        if (!d.rows || !d.rows.length) { out.appendChild(el("p", "help", "No finalist has enough trades for a full survival analysis yet.")); return; }
        var tbl = el("table"), th = el("tr");
        ["Candidate", "Pass eval", "Payout 1", "Parity", "Readiness", "Start risk"].forEach(function (h) { th.appendChild(el("th", null, h)); });
        tbl.appendChild(th);
        d.rows.forEach(function (r) {
          var tr = el("tr");
          tr.appendChild(el("td", null, r.candidate_id));
          tr.appendChild(el("td", null, fmt(r.probability_pass_evaluation) + "%"));
          tr.appendChild(el("td", null, fmt(r.probability_first_payout) + "%"));
          tr.appendChild(el("td", null, (r.parity && r.parity.status) || "NOT_RUN"));
          tr.appendChild(el("td", (r.readiness || "").indexOf("NOT") === 0 ? "notready" : "ready", r.readiness || ""));
          var p0 = r.incubation_plan && r.incubation_plan[0];
          tr.appendChild(el("td", null, p0 ? fmt(p0.risk_pct, 2) + "% for the first " + (r.incubation_plan[1] ? r.incubation_plan[1].from_live_trade : "?") + " live trades" : ""));
          tbl.appendChild(tr);
        });
        out.appendChild(tbl);
        var par = el("div", "elab-parity"), sel = el("select");
        d.rows.forEach(function (r) { var o = el("option", null, r.candidate_id); o.value = r.candidate_id; sel.appendChild(o); });
        var file = el("input"); file.type = "file"; file.accept = ".csv,text/csv";
        var btn = el("button", null, "Compare with MT5 / account export"); btn.type = "button";
        var msg = el("span", "help");
        btn.onclick = function () {
          if (!file.files[0]) { msg.textContent = "Choose a CSV export first (entry time, type/direction, profit)."; return; }
          var fd = new FormData(); fd.append("candidate_id", sel.value); fd.append("file", file.files[0]);
          var h = {}, tok = $("elab").getAttribute("data-csrf");
          if (tok) { h["X-CSRF-Token"] = tok; fd.append("_csrf_token", tok); }
          fetch("/api/evolution/lab/parity", { method: "POST", body: fd, headers: h }).then(function (r) { return r.json(); }).then(function (j) {
            msg.textContent = j.error ? j.error : ("Parity " + j.parity.status + " — " + (j.parity.reasons || []).join(" "));
            if (!j.error) finalists();
          });
        };
        par.appendChild(sel); par.appendChild(file); par.appendChild(btn); par.appendChild(msg); out.appendChild(par);
      });
    })();
  }

  function init() {
    if (!$("elab")) return;
    document.querySelectorAll("[data-elab-filter]").forEach(function (b) {
      b.addEventListener("click", function () {
        state.filter = b.getAttribute("data-elab-filter");
        document.querySelectorAll("[data-elab-filter]").forEach(function (x) { x.classList.toggle("on", x === b); });
        applyFilter();
      });
    });
    var fb = $("elab-fin-btn"); if (fb) fb.addEventListener("click", finalists);
    poll(); state.timer = root.setInterval(poll, POLL_MS);
    document.addEventListener("visibilitychange", function () { if (!document.hidden) poll(); });
  }

  root.T58EvolutionLab = { curveGeometry: curveGeometry };
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
  }
})(typeof window !== "undefined" ? window : globalThis);
