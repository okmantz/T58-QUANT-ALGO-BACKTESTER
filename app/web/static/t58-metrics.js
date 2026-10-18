/* T58 clickable metrics (2026-10-17).
 * Click any metric -- Profit Factor, Max Drawdown, T58 Score, Win Rate, Pass Rate ... -- and a popup opens with what it
 * means, the formula, the inputs of the calculation and its history from the run log: the same idea as the "why NOT READY"
 * verdict popup.
 *
 * Two ways a metric becomes clickable:
 *   1. explicit:  <span data-metric="profit_factor" data-metric-values='{"Gross profit":"$1,200","Gross loss":"$400"}'>
 *   2. automatic: any small leaf element whose whole text is a known metric label ("Profit Factor", "Max Drawdown",
 *      "T58 Score" ...) inside a stat card / table header gets the same behaviour. (Not inside forms/buttons/links/options.)
 * data-metric-values (optional) supplies THIS number's actual inputs so the popup shows the real calculation.
 * data-metric-value (optional) shows the displayed value at the top of the popup.
 */
(function () {
  if (window.T58Metrics) return;
  var labels = null, labelsPromise = null, box = null, lastEl = null;
  var SKIP = { A: 1, BUTTON: 1, OPTION: 1, SELECT: 1, INPUT: 1, TEXTAREA: 1, SCRIPT: 1, STYLE: 1, LABEL: 1, H1: 1, TITLE: 1 };

  function norm(s) { return String(s == null ? "" : s).replace(/\s+/g, " ").trim().toLowerCase(); }
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }

  function css() {
    if (document.getElementById("t58-metric-css")) return;
    var s = document.createElement("style"); s.id = "t58-metric-css";
    s.textContent =
      "[data-metric]{cursor:pointer;text-decoration:underline dotted;text-underline-offset:3px;text-decoration-color:var(--text-dim,#5b6478)}" +
      "[data-metric]:hover{text-decoration-color:var(--teal,#35e0b0)}[data-metric]:focus-visible{outline:2px solid var(--teal,#35e0b0);outline-offset:2px}" +
      ".t58-mx-back{position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;padding:16px}" +
      ".t58-mx{position:relative;width:min(460px,100%);max-height:88vh;overflow:auto;background:var(--panel-2,#12161f);color:var(--text,#e8ecf3);border:1px solid var(--border-light,#2a3142);border-radius:14px;padding:18px 18px 16px;box-shadow:0 18px 50px rgba(0,0,0,.55);font-size:13.5px;line-height:1.5}" +
      ".t58-mx h3{margin:0 28px 2px 0;font-size:17px}.t58-mx .val{font-size:26px;font-weight:700;color:var(--teal,#35e0b0);margin:4px 0 8px}" +
      ".t58-mx h5{margin:14px 0 5px;font-size:11px;letter-spacing:.07em;text-transform:uppercase;color:var(--text-muted,#8a93a6)}" +
      ".t58-mx p{margin:0 0 4px}.t58-mx ul{margin:0;padding-left:18px}.t58-mx li{margin:0 0 4px}" +
      ".t58-mx .formula{font-family:ui-monospace,Menlo,Consolas,monospace;background:var(--panel-3,#151a24);border:1px solid var(--border,#1c2230);border-radius:8px;padding:9px 10px;font-size:12.5px}" +
      ".t58-mx .x{position:absolute;top:8px;right:10px;background:none;border:0;color:inherit;font-size:22px;cursor:pointer;line-height:1}" +
      ".t58-mx table{width:100%;border-collapse:collapse;font-size:12.5px}.t58-mx td{padding:4px 2px;border-bottom:1px solid var(--border,#1c2230)}.t58-mx td:last-child{text-align:right;font-variant-numeric:tabular-nums}" +
      ".t58-mx .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:8px}.t58-mx .stats div{background:var(--panel-3,#151a24);border-radius:8px;padding:6px 8px}" +
      ".t58-mx .stats b{display:block;font-size:14px}.t58-mx .stats span{font-size:10.5px;color:var(--text-muted,#8a93a6);text-transform:uppercase}" +
      ".t58-mx .muted{color:var(--text-muted,#8a93a6)}";
    document.head.appendChild(s);
  }

  function loadLabels() {
    if (labels) return Promise.resolve(labels);
    if (!labelsPromise) {
      labelsPromise = fetch("/api/metrics").then(function (r) { return r.json(); })
        .then(function (d) { labels = (d && d.labels) || {}; return labels; })
        .catch(function () { labels = {}; return labels; });
    }
    return labelsPromise;
  }

  function close() {
    if (box) { box.remove(); box = null; }
    if (lastEl && lastEl.focus) { try { lastEl.focus(); } catch (e) {} }
    lastEl = null;
  }

  function spark(rows) {
    var vals = rows.map(function (r) { return r.value; });
    if (vals.length < 2) return null;
    var w = 400, h = 56, mn = Math.min.apply(null, vals), mx = Math.max.apply(null, vals), rg = (mx - mn) || 1;
    var pts = vals.map(function (v, i) { return (i / (vals.length - 1) * w).toFixed(1) + "," + (h - 4 - (v - mn) / rg * (h - 8)).toFixed(1); }).join(" ");
    var ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", "100%"); svg.setAttribute("height", "56"); svg.setAttribute("preserveAspectRatio", "none");
    var pl = document.createElementNS(ns, "polyline"); pl.setAttribute("points", pts); pl.setAttribute("fill", "none"); pl.setAttribute("stroke", "#35e0b0"); pl.setAttribute("stroke-width", "2");
    svg.appendChild(pl); return svg;
  }

  function fmt(v, unit) {
    if (v == null) return "--";
    var n = Number(v); if (!isFinite(n)) return String(v);
    var s = Math.abs(n) >= 1000 ? n.toLocaleString(undefined, { maximumFractionDigits: 0 }) : String(Math.round(n * 100) / 100);
    return unit === "$" ? (n < 0 ? "-$" + s.replace("-", "") : "$" + s) : s + (unit === "%" ? "%" : (unit === "x" ? "x" : ""));
  }

  function open(src, key) {
    var wasSame = box && box._src === src; close(); if (wasSame) return;
    css(); lastEl = src;
    fetch("/api/metrics/" + encodeURIComponent(key)).then(function (r) { return r.json(); }).then(function (d) {
      if (!d || !d.metric) return;
      var m = d.metric, back = el("div", "t58-mx-back"), card = el("div", "t58-mx");
      back._src = src; box = back; card.setAttribute("role", "dialog"); card.setAttribute("aria-label", m.title);
      var x = el("button", "x", "×"); x.setAttribute("aria-label", "Close"); x.addEventListener("click", close); card.appendChild(x);
      card.appendChild(el("h3", null, m.title));
      var shown = src.getAttribute("data-metric-value"); if (shown) card.appendChild(el("div", "val", shown));
      card.appendChild(el("p", null, m.meaning));
      card.appendChild(el("h5", null, "Formula")); card.appendChild(el("div", "formula", m.formula));
      var vals = null; try { vals = JSON.parse(src.getAttribute("data-metric-values") || "null"); } catch (e) {}
      if (vals && Object.keys(vals).length) {
        card.appendChild(el("h5", null, "Calculation inputs (this number)"));
        var t = el("table"); Object.keys(vals).forEach(function (k) { var tr = el("tr"); tr.appendChild(el("td", null, k)); tr.appendChild(el("td", null, String(vals[k]))); t.appendChild(tr); }); card.appendChild(t);
      } else if (m.inputs && m.inputs.length) {
        card.appendChild(el("h5", null, "What goes into it")); var ul = el("ul"); m.inputs.forEach(function (i) { ul.appendChild(el("li", null, i)); }); card.appendChild(ul);
      }
      if (m.reading && m.reading.length) { card.appendChild(el("h5", null, "How to read it")); var ul2 = el("ul"); m.reading.forEach(function (i) { ul2.appendChild(el("li", null, i)); }); card.appendChild(ul2); }
      var hist = el("div"); card.appendChild(hist);
      back.appendChild(card); document.body.appendChild(back);
      back.addEventListener("click", function (e) { if (e.target === back) close(); });
      if (m.history_field) {
        fetch("/api/metrics/" + encodeURIComponent(m.key) + "/history").then(function (r) { return r.json(); }).then(function (h) {
          hist.appendChild(el("h5", null, "History across your runs"));
          if (!h.rows || !h.rows.length) { hist.appendChild(el("p", "muted", "No runs recorded yet — run a backtest and its values will appear here.")); return; }
          var s = h.summary, g = el("div", "stats");
          [["Latest", s.latest], ["Average", s.average], ["Best", s.best], ["Worst", s.worst]].forEach(function (p) { var c = el("div"); c.appendChild(el("b", null, fmt(p[1], m.unit))); c.appendChild(el("span", null, p[0])); g.appendChild(c); });
          hist.appendChild(g);
          var sp = spark(h.rows); if (sp) hist.appendChild(sp);
          var t2 = el("table"); h.rows.slice(-6).reverse().forEach(function (r) { var tr = el("tr"); tr.appendChild(el("td", null, r.date + " · " + (r.strategy || "") + (r.instrument ? " (" + r.instrument + ")" : ""))); tr.appendChild(el("td", null, fmt(r.value, m.unit))); t2.appendChild(tr); }); hist.appendChild(t2);
        }).catch(function () {});
      }
    }).catch(function () {});
  }

  function bind(node, key) {
    if (node.getAttribute("data-metric")) return;
    node.setAttribute("data-metric", key); node.setAttribute("role", "button"); node.setAttribute("tabindex", "0");
    node.setAttribute("title", "Click for the definition, formula and history");
  }

  function scan(root) {
    css();
    loadLabels().then(function (map) {
      var nodes = (root || document).querySelectorAll("th, td, span, div, p, small, h2, h3, h4, dt, strong, b, em");
      for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        if (n.children.length || n.getAttribute("data-metric") || n.closest("a,button,select,option,label,form,[data-ve],.t58-mx,nav,.t58-sidebar,#t58-sidebar")) continue;
        var t = norm(n.textContent); if (!t || t.length > 40) continue;
        var key = map[t.replace(/:$/, "")]; if (key) bind(n, key);
      }
    });
  }

  document.addEventListener("click", function (e) {
    var t = e.target.closest && e.target.closest("[data-metric]");
    if (t && !t.closest(".t58-mx")) { e.preventDefault(); e.stopPropagation(); open(t, t.getAttribute("data-metric")); }
  }, true);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") close();
    if ((e.key === "Enter" || e.key === " ") && e.target.getAttribute && e.target.getAttribute("data-metric")) { e.preventDefault(); open(e.target, e.target.getAttribute("data-metric")); }
  });

  window.T58Metrics = { scan: scan, open: open, bind: bind };
  function start() { scan(document); var t = null; new MutationObserver(function () { clearTimeout(t); t = setTimeout(function () { scan(document); }, 400); }).observe(document.body, { childList: true, subtree: true }); }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start); else start();
})();
