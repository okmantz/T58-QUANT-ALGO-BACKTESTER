/* Floating Live P&L tracker. Polls /api/live-pnl (running live-deploy + forward-test sessions) and shows a small
 * pill at the bottom of every page while something is running: total P&L (closed + floating) with a health dot.
 * Click it for the per-session breakdown and an equity sparkline. Hidden entirely when nothing is running.
 * All values go into the DOM as text. Dismiss with x (stays hidden for this tab until a new session starts). */
(function () {
  'use strict';
  if (window.__t58LivePnl) return; window.__t58LivePnl = true;
  var FAST = 5000, SLOW = 20000, MAXPTS = 80;
  var hist = {}, open = false, dismissedKey = null, timer = null, root = null, lastRows = [];
  try { open = sessionStorage.getItem('t58-pnl-open') === '1'; } catch (e) {}

  function money(v, sign) {
    if (v == null || isNaN(v)) return '—';
    var s = Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return (v < 0 ? '−' : (sign && v > 0 ? '+' : '')) + '$' + s;
  }
  function el(tag, cls, text) { var n = document.createElement(tag); if (cls) n.className = cls; if (text != null) n.textContent = text; return n; }
  function tone(v) { return v > 0 ? 'up' : v < 0 ? 'down' : ''; }
  function spark(points) {
    var ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 120 28'); svg.setAttribute('class', 'pnl-spark'); svg.setAttribute('aria-hidden', 'true');
    if (points.length < 2) return svg;
    var min = Math.min.apply(null, points), max = Math.max.apply(null, points), span = max - min || 1;
    var d = points.map(function (v, i) { return (i ? 'L' : 'M') + (i * 120 / (points.length - 1)).toFixed(1) + ' ' + (26 - (v - min) / span * 24).toFixed(1); }).join(' ');
    var path = document.createElementNS(ns, 'path'); path.setAttribute('d', d); path.setAttribute('fill', 'none');
    path.setAttribute('stroke-width', '1.6'); path.setAttribute('stroke', points[points.length - 1] >= points[0] ? 'var(--lm-up, #3ed685)' : 'var(--lm-down, #f0596a)');
    svg.appendChild(path); return svg;
  }
  function ensureRoot() {
    if (root) return root;
    root = el('div', 't58-pnl'); root.id = 't58-pnl'; root.hidden = true;
    document.body.appendChild(root); return root;
  }
  function total(r) { return (r.net_pnl || 0) + (r.floating || 0); }

  function render(rows) {
    var host = ensureRoot();
    var key = rows.map(function (r) { return r.kind; }).join('|');
    if (!rows.length) { host.hidden = true; dismissedKey = null; return; }
    if (dismissedKey && dismissedKey === key) { host.hidden = true; return; }
    host.hidden = false; host.textContent = '';
    var sum = rows.reduce(function (a, r) { return a + total(r); }, 0);
    var bad = rows.some(function (r) { return r.alert || r.halted_reason; });
    var live = rows.some(function (r) { return r.kind === 'live'; });

    if (open) {
      var panel = el('div', 'pnl-panel');
      rows.forEach(function (r) {
        var card = el('div', 'pnl-card' + ((r.alert || r.halted_reason) ? ' warn' : ''));
        var head = el('div', 'pnl-head'); head.appendChild(el('strong', null, r.label));
        head.appendChild(el('span', 'pnl-state', r.alert ? 'ALERT' : r.halted_reason ? 'HALTED' : r.running ? (r.connected ? 'RUNNING' : 'CONNECTING') : 'STOPPED'));
        card.appendChild(head);
        var big = el('div', 'pnl-big ' + tone(total(r)), money(total(r), true)); card.appendChild(big);
        var key2 = r.kind, h = hist[key2] || [];
        card.appendChild(spark(h));
        var grid = el('div', 'pnl-grid');
        [['Balance', money(r.balance)], ['Equity', money(r.equity)], ['Open', money(r.floating, true)], ['Closed', money(r.net_pnl, true)],
         ['Trades', r.n_trades_closed == null ? '—' : String(r.n_trades_closed)], ['Win rate', r.win_rate == null ? '—' : (r.win_rate * (r.win_rate <= 1 ? 100 : 1)).toFixed(0) + '%']
        ].forEach(function (kv) { var c = el('div'); c.appendChild(el('span', 'k', kv[0])); c.appendChild(el('span', 'v', kv[1])); grid.appendChild(c); });
        card.appendChild(grid);
        var note = r.alert || r.halted_reason || r.drift_flag;
        if (note) card.appendChild(el('div', 'pnl-note', String(note)));
        panel.appendChild(card);
      });
      var a = el('a', 'pnl-link', 'Open Deploy →'); a.href = '/deploy-live'; panel.appendChild(a);
      host.appendChild(panel);
    }
    var pill = el('div', 'pnl-pill' + (bad ? ' bad' : ''));
    var btn = el('button', 'pnl-main'); btn.type = 'button'; btn.setAttribute('aria-expanded', open ? 'true' : 'false'); btn.title = 'Live P&L — click for details';
    btn.appendChild(el('i', 'pnl-dot'));
    btn.appendChild(el('span', 'pnl-tag', bad ? 'HALTED' : (live ? 'LIVE' : 'DEMO')));
    btn.appendChild(el('span', 'pnl-sum ' + tone(sum), money(sum, true)));
    btn.addEventListener('click', function () { open = !open; try { sessionStorage.setItem('t58-pnl-open', open ? '1' : '0'); } catch (e) {} render(lastRows); });
    var x = el('button', 'pnl-x', '×'); x.type = 'button'; x.title = 'Hide for now'; x.setAttribute('aria-label', 'Hide live P&L');
    x.addEventListener('click', function () { dismissedKey = key; render(lastRows); });
    pill.appendChild(btn); pill.appendChild(x); host.appendChild(pill);
  }

  function tick() {
    clearTimeout(timer);
    if (document.hidden) { timer = setTimeout(tick, SLOW); return; }
    fetch('/api/live-pnl', { credentials: 'same-origin', cache: 'no-store' }).then(function (r) { return r.ok ? r.json() : { sessions: [] }; })
      .catch(function () { return { sessions: [] }; }).then(function (d) {
        lastRows = d.sessions || [];
        lastRows.forEach(function (r) { var h = hist[r.kind] = hist[r.kind] || []; if (r.equity != null) { h.push(r.equity); if (h.length > MAXPTS) h.shift(); } });
        Object.keys(hist).forEach(function (k) { if (!lastRows.some(function (r) { return r.kind === k; })) delete hist[k]; });
        render(lastRows);
        timer = setTimeout(tick, lastRows.length ? FAST : SLOW);
      });
  }
  function boot() { tick(); document.addEventListener('visibilitychange', function () { if (!document.hidden) tick(); }); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
