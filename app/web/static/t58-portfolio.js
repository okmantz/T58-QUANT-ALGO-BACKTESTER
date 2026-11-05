/* Portfolio cards: one card per connected account (firm, live P&L, open and past trades, algo health).
 * Data: GET /api/portfolio. Mount with T58Portfolio.mount(el, {compact:true}). Safe without storage; escapes all text. */
(function () {
  'use strict';
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) { return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]; }); }
  function money(v) { if (v == null || isNaN(v)) return '—'; var n = Number(v); return (n < 0 ? '-' : '') + '$' + Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}); }
  function cls(v) { return v > 0 ? 'pf-pos' : (v < 0 ? 'pf-neg' : ''); }
  function when(ts) { if (!ts) return ''; try { return new Date(ts * 1000).toLocaleString(undefined, {month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}); } catch (e) { return ''; } }
  function tradeRows(list, empty) {
    if (!list.length) return '<div class="pf-empty">' + esc(empty) + '</div>';
    return list.map(function (t) {
      return '<div class="pf-tr"><span class="pf-dir ' + t.direction + '">' + esc(t.direction) + '</span><span>' + esc(t.symbol) + ' · ' + esc(t.volume) + '</span>' +
        '<span class="pf-mut">' + esc(when(t.exit_time || t.entry_time)) + '</span><b class="' + cls(t.pnl) + '">' + (t.pnl == null ? 'open' : money(t.pnl)) + '</b></div>';
    }).join('');
  }
  function card(a, compact) {
    var h = a.health || {};
    var live = a.balance != null;
    return '<article class="pf-card">' +
      '<header><div><div class="pf-name">' + esc(a.nickname) + '</div><div class="pf-sub">' + esc(a.firm || a.platform) + ' · ' + esc(a.platform) + '</div></div>' +
      '<span class="pf-health ' + esc(h.state) + '" title="' + esc(h.why) + '">' + esc(h.label) + '</span></header>' +
      '<div class="pf-nums"><div><span class="pf-k" data-metric="net_profit">Net P&amp;L</span><b class="' + cls(a.net_pnl) + '">' + money(a.net_pnl) + '</b></div>' +
      '<div><span class="pf-k">' + (live ? 'Equity' : 'Floating') + '</span><b>' + (live ? money(a.equity) : '—') + '</b></div>' +
      '<div><span class="pf-k" data-metric="win_rate">Win rate</span><b>' + (a.win_rate == null ? '—' : a.win_rate.toFixed(0) + '%') + '</b></div>' +
      '<div><span class="pf-k" data-metric="trades">Trades</span><b>' + a.n_closed + '</b></div></div>' +
      '<div class="pf-why">' + esc(h.why) + '</div>' +
      (compact ? '' : '<div class="pf-sec">Open now</div>' + tradeRows(a.open_trades, 'No open trades.') +
        '<div class="pf-sec">Recent closed</div>' + tradeRows(a.past_trades, 'No closed trades yet.')) +
      '</article>';
  }
  function render(el, d, opts) {
    opts = opts || {};
    if (!d.accounts || !d.accounts.length) { el.innerHTML = '<div class="pf-empty">' + esc(opts.emptyText || 'No account connected yet.') + '</div>'; return; }
    el.innerHTML = '<div class="pf-total">Total net P&amp;L <b class="' + cls(d.total_net_pnl) + '">' + money(d.total_net_pnl) + '</b></div><div class="pf-grid">' +
      d.accounts.map(function (a) { return card(a, opts.compact); }).join('') + '</div>';
    if (window.T58Metrics && window.T58Metrics.scan) { try { window.T58Metrics.scan(el); } catch (e) {} }
  }
  function load(cb) { fetch('/api/portfolio', {headers: {'X-Requested-With': 'XMLHttpRequest'}}).then(function (r) { return r.json(); }).then(cb).catch(function () { cb({accounts: [], eligible: false}); }); }
  function mount(el, opts) {
    if (!el) return;
    function tick() { load(function (d) { render(el, d, opts); if (opts && opts.onData) opts.onData(d); }); }
    tick(); setInterval(function () { if (!document.hidden) tick(); }, 15000);
  }
  function enabled() { try { return localStorage.getItem('t58-portfolio') === '1'; } catch (e) { return false; } }
  function setEnabled(on) { try { localStorage.setItem('t58-portfolio', on ? '1' : '0'); } catch (e) {} }
  window.T58Portfolio = {mount: mount, load: load, enabled: enabled, setEnabled: setEnabled};
})();
