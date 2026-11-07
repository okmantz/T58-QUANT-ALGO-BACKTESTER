/* Header P&L tracker (top right, beside the bell and profile picture). Polls /api/header-pnl:
 * Demo = the T58 demo prop account, Live = running real sessions. Text only, never innerHTML. */
(function () {
  'use strict';
  var chip = document.getElementById('t58-pnl-chip'), val = document.getElementById('t58-pnl-val');
  if (!chip || !val) return;
  function money(v) {
    if (v == null || isNaN(v)) return '$0.00';
    var s = Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    return (v < 0 ? '−' : v > 0 ? '+' : '') + '$' + s;
  }
  function tick() {
    fetch('/api/header-pnl', { headers: { 'Accept': 'application/json' } }).then(function (r) { return r.json(); }).then(function (d) {
      chip.setAttribute('data-mode', d.mode);
      chip.querySelector('.t58-pnl-mode').textContent = String(d.mode).toUpperCase();
      val.textContent = money(d.pnl);
      chip.classList.toggle('up', d.pnl > 0); chip.classList.toggle('down', d.pnl < 0);
      chip.classList.toggle('bad', d.status === 'breached');
      chip.title = d.label + (d.status && d.status !== 'active' && d.status !== 'idle' ? ' · ' + d.status.toUpperCase() : '') +
        (d.day_pnl != null ? ' · today ' + money(d.day_pnl) : '');
      chip.setAttribute('href', d.mode === 'demo' ? '/demo-account' : '/deploy-live');
    }).catch(function () {});
  }
  tick(); setInterval(tick, 5000);
})();
