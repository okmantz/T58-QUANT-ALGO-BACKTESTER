/* Achievement unlock toasts. Runs on every page (loaded from _sidebar.html): one tiny request a couple of seconds after load,
   shows at most 3 toasts, then marks them seen. Silent when nothing is new or the request fails. */
(function () {
  if (location.pathname.indexOf('/job/') >= 0) return;
  var TIER = { easy: '#35e0b0', medium: '#4d90ff', hard: '#9b8cff', elite: '#f5b83d' };
  function toast(a, i) {
    var el = document.createElement('div');
    el.setAttribute('role', 'status');
    el.style.cssText = 'position:fixed;right:16px;bottom:' + (84 + i * 76) + 'px;z-index:9000;max-width:320px;padding:11px 14px 11px 12px;border-radius:12px;' +
      'background:var(--pop-bg,#10181f);color:var(--text,#fff);border:1px solid ' + (TIER[a.tier] || '#35e0b0') + ';box-shadow:0 10px 30px rgba(0,0,0,.35);' +
      'display:flex;gap:10px;align-items:center;font-size:13px;opacity:0;transform:translateY(10px);transition:opacity .35s,transform .35s;cursor:pointer';
    el.innerHTML = '<span style="flex:none;width:30px;height:30px;border-radius:50%;display:grid;place-items:center;background:' + (TIER[a.tier] || '#35e0b0') + ';color:#04120e;font-weight:800">&#9733;</span>' +
      '<span><b style="display:block">Achievement unlocked: ' + String(a.title).replace(/</g, '&lt;') + '</b><span style="color:var(--text-muted,#9ab)">+' + a.xp + ' XP &middot; tap to view</span></span>';
    el.addEventListener('click', function () { location.href = '/achievements'; });
    document.body.appendChild(el);
    requestAnimationFrame(function () { el.style.opacity = '1'; el.style.transform = 'none'; });
    setTimeout(function () { el.style.opacity = '0'; setTimeout(function () { el.remove(); }, 400); }, 7000 + i * 800);
  }
  function check() {
    fetch('/api/achievements/new', { cache: 'no-store' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (j) {
      if (!j || !j.new || !j.new.length) return;
      j.new.slice(0, 3).forEach(toast);
      fetch('/api/achievements/ack', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ids: j.new.map(function (a) { return a.id; }) }) });
    }).catch(function () {});
  }
  if (location.pathname !== '/achievements') setTimeout(check, 2500);
})();
