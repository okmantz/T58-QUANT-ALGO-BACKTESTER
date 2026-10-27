/* "What next?" card for the strategy-lifecycle pages (Full Pipeline, Validate, Champion, Forward test, Deploy).
   The advice comes from /api/next-step (app/guidance/next_step.py) -- the same resolver the Dashboard cards use -- so it always
   quotes THIS strategy's own scores. Renders once the page has a strategy to talk about; refreshes while a job page is open. */
(function () {
  var path = location.pathname;
  if (!/^\/(full-pipeline|validate-simple|validate|champion-simple|champion|forward-test|deploy-live|optimize-simple)(\/|$)/.test(path)) return;
  if (/^\/optimize-simple\/?$/.test(path)) return;   // the Optimize form has its own suggestion line
  var isJob = /\/job\//.test(path);
  var host;
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]; }); }
  function css() {
    if (document.getElementById('t58-ns-css')) return;
    var st = document.createElement('style'); st.id = 't58-ns-css';
    st.textContent = '.t58-ns{margin:0 0 16px;padding:14px 16px;border-radius:14px;border:1px solid var(--border);background:var(--panel-2,rgba(255,255,255,.04));display:grid;gap:8px;max-width:1240px;--tone:var(--teal-rgb);border-left:3px solid rgb(var(--tone))}' +
      '.t58-ns.bad{--tone:255,107,107}.t58-ns.warn{--tone:var(--amber-rgb,245,184,61)}' +
      '.t58-ns .k{font-size:10.5px;text-transform:uppercase;letter-spacing:.09em;color:var(--text-dim,#8a97a0);font-weight:650}' +
      '.t58-ns h3{margin:0;font-size:15px;font-weight:650}.t58-ns p{margin:0;font-size:13px;color:var(--text-muted)}' +
      '.t58-ns ol{margin:2px 0 0;padding:0;list-style:none;display:grid;gap:3px;font-size:12.5px;color:var(--text)}' +
      '.t58-ns .a{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}.t58-ns .a a{font-size:13px;font-weight:650;padding:8px 14px;border-radius:10px;text-decoration:none;border:1px solid var(--border-light,var(--border));color:var(--text)}' +
      '.t58-ns .a a.p{background:var(--teal);border-color:var(--teal);color:var(--on-teal)}';
    document.head.appendChild(st);
  }
  function render(d) {
    if (!d || d.none) { if (host) host.style.display = 'none'; return; }
    css();
    if (!host) {
      host = document.createElement('section'); host.className = 't58-ns'; host.setAttribute('aria-live', 'polite');
      var main = document.querySelector('.t58-main'); if (!main) return;
      var wrap = main.querySelector('.lc-wrap');
      if (wrap) wrap.insertBefore(host, wrap.firstChild); else main.insertBefore(host, main.firstChild);
    }
    host.style.display = '';
    host.className = 't58-ns ' + (d.tone || 'info');
    host.innerHTML = '<span class="k">Next step &middot; ' + esc(d.name || '') + (d.stage_title ? ' &middot; ' + esc(d.stage_title) : '') + '</span>' +
      '<h3>' + esc(d.headline) + '</h3><p>' + esc(d.why) + '</p>' +
      (d.steps && d.steps.length ? '<ol>' + d.steps.map(function (s) { return '<li>' + esc(s) + '</li>'; }).join('') + '</ol>' : '') +
      '<div class="a"><a class="p" href="' + esc(d.primary.href) + '">' + esc(d.primary.label) + ' &rarr;</a>' +
      (d.secondary || []).map(function (x) { return '<a href="' + esc(x.href) + '">' + esc(x.label) + '</a>'; }).join('') + '</div>';
  }
  function params(cb) {
    var q = new URLSearchParams(location.search), pick = q.get('strategy_pick') || '';
    if (pick) return cb({pick: pick});
    var m = path.match(/\/job\/([A-Za-z0-9_-]+)/);
    if (m) {
      fetch('/job-strategy/' + m[1]).then(function (r) { return r.json(); }).then(function (j) { cb(j && j.strategy ? {name: j.strategy} : {}); }).catch(function () { cb({}); });
      return;
    }
    cb({});
  }
  function load() {
    params(function (p) {
      fetch('/api/next-step?' + new URLSearchParams(p)).then(function (r) { return r.json(); }).then(render).catch(function () {});
    });
  }
  function boot() { load(); if (isJob) setInterval(function () { if (!document.hidden) load(); }, 12000); }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot); else boot();
})();
