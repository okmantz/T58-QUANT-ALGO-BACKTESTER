/* Notification inbox: the bell beside the profile picture. Shows saved messages (the Sunday weekly digest) and opens one in place.
   One request on load; silent if it fails. */
(function () {
  'use strict';
  var bell = document.getElementById('t58-bell'), box = document.getElementById('t58-inbox');
  if (!bell || !box) return;
  var dot = document.getElementById('t58-bell-dot'), list = document.getElementById('t58-inbox-list'), view = document.getElementById('t58-inbox-view');
  var msgs = [];
  function csrfHeaders() {
    var h = { 'Content-Type': 'application/json' };
    var r = document.getElementById('t58-profile'); var t = (r && r.getAttribute('data-csrf')) || window.__T58_CSRF;
    if (t) h['X-CSRF-Token'] = t; return h;
  }
  function post(url, body) { return fetch(url, { method: 'POST', headers: csrfHeaders(), body: JSON.stringify(body || {}), credentials: 'same-origin' }).catch(function () {}); }
  function esc(s) { return String(s == null ? '' : s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  function when(t) { try { return new Date(t * 1000).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' }); } catch (e) { return ''; } }
  function badge() {
    var n = msgs.filter(function (m) { return !m.read; }).length;
    dot.hidden = n === 0; dot.textContent = n > 9 ? '9+' : (n > 1 ? String(n) : ''); dot.className = 't58-bell-dot' + (n > 1 ? ' n' : '');
    bell.title = n ? 'Inbox (' + n + ' unread)' : 'Inbox';
  }
  function renderList() {
    view.hidden = true; list.hidden = false;
    if (!msgs.length) { list.innerHTML = '<div class="t58-inbox-empty">Nothing here yet.<br>Your weekly digest arrives every Sunday and is saved here.</div>'; return; }
    list.innerHTML = msgs.map(function (m) {
      return '<button type="button" class="t58-inbox-item' + (m.read ? '' : ' unread') + '" data-id="' + esc(m.id) + '"><b>' + esc(m.title) + '</b><small>Weekly digest &middot; ' + when(m.created) + '</small></button>';
    }).join('');
  }
  function renderMsg(m) {
    var b = m.body || {}, h = '<h4>' + esc(m.title) + '</h4><div class="sub">' + esc(b.subtitle || '') + '</div>';
    if (b.empty || !(b.sections || []).length) h += '<div class="sec"><p>' + esc(b.empty_line || 'Nothing to report.') + '</p></div>';
    (b.sections || []).forEach(function (s) {
      h += '<div class="sec"><h5>' + esc(s.title) + '</h5><p>' + esc(s.headline) + '</p><ul>' + (s.items || []).map(function (i) { return '<li>' + esc(i) + '</li>'; }).join('') + '</ul></div>';
    });
    h += '<div class="row"><button type="button" data-act="back">&larr; Back</button><button type="button" data-act="del">Delete</button></div>';
    view.innerHTML = h; view.dataset.id = m.id; list.hidden = true; view.hidden = false;
    if (!m.read) { m.read = true; badge(); post('/api/inbox/read', { ids: [m.id] }); }
  }
  function load() {
    fetch('/api/inbox', { cache: 'no-store', credentials: 'same-origin' }).then(function (r) { return r.ok ? r.json() : null; }).then(function (j) {
      if (!j) return; msgs = j.messages || []; badge(); if (!box.hidden) renderList();
    }).catch(function () {});
  }
  function open() { box.hidden = false; bell.setAttribute('aria-expanded', 'true'); renderList(); load(); }
  function close() { box.hidden = true; bell.setAttribute('aria-expanded', 'false'); }
  bell.addEventListener('click', function (e) { e.stopPropagation(); box.hidden ? open() : close(); });
  box.addEventListener('click', function (e) {
    e.stopPropagation();
    var it = e.target.closest('.t58-inbox-item');
    if (it) { var m = msgs.filter(function (x) { return x.id === it.dataset.id; })[0]; if (m) renderMsg(m); return; }
    var a = e.target.closest('[data-act]'); if (!a) return;
    if (a.dataset.act === 'back') renderList();
    if (a.dataset.act === 'del') { var id = view.dataset.id; msgs = msgs.filter(function (x) { return x.id !== id; }); badge(); post('/api/inbox/delete', { id: id }); renderList(); }
  });
  document.addEventListener('click', function () { if (!box.hidden) close(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !box.hidden) { close(); bell.focus(); } });
  setTimeout(load, 1500);
})();
