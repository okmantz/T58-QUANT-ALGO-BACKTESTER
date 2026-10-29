/* T58 preference sync. Loaded first in every page's <head> (injected by app/web/server.py together with window.__T58_PREFS,
 * the server's saved copy). Every `t58-*` localStorage key -- background, theme, accent colour, verse bar, playground scores --
 * is mirrored to the server (data/ui_prefs.json), which the account vault carries across a reinstall.
 *
 *  1. HYDRATE (before first paint): server values are written into localStorage, so a fresh browser profile, a new port or a
 *     reinstalled app comes up with the same look it had.
 *  2. MIGRATE: keys this browser has that the server doesn't yet are pushed up once.
 *  3. MIRROR: setItem / removeItem on a t58-* key is batched (400 ms) and POSTed to /api/prefs.
 * Everything is best effort and silent: if storage or the network is unavailable the page behaves exactly as before. */
(function () {
  'use strict';
  if (window.__t58PrefsSync) return;
  window.__t58PrefsSync = true;
  var PREFIX = 't58-';
  var server = window.__T58_PREFS || {};
  var ls;
  try { ls = window.localStorage; ls.length; } catch (e) { return; }
  var nativeSet = Storage.prototype.setItem, nativeRemove = Storage.prototype.removeItem;
  function mine(k) { return typeof k === 'string' && k.indexOf(PREFIX) === 0; }

  var pending = {}, timer = 0, dirty = false;
  function flush() {
    timer = 0;
    if (!dirty) return;
    var body = JSON.stringify({ changes: pending });
    pending = {}; dirty = false;
    var headers = { 'Content-Type': 'application/json' };
    if (window.__T58_CSRF) headers['X-CSRF-Token'] = window.__T58_CSRF;
    try { fetch('/api/prefs', { method: 'POST', headers: headers, body: body, keepalive: true, credentials: 'same-origin' }).catch(function () {}); } catch (e) {}
  }
  function queue(k, v) { pending[k] = v; dirty = true; if (!timer) timer = setTimeout(flush, 400); }

  // 0. owner fresh start: the server ran its one-time clean slate (new __T58_FRESH_EPOCH), so this browser forgets its own
  //    saved look once too. The marker key deliberately does not start with `t58-` so it is never mirrored or wiped.
  try {
    var epoch = window.__T58_FRESH_EPOCH || '';
    if (epoch && ls.getItem('x58-fresh-epoch') !== epoch) {
      var dead = [];
      for (var j = 0; j < ls.length; j++) { var kk = ls.key(j); if (mine(kk)) dead.push(kk); }
      dead.forEach(function (kk) { nativeRemove.call(ls, kk); });
      nativeSet.call(ls, 'x58-fresh-epoch', epoch);
    }
  } catch (e) {}

  // 1. hydrate: the server's copy wins
  Object.keys(server).forEach(function (k) {
    if (!mine(k)) return;
    try { if (ls.getItem(k) !== server[k]) nativeSet.call(ls, k, server[k]); } catch (e) {}
  });
  // 2. migrate keys only this browser has
  try {
    for (var i = 0; i < ls.length; i++) {
      var k = ls.key(i);
      if (mine(k) && !(k in server)) { var v = ls.getItem(k); if (v !== null && v.length <= 200000) queue(k, v); }
    }
  } catch (e) {}

  // 3. mirror future writes
  Storage.prototype.setItem = function (k, v) {
    nativeSet.apply(this, arguments);
    if (this === ls && mine(k)) queue(k, String(v));
  };
  Storage.prototype.removeItem = function (k) {
    nativeRemove.apply(this, arguments);
    if (this === ls && mine(k)) queue(k, null);
  };
  window.addEventListener('pagehide', flush);
  document.addEventListener('visibilitychange', function () { if (document.hidden) flush(); });
})();
