/* T58 polling governor (2026-10-15 audit). Loaded into every HTML page (injected by app/web/server.py).
 *
 * The sidebar polled every 5 s, job pages every 2-3 s and the live market every 1.5 s -- from every open tab,
 * visible or not, forever. This shim fixes that for all of them without touching each page:
 *
 *  1. HIDDEN TAB = NO POLLING. setInterval ticks (>= 800 ms) and polling-style setTimeout callbacks
 *     (1-60 s) are skipped/deferred while document.hidden, then run ONCE as soon as the tab is visible again,
 *     so a returning user sees fresh data immediately.
 *  2. BACKOFF. A page's recursive `setTimeout(poll, ms)` loop slows down the longer it keeps polling:
 *     full rate for its first ~30 polls, 2x slower up to ~120, 4x slower after that (capped at 15 s). Any
 *     click/keypress or returning to the tab resets it to full rate. Only functions literally named `poll`
 *     are scaled, so ordinary UI timers are untouched.
 *
 * Opt out per page with window.T58_POLL_GOVERNOR = false before this script runs.
 */
(function () {
  if (window.T58_POLL_GOVERNOR === false || window.__t58PollGovernor) return;
  window.__t58PollGovernor = true;

  var nativeSetInterval = window.setInterval.bind(window);
  var nativeClearInterval = window.clearInterval.bind(window);
  var nativeSetTimeout = window.setTimeout.bind(window);
  var nativeClearTimeout = window.clearTimeout.bind(window);

  var pendingIntervals = new Map();   // interval id -> callback that was skipped while hidden
  var deferredTimeouts = new Map();   // timeout id  -> {fn,args} that fired while hidden
  var pollCounts = new WeakMap();     // poll function -> times rescheduled
  var intervalFns = new Map();        // interval id -> callback

  function resetBackoff() { pollCounts = new WeakMap(); }

  window.setInterval = function (fn, delay) {
    var d = Number(delay) || 0;
    if (typeof fn !== 'function' || d < 800) return nativeSetInterval.apply(null, arguments);
    var extra = Array.prototype.slice.call(arguments, 2);
    var id = nativeSetInterval(function () {
      if (document.hidden) { pendingIntervals.set(id, true); return; }
      fn.apply(null, extra);
    }, d);
    intervalFns.set(id, function () { fn.apply(null, extra); });
    return id;
  };
  window.clearInterval = function (id) {
    pendingIntervals.delete(id); intervalFns.delete(id);
    return nativeClearInterval(id);
  };

  window.setTimeout = function (fn, delay) {
    var d = Number(delay) || 0;
    if (typeof fn !== 'function' || d < 1000 || d > 60000) return nativeSetTimeout.apply(null, arguments);
    var extra = Array.prototype.slice.call(arguments, 2);
    if (fn.name === 'poll' && d <= 15000) {
      var n = (pollCounts.get(fn) || 0) + 1;
      pollCounts.set(fn, n);
      var factor = n > 120 ? 4 : (n > 30 ? 2 : 1);
      d = Math.max(d, Math.min(d * factor, 15000));
    }
    var id = nativeSetTimeout(function () {
      if (document.hidden) { deferredTimeouts.set(id, { fn: fn, args: extra }); return; }
      fn.apply(null, extra);
    }, d);
    return id;
  };
  window.clearTimeout = function (id) {
    deferredTimeouts.delete(id);
    return nativeClearTimeout(id);
  };

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) return;
    resetBackoff();
    var t = Array.from(deferredTimeouts.values()); deferredTimeouts.clear();
    t.forEach(function (e) { try { e.fn.apply(null, e.args); } catch (err) { console.error(err); } });
    var i = Array.from(pendingIntervals.keys()); pendingIntervals.clear();
    i.forEach(function (id) {
      var f = intervalFns.get(id);
      if (f) { try { f(); } catch (err) { console.error(err); } }
    });
  });
  ['click', 'keydown', 'pointerdown'].forEach(function (ev) {
    document.addEventListener(ev, resetBackoff, { passive: true, capture: true });
  });
})();
