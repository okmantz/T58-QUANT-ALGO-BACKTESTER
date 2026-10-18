/* Floating "running jobs" pill (2026-10-17).
 *
 * Shown on EVERY page -- including the job/progress pages that have no sidebar and the phone layout where the sidebar is a
 * hidden drawer -- whenever something is running. Tap it to see every running job and jump straight back in; "All runs"
 * opens the Runs page. This is the "I pressed Back while it was running and it vanished" safety net: a run never leaves the
 * server just because you left its page, and now it never leaves your screen either.
 */
(function () {
  if (window.__t58runs) return;
  window.__t58runs = true;
  if (/^\/(runs|activate|lock)(\/|$)/.test(location.pathname)) return;

  var delay = 5000, timer = null, busy = false, open = false, last = { running: [], recent: [] };
  var pill, panel;

  function el(tag, css, text) { var e = document.createElement(tag); if (css) e.style.cssText = css; if (text != null) e.textContent = text; return e; }
  function fmt(sec) {
    if (sec == null) return "";
    sec = Math.floor(sec);
    if (sec < 60) return sec + "s";
    if (sec < 3600) return Math.floor(sec / 60) + "m " + (sec % 60) + "s";
    return Math.floor(sec / 3600) + "h " + Math.floor((sec % 3600) / 60) + "m";
  }
  function here(j) { try { return j.page_url && (location.pathname + location.search).indexOf(j.page_url.split("?")[0]) === 0; } catch (e) { return false; } }

  function build() {
    if (pill) return;
    pill = el("button", "position:fixed;right:14px;bottom:calc(14px + env(safe-area-inset-bottom,0px));z-index:9000;display:none;align-items:center;gap:8px;" +
      "padding:10px 14px;border-radius:999px;border:1px solid var(--teal,#35e0b0);background:var(--panel-2,#12161f);color:var(--text,#e7ebf2);" +
      "font:600 13px system-ui,sans-serif;box-shadow:0 8px 24px rgba(0,0,0,.45);cursor:pointer");
    pill.type = "button";
    pill.id = "t58-runs-pill";
    pill.setAttribute("aria-label", "Running jobs");
    pill.addEventListener("click", function () { open = !open; render(); });
    document.body.appendChild(pill);
    panel = el("div", "position:fixed;right:14px;bottom:calc(66px + env(safe-area-inset-bottom,0px));z-index:9001;display:none;width:min(360px,calc(100vw - 28px));" +
      "max-height:60vh;overflow:auto;padding:10px;border-radius:14px;border:1px solid var(--border-light,#2a3242);background:var(--panel-2,#12161f);" +
      "color:var(--text,#e7ebf2);font:13px system-ui,sans-serif;box-shadow:0 12px 32px rgba(0,0,0,.5)");
    panel.id = "t58-runs-panel";
    document.body.appendChild(panel);
    document.addEventListener("click", function (e) { if (open && !panel.contains(e.target) && !pill.contains(e.target)) { open = false; render(); } });
  }

  function row(j, finished) {
    var a = el("a", "display:flex;justify-content:space-between;gap:10px;padding:9px 10px;border-radius:9px;text-decoration:none;color:inherit;" +
      (here(j) ? "background:rgba(53,224,176,.12);" : ""));
    a.href = j.page_url;
    var left = el("div", "min-width:0");
    left.appendChild(el("div", "font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis", j.tool + (here(j) ? "  (you are here)" : "")));
    if (j.instrument) left.appendChild(el("div", "color:var(--text-muted,#8a93a6);font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis", j.instrument));
    var right = el("div", "white-space:nowrap;font-size:12px;text-align:right;color:" + (finished ? (j.error ? "var(--coral,#ff6f6f)" : "var(--text-muted,#8a93a6)") : "var(--teal,#35e0b0)"));
    right.textContent = finished ? (j.error ? "failed" : "done") + " · " + fmt(j.elapsed_seconds) : "● " + fmt(j.elapsed_seconds) +
      (j.progress != null ? " · " + Math.round(j.progress) + "%" : "");
    a.appendChild(left); a.appendChild(right);
    return a;
  }

  function render() {
    build();
    var n = last.running.length;
    if (!n && !open) { pill.style.display = "none"; panel.style.display = "none"; return; }
    pill.style.display = "inline-flex";
    pill.textContent = "";
    var dot = el("span", "width:9px;height:9px;border-radius:50%;background:var(--teal,#35e0b0);display:inline-block");
    pill.appendChild(dot);
    pill.appendChild(document.createTextNode(n ? n + " running" : "Runs"));
    panel.style.display = open ? "block" : "none";
    if (!open) return;
    panel.textContent = "";
    if (n) { panel.appendChild(el("div", "font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted,#8a93a6);padding:2px 10px 4px", "Running now")); last.running.forEach(function (j) { panel.appendChild(row(j, false)); }); }
    if (last.recent.length) { panel.appendChild(el("div", "font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--text-muted,#8a93a6);padding:8px 10px 4px", "Recently finished")); last.recent.slice(0, 5).forEach(function (j) { panel.appendChild(row(j, true)); }); }
    var all = el("a", "display:block;text-align:center;margin-top:8px;padding:8px;border-top:1px solid var(--border,#2a3142);color:var(--teal,#35e0b0);text-decoration:none;font-weight:600", "All runs →");
    all.href = "/runs";
    panel.appendChild(all);
  }

  function schedule() { busy = false; timer = setTimeout(poll, delay); }
  function poll() {
    busy = true;
    fetch("/api/running-jobs.json", { credentials: "same-origin" }).then(function (r) { return r.json(); }).then(function (d) {
      last = { running: (d && d.running) || [], recent: (d && d.recent) || [] };
      delay = last.running.length ? 4000 : Math.min(delay * 2, 30000);
      if (last.running.length) delay = 4000;
      render();
    }).catch(function () { delay = Math.min(delay * 2, 30000); }).then(schedule);
  }
  function kick() { clearTimeout(timer); delay = 4000; if (!busy) poll(); }

  document.addEventListener("visibilitychange", function () { if (!document.hidden) kick(); });
  // Back/forward restores a frozen copy of the page from the browser's cache; refresh immediately so it is never stale.
  window.addEventListener("pageshow", function (e) { if (e.persisted) kick(); });
  window.t58Runs = { refresh: kick };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", poll); else poll();
})();
