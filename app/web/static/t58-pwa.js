/* T58 install-as-app helper (2026-10-17).
 *
 * - Registers the ROOT-scope service worker (/sw.js) on every page.
 * - Captures Chrome's `beforeinstallprompt` and shows an "Install app" button in the sidebar when the browser allows it.
 * - When the browser does NOT allow it (the usual reason on a phone: the page is on plain http:// -- Chrome only installs
 *   apps from HTTPS or localhost), the same button explains exactly why and how to fix it, instead of leaving you with
 *   Chrome's "This app cannot be installed."
 */
(function () {
  if (window.__t58pwa) return;
  window.__t58pwa = true;

  var deferred = null;
  var standalone = (window.matchMedia && window.matchMedia("(display-mode: standalone)").matches) || window.navigator.standalone === true;
  var secure = window.isSecureContext === true;

  try {
    if ("serviceWorker" in navigator && secure) {
      navigator.serviceWorker.register("/sw.js", { scope: "/" }).catch(function () {});
    }
  } catch (e) {}

  window.addEventListener("beforeinstallprompt", function (e) {
    e.preventDefault();
    deferred = e;
    paint();
  });
  window.addEventListener("appinstalled", function () { deferred = null; paint(); });

  function esc(t) { var d = document.createElement("div"); d.textContent = String(t == null ? "" : t); return d.innerHTML; }

  function reason() {
    if (standalone) return { ok: true, text: "Already running as an installed app." };
    if (deferred) return { ok: true, text: "Ready to install." };
    if (!secure) {
      return {
        ok: false,
        text: "Chrome only installs apps from HTTPS pages. This page is on plain http://" + location.host + ", so Chrome says " +
              "“This app cannot be installed” and only offers a shortcut."
      };
    }
    return { ok: false, text: "Chrome has not offered the install yet. Use the ⋮ menu → Install app (or Add to Home screen), or reload once and try again." };
  }

  function dialog() {
    var r = reason();
    var old = document.getElementById("t58-pwa-dialog");
    if (old) old.remove();
    var wrap = document.createElement("div");
    wrap.id = "t58-pwa-dialog";
    wrap.style.cssText = "position:fixed;inset:0;z-index:10000;background:rgba(0,0,0,.6);display:flex;align-items:center;justify-content:center;padding:16px";
    var steps = "";
    if (!r.ok && !secure) {
      steps =
        '<ol style="padding-left:20px;line-height:1.6;margin:10px 0">' +
        '<li><b>Best (free, 2 minutes):</b> on the PC open <a href="/mobile-access" style="color:var(--teal,#35e0b0)">Mobile Access</a> and press ' +
        '<b>Enable HTTPS (Tailscale)</b>. Then open the <code>https://…ts.net</code> address it shows on your phone and tap Install.</li>' +
        '<li><b>No Tailscale?</b> Run <code>cloudflared tunnel --url http://localhost:' + esc(location.port || "5000") + '</code> on the PC and open the https address it prints.</li>' +
        '<li><b>Quick workaround (this phone only):</b> in Chrome open <code>chrome://flags/#unsafely-treat-insecure-origin-as-secure</code>, ' +
        'add <code>' + esc(location.origin) + '</code>, set it to Enabled, relaunch Chrome, then Install appears.</li>' +
        '<li>Until then, <b>Create shortcut</b> still puts an icon on your home screen (it opens in Chrome).</li></ol>';
    }
    wrap.innerHTML =
      '<div role="dialog" aria-label="Install T58 Backtester" style="max-width:460px;width:100%;background:var(--panel-2,#12161f);color:var(--text,#e7ebf2);' +
      'border:1px solid var(--border,#2a3142);border-radius:14px;padding:18px 20px;font-size:13.5px;line-height:1.5">' +
      '<div style="display:flex;justify-content:space-between;align-items:center"><b style="font-size:15px">Install T58 Backtester</b>' +
      '<button id="t58-pwa-x" aria-label="Close" style="background:none;border:0;color:inherit;font-size:20px;cursor:pointer">×</button></div>' +
      '<p style="margin:10px 0;color:' + (r.ok ? "var(--teal,#35e0b0)" : "var(--text-dim,#9aa4b8)") + '">' + esc(r.text) + '</p>' + steps +
      (deferred ? '<button id="t58-pwa-go" style="margin-top:6px;padding:9px 16px;border-radius:9px;border:1px solid var(--teal,#35e0b0);background:var(--teal,#35e0b0);color:#05070a;font-weight:700;cursor:pointer">Install now</button>' : '') +
      '</div>';
    document.body.appendChild(wrap);
    function close() { wrap.remove(); }
    wrap.addEventListener("click", function (e) { if (e.target === wrap) close(); });
    wrap.querySelector("#t58-pwa-x").addEventListener("click", close);
    var go = wrap.querySelector("#t58-pwa-go");
    if (go) go.addEventListener("click", function () {
      var p = deferred; deferred = null;
      if (!p) return close();
      p.prompt();
      (p.userChoice || Promise.resolve()).then(function () { close(); paint(); });
    });
  }

  function paint() {
    var host = document.getElementById("t58-install-slot");
    if (!host) return;
    if (standalone) { host.innerHTML = ""; return; }
    var ok = !!deferred;
    host.innerHTML = '<button type="button" id="t58-install-btn" class="t58-nav-item top-level" style="width:100%;text-align:left;background:none;border:0;cursor:pointer;font:inherit;color:inherit">' +
      '<span class="t58-nav-icon">⬇</span><span class="label">Install app' + (ok ? "" : " (how?)") + '</span></button>';
    host.querySelector("button").addEventListener("click", function () {
      if (deferred) { dialog(); } else { dialog(); }
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", paint); else paint();
  window.t58Pwa = { reason: reason, open: dialog };
})();
