// T58 service worker (v4). Served from /sw.js so its scope is the WHOLE app ("/"). It used to live only at
// /static/sw.js, whose scope is /static/ -- it never controlled "/" or "/dashboard", which is one of the things Chrome
// checks before it offers "Install app". It caches the app shell + static assets and falls back to a small offline page
// for navigations. It never caches /run, /reports/ or any POST/API response (those need the live backend).
const CACHE_NAME = "t58-backtester-shell-v4";   // bump to drop every older cache (v3 served stale CSS next to fresh HTML)
const SHELL_ASSETS = [
  "/static/offline.html", "/manifest.json",
  "/static/icon-192.png", "/static/icon-512.png",
  "/static/theme.css", "/static/t58-chrome.js",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => Promise.all(SHELL_ASSETS.map((u) => cache.add(u).catch(() => {}))))
      .catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;
  const url = new URL(req.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/run") || url.pathname.startsWith("/reports/") || url.pathname.startsWith("/api/")) return;

  // Pages: always network first (they are live), offline page when the PC is unreachable.
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req).catch(() => caches.match("/static/offline.html").then((r) => r || new Response("Offline", { status: 503 })))
    );
    return;
  }
  // Code (CSS / JS / JSON / HTML) is NETWORK FIRST: the page's markup always comes fresh from the server, so the stylesheet and
  // scripts it was written for must be fresh too. (v3 served cached CSS first, so after an update the new page met the OLD
  // stylesheet -- giant sidebar icons, a squashed lock card.) The cache is only the offline fallback.
  if (/\.(css|js|json|html)$/.test(url.pathname) || url.pathname === "/manifest.json") {
    event.respondWith(
      fetch(req).then((res) => {
        if (res && res.status === 200) { const copy = res.clone(); caches.open(CACHE_NAME).then((c) => c.put(req, copy)); }
        return res;
      }).catch(() => caches.match(req).then((r) => r || new Response("", { status: 503 })))
    );
    return;
  }
  // Images / fonts under /static: cache first, refresh in the background (new artwork gets a new file name).
  if (url.pathname.startsWith("/static/")) {
    event.respondWith(
      caches.match(req).then((cached) => {
        const refresh = fetch(req).then((res) => {
          if (res && res.status === 200) { const copy = res.clone(); caches.open(CACHE_NAME).then((c) => c.put(req, copy)); }
          return res;
        }).catch(() => cached);
        return cached || refresh;
      })
    );
  }
});
