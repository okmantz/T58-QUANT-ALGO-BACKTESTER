/* Global Events page. Vanilla JS, no dependencies, no network except this app's own /events/api/*.
 * Performance rules: one small bootstrap payload (ETag -> 304 when unchanged), polling only while the tab is visible,
 * the map's land layer is drawn once to an offscreen canvas and re-used, the map redraws only when data / size / hover
 * changes (requestAnimationFrame-coalesced), and every external string goes into the DOM as plain text, never as markup. */
(function () {
  'use strict';
  var API = '/events/api/';
  var POLL_MS = 30000;
  var S = { range: '24h', minImpact: 0, theater: null, data: null, layers: { quake: true, hazard: true, theater: true }, shown: 12, skew: 0, etag: null };
  var $ = function (id) { return document.getElementById(id); };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function safeUrl(u) { return typeof u === 'string' && /^https?:\/\//i.test(u) ? u : null; }
  function now() { return Date.now() + S.skew; }
  function ago(iso) {
    if (!iso) return '';
    var s = Math.max(0, (now() - Date.parse(iso)) / 1000);
    if (s < 90) return 'just now';
    if (s < 5400) return Math.round(s / 60) + 'm ago';
    if (s < 129600) return Math.round(s / 3600) + 'h ago';
    return Math.round(s / 86400) + 'd ago';
  }
  var fmtTime = new Intl.DateTimeFormat(undefined, { weekday: 'short', hour: 'numeric', minute: '2-digit' });
  function fmtNum(v) {
    if (v == null) return '—';
    var a = Math.abs(v);
    return v.toLocaleString(undefined, { maximumFractionDigits: a >= 1000 ? 0 : a >= 100 ? 1 : 2 });
  }

  /* ------------------------------------------------------------------ data */
  function load(force) {
    var q = 'bootstrap?range=' + encodeURIComponent(S.range) + (S.minImpact ? '&min_impact=' + S.minImpact : '') + (S.theater ? '&theater=' + encodeURIComponent(S.theater) : '');
    var headers = {};
    if (S.etag && !force && S.etagKey === q) headers['If-None-Match'] = S.etag;
    return fetch(API + q, { headers: headers, credentials: 'same-origin' }).then(function (r) {
      if (r.status === 304) { tickClock(); return null; }
      if (!r.ok) throw new Error('HTTP ' + r.status);
      S.etag = r.headers.get('ETag'); S.etagKey = q;
      return r.json();
    }).then(function (d) {
      if (!d) return;
      S.data = d;
      S.skew = Date.parse(d.server_time) - Date.now();
      render();
    }).catch(function () {
      $('gx').dataset.state = 'error';
      setFresh('no_data', 'Offline — showing last data');
    });
  }
  var timer = null;
  function schedule() {
    clearTimeout(timer);
    if (document.hidden) return;
    timer = setTimeout(function () { load().then(schedule, schedule); }, POLL_MS);
  }
  document.addEventListener('visibilitychange', function () { if (!document.hidden) { load().then(schedule, schedule); } else clearTimeout(timer); });

  /* ------------------------------------------------------------------ render */
  function setFresh(state, text) {
    var p = $('gx-fresh'); p.dataset.s = state; p.lastElementChild.textContent = text;
  }
  function render() {
    var d = S.data;
    $('gx').dataset.state = 'ready';
    var sum = d.health.summary, age = sum.freshest_age_sec;
    var txt = sum.overall === 'no_data' ? 'Waiting for first data…' : (age == null ? 'No fresh data' : 'Updated ' + (age < 90 ? 'just now' : Math.round(age / 60) + 'm ago'));
    setFresh(sum.overall === 'no_data' ? 'no_data' : sum.overall === 'ok' ? 'ok' : 'degraded', txt);
    renderStrip(d); renderEvents(d); renderTheaters(d); renderHeadlines(d); renderHealth(d); drawMap();
  }

  function sparkline(values, up) {
    var ns = 'http://www.w3.org/2000/svg', svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', '0 0 42 14'); svg.setAttribute('aria-hidden', 'true');
    if (values.length < 2) return svg;
    var min = Math.min.apply(null, values), max = Math.max.apply(null, values), span = max - min || 1;
    var pts = values.map(function (v, i) { return (i / (values.length - 1) * 42).toFixed(1) + ',' + (13 - (v - min) / span * 12).toFixed(1); }).join(' ');
    var pl = document.createElementNS(ns, 'polyline');
    pl.setAttribute('points', pts); pl.setAttribute('fill', 'none'); pl.setAttribute('stroke', up ? '#35e0b0' : '#ff6f6f'); pl.setAttribute('stroke-width', '1.4');
    pl.setAttribute('stroke-linejoin', 'round'); svg.appendChild(pl);
    return svg;
  }
  function renderStrip(d) {
    var box = $('gx-strip'); box.textContent = '';
    d.quotes.forEach(function (q) {
      var up = q.chg_pct != null && q.chg_pct >= 0;
      var c = el('div', 'gx-q ' + (q.price == null ? 'na' : up ? 'up' : 'down'));
      var s = el('div', 's'); s.appendChild(el('span', null, q.symbol)); c.appendChild(s);
      c.title = (d.quote_labels && d.quote_labels[q.symbol] || q.symbol) + (q.ts ? ' · ' + ago(q.ts) + ' · change vs session open' : '');
      c.appendChild(el('div', 'p', q.price == null ? 'n/a' : fmtNum(q.price)));
      var row = el('div', 'c');
      row.appendChild(el('b', null, q.chg_pct == null ? '—' : (q.chg_pct >= 0 ? '+' : '') + q.chg_pct.toFixed(2) + '%'));
      if (q.spark && q.spark.length > 1) row.appendChild(sparkline(q.spark, up));
      c.appendChild(row); box.appendChild(c);
    });
  }

  function cd(ms) {
    var neg = ms < 0, s = Math.abs(Math.floor(ms / 1000)), h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60), sec = s % 60;
    var t = h > 0 ? h + 'h ' + String(m).padStart(2, '0') + 'm' : m > 0 ? m + 'm ' + String(sec).padStart(2, '0') + 's' : sec + 's';
    return neg ? t + ' ago' : 'T-' + t;
  }
  function renderEvents(d) {
    var ul = $('gx-events'); ul.textContent = '';
    var list = d.events.filter(function (e) { return Date.parse(e.when) > now() - 15 * 60000; }).slice(0, 6);
    $('gx-events-sub').textContent = list.length ? 'local time' : '';
    if (!list.length) { ul.appendChild(el('li', 'gx-empty', 'No high or medium impact events in the next 72h in the local store yet.')); return; }
    list.forEach(function (e) {
      var li = el('li', 'gx-ev i' + e.impact); li.dataset.when = e.when;
      li.appendChild(el('span', 'bar'));
      var mid = el('div'); mid.appendChild(el('div', 'n', (e.country ? e.country + ' · ' : '') + e.name));
      var meta = fmtTime.format(new Date(e.when)); if (e.forecast) meta += ' · fcst ' + e.forecast; if (e.previous) meta += ' · prev ' + e.previous; if (e.actual) meta += ' · act ' + e.actual;
      mid.appendChild(el('div', 'm', meta)); li.appendChild(mid);
      li.appendChild(el('div', 't', ''));
      ul.appendChild(li);
    });
    tickClock();
  }
  function tickClock() {
    var items = document.querySelectorAll('#gx-events .gx-ev');
    for (var i = 0; i < items.length; i++) {
      var ms = Date.parse(items[i].dataset.when) - now();
      items[i].lastElementChild.textContent = cd(ms);
      items[i].classList.toggle('past', ms < 0);
      items[i].classList.toggle('soon', ms > 0 && ms < 15 * 60000);
    }
  }
  setInterval(function () { if (!document.hidden) tickClock(); }, 1000);

  function renderTheaters(d) {
    var ul = $('gx-theaters'); ul.textContent = '';
    d.theaters.slice().sort(function (a, b) { return (b.score == null ? -1 : b.score) - (a.score == null ? -1 : a.score); }).forEach(function (t) {
      var li = el('li', 'gx-th' + (t.score == null ? ' na' : t.score >= 70 ? ' hot' : t.score >= 52 ? ' warm' : '') + (S.theater === t.id ? ' sel' : ''));
      li.tabIndex = 0; li.setAttribute('role', 'button'); li.setAttribute('aria-pressed', S.theater === t.id ? 'true' : 'false');
      var n = el('div'); n.appendChild(el('div', 'n', t.name)); n.appendChild(el('div', 'i', t.instruments.join(' · '))); li.appendChild(n);
      var m = el('div', 'meter'); var b = el('i'); b.style.width = (t.score == null ? 0 : t.score) + '%'; m.appendChild(b); li.appendChild(m);
      var v = el('div', 'v', t.score == null ? '—' : String(Math.round(t.score)));
      if (t.components && t.components.calibrating) v.appendChild(el('small', null, 'calibrating'));
      else if (t.delta != null && Math.abs(t.delta) >= 1) { v.appendChild(el('small', t.delta > 0 ? 'up' : 'down', (t.delta > 0 ? '▲' : '▼') + Math.abs(Math.round(t.delta)))); }
      li.appendChild(v);
      if (t.components) li.title = 'vol z ' + t.components.vol_z + ' · impact≥2 (6h) ' + t.components.hi_impact_6h + ' · hazards (24h) ' + t.components.hazard_24h;
      var pick = function () { S.theater = S.theater === t.id ? null : t.id; S.shown = 12; load(true); };
      li.addEventListener('click', pick);
      li.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(); } });
      ul.appendChild(li);
    });
  }

  function renderHeadlines(d) {
    var ul = $('gx-headlines'); ul.textContent = '';
    var note = $('gx-filter-note');
    if (S.theater) {
      var t = d.theaters.filter(function (x) { return x.id === S.theater; })[0];
      note.hidden = false; note.textContent = 'Filtered to ' + (t ? t.name : S.theater) + '.';
      var clr = el('button', null, 'Clear'); clr.type = 'button'; clr.addEventListener('click', function () { S.theater = null; load(true); }); note.appendChild(clr);
    } else note.hidden = true;
    var list = d.headlines;
    if (!list.length) {
      ul.appendChild(el('li', 'gx-empty', d.health.summary.overall === 'no_data' ? 'No headlines yet — the first refresh is running (usually under a minute).' : 'No headlines match these filters in this time range.'));
      $('gx-more').hidden = true; return;
    }
    list.slice(0, S.shown).forEach(function (h) {
      var li = el('li', 'gx-hl i' + h.impact); li.appendChild(el('span', 'bar'));
      var body = el('div'); var url = safeUrl(h.url);
      var a = el('a', 't', h.title); if (url) { a.href = url; a.target = '_blank'; a.rel = 'noopener noreferrer'; } body.appendChild(a);
      var m = el('div', 'm'); m.appendChild(el('span', null, h.publisher)); if (h.tier) m.appendChild(el('span', 'tier', 'T' + h.tier));
      m.appendChild(el('span', null, ago(h.published)));
      if (h.outlets > 1) { var o = el('span', 'out', h.outlets + ' outlets'); if (h.also && h.also.length) o.title = 'Also: ' + h.also.join(', '); m.appendChild(o); }
      if (h.instruments && h.instruments.length) { var ins = el('span', 'gx-inst'); h.instruments.slice(0, 4).forEach(function (x) { ins.appendChild(el('span', null, x)); }); m.appendChild(ins); }
      body.appendChild(m); li.appendChild(body);
      var ask = el('a', 'ask', 'Ask AI'); ask.href = '/assistant/?ask=' + encodeURIComponent('Context on this headline and how it could affect ' + ((h.instruments || []).slice(0, 4).join(', ') || 'my instruments') + ': "' + h.title + '" (' + h.publisher + ')');
      li.appendChild(ask);
      ul.appendChild(li);
    });
    $('gx-more').hidden = list.length <= S.shown;
  }

  function renderHealth(d) {
    var sum = d.health.summary, ul = $('gx-health'); ul.textContent = '';
    var ok = sum.counts.ok || 0;
    $('gx-health-sub').textContent = sum.sources ? ok + ' of ' + sum.sources + ' sources fresh' : 'no sources polled yet';
    d.health.sources.forEach(function (s) {
      var li = el('li', s.status); li.appendChild(el('i'));
      li.appendChild(el('span', null, s.label)); var r = el('span', 'r', s.status === 'ok' ? s.count + ' items · ' + ago(s.last_ok) : s.status.replace('_', ' ') + (s.age_sec != null ? ' · ' + ago(s.last_ok) : ''));
      if (s.error) li.title = s.error; li.appendChild(r); ul.appendChild(li);
    });
  }

  /* ------------------------------------------------------------------ map */
  var LAT_TOP = 80, LAT_BOT = -56, LAND = null, landCanvas = null, landKey = '', hits = [], hover = null, raf = 0;
  var canvas = $('gx-map'), ctx = canvas.getContext('2d');
  fetch('/static/global-events/land.json').then(function (r) { return r.json(); }).then(function (j) { LAND = j; landKey = ''; drawMap(); }).catch(function () {});

  function geom() {
    var w = canvas.clientWidth, h = canvas.clientHeight, sc = Math.min(w / 360, h / (LAT_TOP - LAT_BOT));
    var ox = (w - 360 * sc) / 2, oy = (h - (LAT_TOP - LAT_BOT) * sc) / 2;
    return { w: w, h: h, sc: sc, x: function (lon) { return ox + (lon + 180) * sc; }, y: function (lat) { return oy + (LAT_TOP - lat) * sc; } };
  }
  function sizeCanvas() {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); landKey = ''; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return dpr;
  }
  function buildLand(g, dpr) {
    var key = g.w + 'x' + g.h + '@' + dpr;
    if (landKey === key && landCanvas) return;
    landCanvas = document.createElement('canvas'); landCanvas.width = Math.round(g.w * dpr); landCanvas.height = Math.round(g.h * dpr);
    var c = landCanvas.getContext('2d'); c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.strokeStyle = 'rgba(130,235,220,.06)'; c.lineWidth = 1; c.beginPath();
    for (var lon = -180; lon <= 180; lon += 30) { c.moveTo(g.x(lon), g.y(LAT_TOP)); c.lineTo(g.x(lon), g.y(LAT_BOT)); }
    for (var lat = -30; lat <= 60; lat += 30) { c.moveTo(g.x(-180), g.y(lat)); c.lineTo(g.x(180), g.y(lat)); }
    c.stroke();
    if (LAND) {
      c.beginPath();
      LAND.forEach(function (ring) { for (var i = 0; i < ring.length; i += 2) { var px = g.x(ring[i]), py = g.y(ring[i + 1]); if (i === 0) c.moveTo(px, py); else c.lineTo(px, py); } c.closePath(); });
      c.fillStyle = 'rgba(120,225,210,.075)'; c.fill();
      c.strokeStyle = 'rgba(130,235,220,.30)'; c.lineWidth = .7; c.stroke();
    }
    landKey = key;
  }
  var KIND_COLOR = { quake: '#35e0b0', storm: '#8b7cff', fire: '#f0b429', flood: '#4cc9f0', volcano: '#ff6f6f', other: '#c792ea' };
  function drawMap() { if (!raf) raf = requestAnimationFrame(function () { raf = 0; paintMap(); }); }
  function paintMap() {
    if (!S.data) return;
    var dpr = sizeCanvas(), g = geom(); if (g.w < 10) return;
    buildLand(g, dpr);
    ctx.clearRect(0, 0, g.w, g.h); ctx.drawImage(landCanvas, 0, 0, g.w, g.h);
    hits = [];
    if (S.layers.theater) {
      S.data.theaters.forEach(function (t) {
        if (!t.bbox) return;
        var x0 = g.x(t.bbox[0]), y0 = g.y(t.bbox[3]), w = g.x(t.bbox[2]) - x0, h = g.y(t.bbox[1]) - y0, sc = t.score == null ? 0 : t.score / 100;
        var sel = S.theater === t.id, col = sc >= .7 ? '255,111,111' : sc >= .52 ? '240,180,41' : '53,224,176';
        ctx.beginPath(); ctx.roundRect(x0, y0, w, h, 8);
        ctx.fillStyle = 'rgba(' + col + ',' + (0.03 + sc * 0.14).toFixed(3) + ')'; ctx.fill();
        ctx.strokeStyle = 'rgba(' + col + ',' + (sel ? .9 : .28 + sc * .3).toFixed(2) + ')'; ctx.lineWidth = sel ? 1.6 : 1; ctx.setLineDash(sel ? [] : [4, 4]); ctx.stroke(); ctx.setLineDash([]);
        var lx = g.x(t.label_at ? t.label_at[0] : t.bbox[0]), ly = g.y(t.label_at ? t.label_at[1] : t.bbox[3]);
        ctx.font = '600 10.5px Inter, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = 'rgba(' + col + ',.95)';
        ctx.fillText(t.short + (t.score == null ? '' : ' ' + Math.round(t.score)), lx, ly);
        hits.push({ x: x0 + w / 2, y: y0 + h / 2, r: Math.min(w, h) / 2, kind: 'theater', t: t, box: [x0, y0, w, h] });
      });
    }
    var n = 0;
    S.data.geo.forEach(function (e) {
      if (e.lat == null || e.lon == null) return;
      var isQuake = e.kind === 'quake';
      if (isQuake ? !S.layers.quake : !S.layers.hazard) return;
      var x = g.x(e.lon), y = g.y(e.lat), r = isQuake ? 2 + Math.max(0, (e.mag || 4.5) - 4) * 1.9 : 3.4 + e.sev * .8, col = KIND_COLOR[e.kind] || KIND_COLOR.other;
      ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fillStyle = col + '55'; ctx.fill();
      ctx.lineWidth = 1.1; ctx.strokeStyle = col; ctx.stroke();
      if (e.mkt) { ctx.beginPath(); ctx.arc(x, y, r + 3.5, 0, 6.2832); ctx.strokeStyle = 'rgba(255,255,255,.45)'; ctx.lineWidth = .8; ctx.stroke(); }
      hits.push({ x: x, y: y, r: r + 4, kind: 'geo', e: e }); n++;
    });
    $('gx-map-empty').hidden = n > 0 || !!(S.data.theaters.some(function (t) { return t.score != null; }));
    if (hover && hover.box) { /* theater hover handled via tooltip only */ }
  }
  new ResizeObserver(function () { drawMap(); }).observe($('gx-map-wrap'));

  var tip = $('gx-tip');
  canvas.addEventListener('mousemove', function (ev) {
    var rect = canvas.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top, best = null, bd = 1e9;
    for (var i = 0; i < hits.length; i++) {
      var h = hits[i];
      if (h.kind === 'geo') { var dd = Math.hypot(h.x - mx, h.y - my); if (dd <= h.r && dd < bd) { best = h; bd = dd; } }
    }
    if (!best) for (var j = 0; j < hits.length; j++) { var t = hits[j]; if (t.kind === 'theater' && mx >= t.box[0] && mx <= t.box[0] + t.box[2] && my >= t.box[1] && my <= t.box[1] + t.box[3]) { best = t; break; } }
    hover = best; canvas.classList.toggle('hit', !!best && (best.kind === 'theater' || !!safeUrl(best.e.url)));
    if (!best) { tip.hidden = true; return; }
    tip.textContent = '';
    if (best.kind === 'geo') { tip.appendChild(el('b', null, best.e.title)); tip.appendChild(el('span', null, ago(best.e.t) + (best.e.mkt ? ' · near a market-relevant area' : '') + ' · ' + best.e.src.toUpperCase())); }
    else { tip.appendChild(el('b', null, best.t.name + (best.t.score == null ? '' : ' · heat ' + Math.round(best.t.score)))); tip.appendChild(el('span', null, best.t.top ? best.t.top.title : 'Moves: ' + best.t.instruments.join(', '))); }
    tip.hidden = false;
    var tx = Math.min(mx + 14, canvas.clientWidth - tip.offsetWidth - 8), ty = Math.min(my + 14, canvas.clientHeight - tip.offsetHeight - 8);
    tip.style.left = Math.max(6, tx) + 'px'; tip.style.top = Math.max(6, ty) + 'px';
  });
  canvas.addEventListener('mouseleave', function () { tip.hidden = true; hover = null; });
  canvas.addEventListener('click', function () {
    if (!hover) return;
    if (hover.kind === 'geo') { var u = safeUrl(hover.e.url); if (u) window.open(u, '_blank', 'noopener'); }
    else { S.theater = S.theater === hover.t.id ? null : hover.t.id; S.shown = 12; load(true); }
  });

  /* ------------------------------------------------------------------ controls */
  $('gx-range').addEventListener('click', function (e) {
    var b = e.target.closest('button[data-range]'); if (!b) return;
    S.range = b.dataset.range; S.shown = 12; [].forEach.call(this.children, function (x) { x.classList.toggle('on', x === b); }); load(true);
  });
  $('gx-news-filter').addEventListener('click', function (e) {
    var b = e.target.closest('button[data-min]'); if (!b) return;
    S.minImpact = +b.dataset.min; S.shown = 12; [].forEach.call(this.children, function (x) { x.classList.toggle('on', x === b); }); load(true);
  });
  $('gx-layers').addEventListener('click', function (e) {
    var b = e.target.closest('button[data-layer]'); if (!b) return;
    S.layers[b.dataset.layer] = !S.layers[b.dataset.layer]; b.classList.toggle('on', S.layers[b.dataset.layer]); drawMap();
  });
  $('gx-more').addEventListener('click', function () { S.shown += 12; if (S.data) renderHeadlines(S.data); });
  $('gx-refresh').addEventListener('click', function () {
    var b = this; b.classList.remove('spin'); void b.offsetWidth; b.classList.add('spin');
    fetch(API + 'refresh', { method: 'POST', credentials: 'same-origin' }).catch(function () {});
    setTimeout(function () { load(true); }, 4000); setTimeout(function () { load(true); }, 12000);
  });
  function briefTime(iso) {   // event times arrive in UTC; show them in the reader's own clock
    try { var d = new Date(String(iso).replace(' ', 'T') + (/[zZ]|[+-]\d\d:?\d\d$/.test(iso) ? '' : 'Z')); if (isNaN(d)) return String(iso).slice(11, 16);
      return d.toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); } catch (e) { return String(iso).slice(11, 16); }
  }
  function renderBrief(box, j) {
    box.className = 'gx-brief rich'; box.textContent = '';
    var b = j.brief;
    if (!b) { box.textContent = j.text || ''; return; }
    box.appendChild(el('p', 'gx-b-take', b.takeaway));
    var grid = el('div', 'gx-b-grid');
    b.sections.forEach(function (sec) {
      var card = el('div', 'gx-b-card gx-b-' + sec.type); card.appendChild(el('h3', '', sec.title));
      if (!sec.items.length) { card.appendChild(el('div', 'gx-b-empty', sec.empty)); grid.appendChild(card); return; }
      if (sec.type === 'markets') {
        var chips = el('div', 'gx-b-chips');
        sec.items.forEach(function (m) { var c = el('span', 'gx-b-chip ' + (m.chg_pct > 0 ? 'up' : m.chg_pct < 0 ? 'down' : ''), ''); c.appendChild(el('b', '', m.symbol)); c.appendChild(document.createTextNode((m.chg_pct > 0 ? '▲ ' : m.chg_pct < 0 ? '▼ ' : '') + (m.chg_pct > 0 ? '+' : '') + m.chg_pct.toFixed(2) + '%')); chips.appendChild(c); });
        card.appendChild(chips);
      } else {
        var ul = el('ul', 'gx-b-list');
        sec.items.forEach(function (it) {
          var li = el('li', '');
          if (sec.type === 'events') { li.appendChild(el('span', 'tm', briefTime(it.when))); li.appendChild(el('span', 'tx', (it.country ? it.country + ' · ' : '') + it.name)); }
          else if (sec.type === 'theaters') { li.appendChild(el('span', 'tx', it.name)); var bar = el('span', 'bar'); var i = el('i', ''); i.style.width = Math.max(4, Math.min(100, it.score)) + '%'; bar.appendChild(i); li.appendChild(bar); li.appendChild(el('span', 'sc', String(it.score))); }
          else { li.appendChild(el('span', 'tx', it.title)); if (it.publisher) li.appendChild(el('span', 'pub', it.publisher)); }
          ul.appendChild(li);
        });
        card.appendChild(ul);
        if (sec.note) card.appendChild(el('div', 'gx-b-empty', sec.note));
      }
      grid.appendChild(card);
    });
    box.appendChild(grid);
    if (j.ai) {
      var ai = el('div', 'gx-b-ai'); ai.appendChild(el('h3', '', 'Analyst take (AI wording of the figures above)'));
      String(j.ai).split(/\n+/).forEach(function (line) { line = line.trim(); if (!line) return;
        var bullet = /^[-*•]\s+/.test(line); var node = el(bullet ? 'div' : 'p', bullet ? 'gx-b-bullet' : '', line.replace(/^[-*•]\s+/, '').replace(/\*\*/g, '')); ai.appendChild(node); });
      box.appendChild(ai);
    }
    var foot = (b.health && b.health.overall ? 'Data health: ' + b.health.overall + ' (freshest source ' + b.health.age + '). ' : '') + (j.error ? j.error + ' ' : '') + 'Every figure comes from the local store.';
    box.appendChild(el('div', 'note', foot));
  }
  $('gx-brief-btn').addEventListener('click', function () {
    var btn = this, box = $('gx-brief'); btn.disabled = true; box.className = 'gx-brief muted'; box.textContent = 'Generating…';
    fetch(API + 'brief', { method: 'POST', credentials: 'same-origin' }).then(function (r) { return r.json(); }).then(function (j) { renderBrief(box, j); })
      .catch(function () { box.className = 'gx-brief muted'; box.textContent = 'Could not generate a brief right now.'; }).then(function () { btn.disabled = false; });
  });

  /* ------------------------------------------------------------------ X posts (own endpoint, 5-min server cache) */
  var X = { loaded: false };
  function xNote(msg, isErr) {
    var n = $('gx-x-note'); if (!msg) { n.hidden = true; n.textContent = ''; return; }
    n.hidden = false; n.textContent = msg; n.className = 'gx-x-note' + (isErr ? ' err' : '');
  }
  function renderX(d) {
    var chips = $('gx-x-chips'), list = $('gx-x-posts');
    chips.textContent = ''; list.textContent = '';
    (d.accounts || []).forEach(function (u) {
      var c = el('span', 'gx-x-chip' + (d.errors && d.errors[u] ? ' bad' : ''));
      c.appendChild(el('span', null, '@' + u));
      var rm = el('button', null, '×'); rm.type = 'button'; rm.title = 'Remove @' + u; rm.setAttribute('aria-label', 'Remove @' + u);
      rm.addEventListener('click', function () {
        fetch(API + 'x/accounts/remove', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: u }) })
          .then(function () { loadX(true); });
      });
      c.appendChild(rm); chips.appendChild(c);
    });
    if (!d.configured) {
      xNote('Add your X bearer token on the API Keys page to show posts here.', false);
      var a = el('a', null, 'Open API Keys'); a.href = '/settings/api-keys'; a.style.marginLeft = '6px'; $('gx-x-note').appendChild(a);
    } else if (!(d.accounts || []).length) {
      xNote('Add one or more X usernames above to follow them here.', false);
    } else {
      var errs = Object.keys(d.errors || {}).map(function (u) { return '@' + u + ': ' + d.errors[u]; });
      xNote(errs.length ? errs.join('  ·  ') : '', true);
    }
    (d.posts || []).forEach(function (p) {
      var li = el('li', 'gx-x-post');
      var head = el('div', 'gx-x-head');
      head.appendChild(el('strong', null, p.name || p.username));
      head.appendChild(el('span', 'gx-x-handle', '@' + p.username));
      head.appendChild(el('span', 'gx-x-when', ago(p.created_at)));
      li.appendChild(head);
      li.appendChild(el('p', 'gx-x-text', p.text));
      var url = safeUrl(p.url);
      if (url) { var a2 = el('a', 'gx-x-link', 'View on X ↗'); a2.href = url; a2.target = '_blank'; a2.rel = 'noopener noreferrer'; li.appendChild(a2); }
      list.appendChild(li);
    });
  }
  function loadX(force) {
    return fetch(API + 'x' + (force ? '?refresh=1' : ''), { credentials: 'same-origin' })
      .then(function (r) { return r.json(); }).then(renderX)
      .catch(function () { xNote('Could not load X posts right now.', true); });
  }
  $('gx-x-form').addEventListener('submit', function (e) {
    e.preventDefault();
    var inp = $('gx-x-input'), v = inp.value.trim(); if (!v) return;
    fetch(API + 'x/accounts', { method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username: v }) })
      .then(function (r) { return r.json(); }).then(function (j) {
        if (j.ok) { inp.value = ''; loadX(true); } else { xNote(j.message || 'Could not add that account.', true); }
      });
  });
  loadX(false); setInterval(function () { if (!document.hidden) loadX(false); }, 300000);

  load().then(schedule, schedule);
})();
