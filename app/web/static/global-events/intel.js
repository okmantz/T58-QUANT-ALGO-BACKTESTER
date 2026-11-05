/* Global Events: market intelligence panels. Reads /events/api/intel (computed locally); builds DOM with textContent only. */
(function () {
  'use strict';
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function f(v, d) { return v == null || v !== v ? '—' : Number(v).toFixed(d == null ? 2 : d); }
  function pct(v) { return v == null || v !== v ? '—' : (v > 0 ? '+' : '') + Number(v).toFixed(2) + '%'; }
  function sign(v) { return v > 0 ? 'up' : (v < 0 ? 'dn' : ''); }
  function empty(box, msg) { box.textContent = ''; box.appendChild(el('p', 'gxi-empty', msg)); }
  var NO_DATA = 'Daily history is still loading (first fetch takes a minute after opening this page).';

  function table(head, rows) {
    var t = el('table', 'gxi-tbl'), th = el('thead'), tr = el('tr');
    head.forEach(function (h) { tr.appendChild(el('th', null, h)); }); th.appendChild(tr); t.appendChild(th);
    var tb = el('tbody');
    rows.forEach(function (r) { var x = el('tr'); r.forEach(function (c) { var td = el('td'); if (c && typeof c === 'object') { td.textContent = c.t; if (c.c) td.className = c.c; } else td.textContent = c; x.appendChild(td); }); tb.appendChild(x); });
    t.appendChild(tb); return t;
  }
  function when(iso) { try { return new Date(iso).toLocaleString(undefined, { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' }); } catch (e) { return iso || ''; } }

  function regimes(d) {
    var box = $('gxi-regimes'); box.textContent = '';
    var avail = d.regimes.filter(function (r) { return r.regime !== 'Unknown'; });
    if (!avail.length) { empty(box, NO_DATA); }
    d.regimes.forEach(function (r) {
      var c = el('div', 'gxi-reg gxi-' + r.regime.toLowerCase().split(' ')[0].replace(/[^a-z]/g, ''));
      c.appendChild(el('div', 'gxi-sym', r.symbol)); c.appendChild(el('div', 'gxi-lab', r.regime)); c.appendChild(el('div', 'gxi-why', r.why));
      var m = r.metrics || {};
      if (m.efficiency_ratio_20d != null) c.appendChild(el('div', 'gxi-met', 'ER ' + f(m.efficiency_ratio_20d) + ' · vol pct ' + f(m.vol_percentile_1y, 0) + ' · 20d ' + pct(m.return_20d_pct)));
      box.appendChild(c);
    });
    var ul = $('gxi-rule-list'); ul.textContent = ''; d.rules.text.forEach(function (t) { ul.appendChild(el('li', null, t)); });
  }
  function macro(d) {
    var box = $('gxi-macro-body'); box.textContent = '';
    var now = Date.now(), list = d.macro.filter(function (r) { return Date.parse(r.when) > now - 36 * 3600e3; }).slice(0, 14);
    if (!list.length) return empty(box, 'No CPI / PPI / jobs / GDP / rate events in the loaded calendar window.');
    box.appendChild(table(['When', 'Release', 'Group', 'Fcst', 'Prev', 'Actual'], list.map(function (r) {
      return [when(r.when), { t: (r.ccy ? r.ccy + ' ' : '') + r.name, c: r.impact >= 3 ? 'hi' : '' }, r.group, r.forecast || '—', r.previous || '—', r.actual || '—'];
    })));
    box.appendChild(el('p', 'gxi-empty', 'Calendar feed covers the current week; the Next events list above shows the nearest high-impact items.'));
  }
  function cb(d) {
    var box = $('gxi-cb-body'); box.textContent = '';
    box.appendChild(table(['Bank', 'Next decision', 'In', 'Moves'], d.central_banks.map(function (b) {
      return [{ t: b.name + (b.forecast ? ' (fcst ' + b.forecast + ', prev ' + (b.previous || '—') + ')' : '') }, b.next || 'list ended', b.days == null ? '—' : (b.days < 1 ? 'today' : b.days.toFixed(0) + 'd'), (b.moves || []).join(' ')];
    })));
    box.appendChild(el('p', 'gxi-empty', d.central_banks.length ? 'Dates are each bank’s published schedule: confirm on the bank’s own page before trading around them.' : ''));
  }
  function sessions(d) {
    var box = $('gxi-sessions-body'); box.textContent = '';
    d.sessions.forEach(function (s) {
      var r = el('div', 'gxi-sess'); r.appendChild(el('i', s.open ? 'on' : 'off'));
      var b = el('div'); b.appendChild(el('b', null, s.name + ' · ' + (s.open ? 'open' : 'closed'))); b.appendChild(el('span', 'gxi-met', s.local_time + ' local · ' + s.until + ' · ' + s.hours)); r.appendChild(b); box.appendChild(r);
    });
    if (!d.sessions.length) empty(box, 'Time-zone data is not available on this install.');
  }
  function vol(d) {
    var box = $('gxi-vol-body'), v = d.vol; box.textContent = '';
    if (!v.available) return empty(box, NO_DATA);
    box.appendChild(el('div', 'gxi-big', 'VIX ' + f(v.vix, 1) + ' — ' + v.bucket));
    [['1-year percentile', f(v.vix_pct_1y, 0) + 'th'], ['5-day change', pct(v.vix_chg_5d)], ['ES realised vol (20d)', v.es_rv20 == null ? '—' : f(v.es_rv20, 1) + '%'], ['ES vol percentile', v.es_rv_pct_1y == null ? '—' : f(v.es_rv_pct_1y, 0) + 'th']].forEach(function (p) {
      var r = el('div', 'gxi-kv'); r.appendChild(el('span', null, p[0])); r.appendChild(el('b', null, p[1])); box.appendChild(r);
    });
    if (v.premium_note) box.appendChild(el('p', 'gxi-empty', v.premium_note));
  }
  function futures(d) {
    var box = $('gxi-futures-body'); box.textContent = '';
    var rows = d.futures.filter(function (x) { return x.available; });
    if (!rows.length) return empty(box, NO_DATA);
    box.appendChild(table(['', 'Last', '1d', '5d', '20d', '20d range pos', 'RV20', 'ATR14'], rows.map(function (x) {
      return [x.symbol, f(x.last), { t: pct(x.chg_1d), c: sign(x.chg_1d) }, { t: pct(x.chg_5d), c: sign(x.chg_5d) }, { t: pct(x.chg_20d), c: sign(x.chg_20d) }, f(x.range_pos_20d, 0) + '%', f(x.rv20, 1) + '%', x.atr14_pct == null ? '—' : f(x.atr14_pct) + '%'];
    })));
  }
  function cross(d) {
    var box = $('gxi-cross-body'), c = d.cross_asset; box.textContent = '';
    var rows = c.rows.filter(function (x) { return x.available; });
    if (!rows.length) return empty(box, NO_DATA);
    if (c.tone) box.appendChild(el('div', 'gxi-big', 'Tone: ' + c.tone));
    box.appendChild(table(['', '1d', '5d', '20d', 'Z(1d)'], rows.map(function (x) {
      return [{ t: x.symbol + (x.unusual ? ' ⚠' : '') }, { t: pct(x.chg_1d), c: sign(x.chg_1d) }, { t: pct(x.chg_5d), c: sign(x.chg_5d) }, { t: pct(x.chg_20d), c: sign(x.chg_20d) }, f(x.z_1d, 1)];
    })));
    box.appendChild(el('p', 'gxi-empty', c.tone_rule + ' ' + c.unusual_rule));
  }
  function corr(d) {
    var box = $('gxi-corr-body'), c = d.correlations; box.textContent = '';
    if (!c.symbols.length) return empty(box, NO_DATA);
    var t = el('table', 'gxi-tbl gxi-corr'), tr = el('tr'); tr.appendChild(el('th'));
    c.symbols.forEach(function (s) { tr.appendChild(el('th', null, s)); }); t.appendChild(tr);
    c.matrix.forEach(function (row, i) {
      var r = el('tr'); r.appendChild(el('th', null, c.symbols[i]));
      row.forEach(function (v) {
        var td = el('td', null, v == null ? '' : v.toFixed(2));
        if (v != null) { var a = Math.min(1, Math.abs(v)); td.style.background = v >= 0 ? 'rgba(53,224,176,' + (a * .55) + ')' : 'rgba(255,93,115,' + (a * .55) + ')'; }
        r.appendChild(td);
      }); t.appendChild(r);
    });
    box.appendChild(t); box.appendChild(el('p', 'gxi-empty', c.note));
  }
  function geo(d) {
    var box = $('gxi-geo-body'); box.textContent = '';
    if (!d.geopolitics.length) return empty(box, 'No theater has a heat score yet.');
    d.geopolitics.forEach(function (g) {
      var r = el('div', 'gxi-geo'); r.appendChild(el('b', null, g.name + ' · heat ' + f(g.score, 0) + (g.delta != null ? ' (' + (g.delta > 0 ? '+' : '') + f(g.delta, 0) + ' vs 24h)' : '')));
      r.appendChild(el('span', 'gxi-met', 'Touches: ' + (g.instruments.join(', ') || '—')));
      if (g.top) { var a = el('a', null, g.top.title); if (/^https?:\/\//i.test(g.top.url || '')) { a.href = g.top.url; a.target = '_blank'; a.rel = 'noopener noreferrer'; } r.appendChild(a); }
      box.appendChild(r);
    });
  }
  function render(d) { [regimes, macro, cb, sessions, vol, futures, cross, corr, geo].forEach(function (fn) { try { fn(d); } catch (e) {} }); $('gxi-note').textContent = d.data_note; }
  function load() { fetch('/events/api/intel', { credentials: 'same-origin' }).then(function (r) { return r.json(); }).then(render).catch(function () {}); }
  load(); setInterval(function () { if (!document.hidden) load(); }, 120000);
})();
