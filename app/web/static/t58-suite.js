/* Tiny helpers shared by the prop-journey pages. No dependencies; all text goes in through textContent. */
(function (w) {
  'use strict';
  var T = {};
  T.el = function (tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  T.money = function (n, signed) { if (n == null || isNaN(n)) return '—'; var a = Math.abs(n), s = '$' + a.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: a % 1 ? 2 : 0 }); return n < 0 ? '-' + s : (signed && n > 0 ? '+' + s : s); };
  T.api = function (m, u, b) { return fetch(u, { method: m, headers: { 'Content-Type': 'application/json' }, body: b ? JSON.stringify(b) : undefined }).then(function (r) { return r.json(); }); };
  T.toast = function (t) { var d = T.el('div', 's-toast', t); document.body.appendChild(d); setTimeout(function () { d.remove(); }, 2400); };
  T.badge = function (text, kind) { return T.el('span', 's-badge ' + (kind || ''), text); };
  T.kv = function (pairs) { var kv = T.el('div', 's-kv'); pairs.forEach(function (x) { var d = T.el('div'); d.appendChild(T.el('div', 'k', x[0])); d.appendChild(T.el('div', 'v', x[1] == null ? '—' : String(x[1]))); kv.appendChild(d); }); return kv; };
  T.meter = function (frac, kind) { var m = T.el('div', 's-meter ' + (kind || '')), i = T.el('i'); i.style.width = Math.max(0, Math.min(100, (frac || 0) * 100)) + '%'; m.appendChild(i); return m; };
  T.levelKind = function (l) { return { ok: 'ok', warn: 'warn', reduce: 'warn', block: 'bad', flatten: 'bad', watch: 'warn', warning: 'warn', critical: 'bad' }[l] || ''; };
  T.modal = function (title, build) {
    var back = T.el('div', 's-modal-back show'), m = T.el('div', 's-modal'); m.appendChild(T.el('h3', null, title)); back.appendChild(m);
    var close = function () { back.remove(); }; back.addEventListener('click', function (e) { if (e.target === back) close(); });
    build(m, close); document.body.appendChild(back); return close;
  };
  T.field = function (label, type, value) { var l = T.el('label', 's-field', label), i = T.el('input', 's-input'); i.type = type || 'text'; if (value != null) i.value = value; l.appendChild(i); return { label: l, input: i }; };
  w.T58 = T;
})(window);
