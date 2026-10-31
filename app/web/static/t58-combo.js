/* T58 combo: turns a long grouped <select> (stored datasets) into a searchable popover that stays open until you pick, press Esc or click away.
 * The native <select> stays in the form (hidden) and keeps the value, so form posts and any onchange handlers behave exactly as before.
 * Usage: T58Combo.enhance('select[name$="_existing_dataset"]'). Safe to call twice; a page without JS keeps the native select. */
(function () {
  'use strict';
  var CSS = '.t58c{position:relative;width:100%}.t58c-btn{width:100%;display:flex;align-items:center;justify-content:space-between;gap:10px;text-align:left;cursor:pointer;' +
    'padding:10px 12px;border-radius:10px;border:1px solid var(--border-light);background:var(--panel-2);color:var(--text);font:inherit;font-size:14px}' +
    '.t58c-btn:focus-visible,.t58c.open .t58c-btn{outline:none;border-color:var(--teal);box-shadow:0 0 0 3px rgba(var(--teal-rgb),.18)}' +
    '.t58c-btn .v{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.t58c-btn .v.ph{color:var(--text-dim)}' +
    '.t58c-btn .car{flex:none;width:8px;height:8px;border-right:2px solid var(--text-muted);border-bottom:2px solid var(--text-muted);transform:rotate(45deg) translateY(-2px);transition:transform .15s}' +
    '.t58c.open .car{transform:rotate(225deg) translateY(-2px)}' +
    '.t58c-pop{position:absolute;left:0;right:0;top:calc(100% + 6px);z-index:60;background:var(--pop-bg,var(--panel));border:1px solid var(--pop-bd,var(--border-light));border-radius:12px;' +
    'box-shadow:var(--shadow-lg,0 14px 38px rgba(0,0,0,.5));opacity:0;transform:translateY(-4px);pointer-events:none;visibility:hidden;transition:opacity .12s,transform .12s,visibility 0s .12s;overflow:hidden}' +
    '.t58c.open .t58c-pop{opacity:1;transform:none;pointer-events:auto;visibility:visible;transition:opacity .12s,transform .12s}' +
    '.t58c-pop input{width:100%;box-sizing:border-box;border:0;border-bottom:1px solid var(--border);border-radius:0;background:transparent;color:var(--text);padding:10px 12px;font:inherit;font-size:13px}' +
    '.t58c-pop input:focus{outline:none;box-shadow:none}' +
    '.t58c-list{max-height:260px;overflow-y:auto;padding:4px;overscroll-behavior:contain}' +
    '.t58c-grp{padding:8px 10px 3px;font-size:10.5px;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--teal)}' +
    '.t58c-opt{padding:7px 10px;border-radius:8px;cursor:pointer;font-size:13.5px;color:var(--text)}.t58c-opt.hl{background:rgba(var(--teal-rgb),.16)}' +
    '.t58c-opt.sel{font-weight:700}.t58c-opt.empty{color:var(--text-dim)}.t58c-none{padding:12px;font-size:13px;color:var(--text-muted)}';
  var openOne = null;
  function el(t, c, x) { var e = document.createElement(t); if (c) e.className = c; if (x != null) e.textContent = x; return e; }
  function css() { if (document.getElementById('t58-combo-css')) return; var s = el('style'); s.id = 't58-combo-css'; s.textContent = CSS; document.head.appendChild(s); }

  function build(sel) {
    if (sel.__t58c) return; sel.__t58c = true;
    var items = [];                                        // {group, value, label, empty}
    Array.prototype.forEach.call(sel.children, function (n) {
      if (n.tagName === 'OPTGROUP') Array.prototype.forEach.call(n.children, function (o) { items.push({ g: n.label, v: o.value, t: o.textContent, e: /empty/i.test(o.textContent) }); });
      else if (n.tagName === 'OPTION') items.push({ g: '', v: n.value, t: n.textContent, e: false });
    });
    var root = el('div', 't58c'), btn = el('button', 't58c-btn'), val = el('span', 'v'), car = el('span', 'car'),
        pop = el('div', 't58c-pop'), q = el('input'), list = el('div', 't58c-list');
    btn.type = 'button'; btn.setAttribute('aria-haspopup', 'listbox'); btn.setAttribute('aria-expanded', 'false');
    q.type = 'text'; q.placeholder = 'Search datasets…'; q.setAttribute('aria-label', 'Search datasets'); q.autocomplete = 'off';
    list.setAttribute('role', 'listbox');
    btn.appendChild(val); btn.appendChild(car); pop.appendChild(q); pop.appendChild(list); root.appendChild(btn); root.appendChild(pop);
    sel.style.display = 'none'; sel.parentNode.insertBefore(root, sel.nextSibling);
    var shown = [], hl = -1;

    function label() { var cur = items.filter(function (i) { return i.v === sel.value; })[0]; val.textContent = cur && cur.v ? cur.t : (items[0] ? items[0].t : 'Choose…'); val.className = 'v' + (cur && cur.v ? '' : ' ph'); }
    function render() {
      var term = q.value.trim().toLowerCase(), last = null; list.innerHTML = ''; shown = [];
      items.forEach(function (i) {
        if (term && (i.t + ' ' + i.g).toLowerCase().indexOf(term) < 0) return;
        if (i.g !== last && i.g) { list.appendChild(el('div', 't58c-grp', i.g)); } last = i.g;
        var o = el('div', 't58c-opt' + (i.v === sel.value ? ' sel' : '') + (i.e ? ' empty' : ''), i.t); o.setAttribute('role', 'option');
        o.__i = i; o.addEventListener('mousedown', function (ev) { ev.preventDefault(); pick(i); });
        o.addEventListener('mousemove', function () { setHl(shown.indexOf(o)); });
        list.appendChild(o); shown.push(o);
      });
      if (!shown.length) list.appendChild(el('div', 't58c-none', 'No dataset matches “' + q.value + '”.'));
      setHl(shown.length ? Math.max(0, shown.map(function (o) { return o.__i.v; }).indexOf(sel.value)) : -1, true);
    }
    function setHl(n, scroll) { hl = n; shown.forEach(function (o, k) { o.classList.toggle('hl', k === n); }); if (scroll && shown[n]) shown[n].scrollIntoView({ block: 'nearest' }); }
    function open() { if (openOne && openOne !== api) openOne.close(); css(); root.classList.add('open'); btn.setAttribute('aria-expanded', 'true'); q.value = ''; render(); openOne = api; setTimeout(function () { q.focus(); }, 0); }
    function close(refocus) { root.classList.remove('open'); btn.setAttribute('aria-expanded', 'false'); if (openOne === api) openOne = null; if (refocus) btn.focus(); }
    function pick(i) {
      sel.value = i.v; label(); close(true);
      sel.dispatchEvent(new Event('change', { bubbles: true }));                 // runs any inline onchange="…" too
    }
    var api = { close: close, root: root };
    btn.addEventListener('click', function () { root.classList.contains('open') ? close() : open(); });
    btn.addEventListener('keydown', function (e) { if (e.key === 'ArrowDown' || e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
    q.addEventListener('input', render);
    q.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setHl(Math.min(shown.length - 1, hl + 1), true); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setHl(Math.max(0, hl - 1), true); }
      else if (e.key === 'Enter') { e.preventDefault(); if (shown[hl]) pick(shown[hl].__i); }
      else if (e.key === 'Escape' || e.key === 'Tab') { close(e.key === 'Escape'); }
    });
    sel.addEventListener('change', label);
    label();
  }
  document.addEventListener('mousedown', function (e) { if (openOne && !openOne.root.contains(e.target)) openOne.close(); });
  window.T58Combo = { enhance: function (selector) { css(); Array.prototype.forEach.call(document.querySelectorAll(selector), build); } };
})();
