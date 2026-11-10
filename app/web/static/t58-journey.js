/* Journey timeline: a thin strip at the top of every page showing where you are on the First Payout Mission.
   Click a stage to see exactly what must be met before moving on. Reads /api/journey (cached for a minute). */
(function () {
  'use strict';
  if (window.__t58Journey || /^\/(login|offline|static)/.test(location.pathname)) return;
  window.__t58Journey = true;
  var KEY = 't58-journey-v1';
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function cached() { try { var d = JSON.parse(sessionStorage.getItem(KEY) || 'null'); if (d && Date.now() - d.t < 60000) return d.v; } catch (e) {} return null; }
  function store(v) { try { sessionStorage.setItem(KEY, JSON.stringify({ t: Date.now(), v: v })); } catch (e) {} }
  var css = el('style'); css.textContent =
    '.jt{display:flex;flex-wrap:wrap;align-items:center;gap:2px;margin:0 0 14px;padding:5px 10px;border:1px solid var(--border);border-radius:14px;background:var(--panel-2);font-size:11px;color:var(--text-muted);position:relative;overflow:visible;max-width:100%}' +
    '.jt-lab{letter-spacing:.14em;font-weight:800;font-size:9.5px;margin-right:8px;color:var(--text-dim);white-space:nowrap}' +
    '.jt-st{display:flex;align-items:center;gap:5px;background:none;border:0;color:inherit;cursor:pointer;padding:3px 6px;border-radius:999px;font:inherit;white-space:nowrap}' +
    '.jt-st:hover{background:rgba(var(--teal-rgb),.08)}' +
    '.jt-dot{width:8px;height:8px;border-radius:50%;border:1.5px solid var(--border-light);flex:none}' +
    '.jt-st.done .jt-dot{background:var(--ok);border-color:var(--ok)}' +
    '.jt-st.current{color:var(--text)}.jt-st.current .jt-dot{border-color:var(--teal);box-shadow:0 0 0 3px rgba(var(--teal-rgb),.2)}' +
    '.jt-bar{width:14px;height:1.5px;background:var(--border-light);flex:none}.jt-bar.done{background:var(--ok)}' +
    '.jt-pop{position:absolute;top:calc(100% + 8px);left:10px;z-index:9000;width:min(380px,calc(100vw - 32px));background:var(--pop-bg,var(--panel));border:1px solid var(--pop-bd,var(--border-light));border-radius:12px;padding:12px 14px;box-shadow:0 14px 40px rgba(0,0,0,.45);color:var(--text);font-size:12.5px}' +
    '.jt-pop h4{margin:0 0 6px;font-size:13px}.jt-pop .r{display:flex;gap:8px;padding:3px 0}.jt-pop .r i{font-style:normal;width:14px;flex:none}.jt-pop .d{color:var(--text-muted);font-size:11.5px}' +
    '.jt-pop a{color:var(--teal);font-weight:700;text-decoration:none}' +
    '@media(max-width:900px){.jt-st span.t{display:none}.jt-st.current span.t{display:inline}}';
  document.head.appendChild(css);
  function build(d) {
    var main = document.querySelector('.t58-main'); if (!main || document.getElementById('t58-journey')) return;
    var bar = el('div', 'jt'); bar.id = 't58-journey'; bar.setAttribute('role', 'navigation'); bar.setAttribute('aria-label', 'First Payout Mission progress');
    bar.appendChild(el('span', 'jt-lab', 'T58 MISSION'));
    var pop = null;
    function close() { if (pop) { pop.remove(); pop = null; } }
    d.stages.forEach(function (s, i) {
      if (i) bar.appendChild(el('span', 'jt-bar' + (s.state === 'done' || d.stages[i - 1].state === 'done' ? ' done' : '')));
      var b = el('button', 'jt-st ' + s.state); b.type = 'button'; b.setAttribute('aria-label', s.label + (s.state === 'current' ? ' (you are here)' : ''));
      b.appendChild(el('span', 'jt-dot')); b.appendChild(el('span', 't', s.label));
      b.addEventListener('click', function (ev) {
        ev.stopPropagation(); var same = pop && pop.dataset.k === s.key; close(); if (same) return;
        pop = el('div', 'jt-pop'); pop.dataset.k = s.key; pop.appendChild(el('h4', null, (i + 1) + '. ' + s.label + (s.state === 'done' ? ' — done' : (s.state === 'current' ? ' — you are here' : ''))));
        pop.appendChild(el('div', 'd', s.state === 'done' ? 'Everything required here is met.' : 'To move past this step:'));
        s.requirements.forEach(function (r) { var row = el('div', 'r'); row.appendChild(el('i', null, r.met === true ? '✓' : (r.met === false ? '○' : '·'))); var t = el('div'); t.appendChild(document.createTextNode(r.label)); if (r.detail) t.appendChild(el('div', 'd', r.detail)); row.appendChild(t); pop.appendChild(row); });
        var more = el('a', null, 'Open this step'); more.href = s.href; var all = el('a', null, 'Full mission'); all.href = '/mission'; all.style.marginLeft = '14px';
        var f = el('div'); f.style.marginTop = '8px'; f.appendChild(more); f.appendChild(all); pop.appendChild(f); bar.appendChild(pop);
      });
      bar.appendChild(b);
    });
    document.addEventListener('click', function (e) { if (pop && !bar.contains(e.target)) close(); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') close(); });
    var hdr = main.querySelector('.t58-page-header');
    main.insertBefore(bar, hdr || main.firstChild);
  }
  var c = cached();
  if (c) build(c); else fetch('/api/journey').then(function (r) { return r.json(); }).then(function (d) { if (d && d.ok) { store(d); build(d); } }).catch(function () {});
})();
