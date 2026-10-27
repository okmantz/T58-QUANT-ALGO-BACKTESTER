/* Playground shared helpers. Everything runs locally: no network, no market data. Each toy is registered with
   PG.register(section, id, {title, sub, mount(el, ctx)}) and mount() returns {destroy()}; the page mounts one toy at a time
   so only the visible one ever runs, and every animation loop here pauses while the tab is hidden. */
(function () {
  var reg = {};
  var PG = window.PG = {
    register: function (section, id, def) { (reg[section] = reg[section] || []).push(Object.assign({ id: id }, def)); },
    list: function (section) { return reg[section] || []; },
    store: {
      get: function (k, d) { try { var v = localStorage.getItem('t58-pg-' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
      set: function (k, v) { try { localStorage.setItem('t58-pg-' + k, JSON.stringify(v)); } catch (e) {} }
    },
    el: function (tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; },
    /* mulberry32 seeded RNG: same seed -> same sequence (daily challenges) */
    rng: function (seed) { var a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; },
    dailySeed: function () { var d = new Date(); return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate(); },
    today: function () { var d = new Date(); return d.getFullYear() + '-' + ('0' + (d.getMonth() + 1)).slice(-2) + '-' + ('0' + d.getDate()).slice(-2); },
    fmtTime: function (s) { s = Math.max(0, Math.floor(s)); return Math.floor(s / 60) + ':' + ('0' + (s % 60)).slice(-2); },
    /* A canvas that fills its wrapper at a fixed aspect ratio, DPR-aware (capped at 2 so phones never render 3x). */
    canvas: function (host, aspect) {
      var wrap = PG.el('div', 'pg-cv-wrap'); wrap.style.aspectRatio = String(aspect || 1.6);
      var cv = document.createElement('canvas'); wrap.appendChild(cv); host.appendChild(wrap);
      var ctx = cv.getContext('2d'), o = { cv: cv, ctx: ctx, wrap: wrap, w: 0, h: 0, dpr: 1, onresize: null };
      function fit() {
        var r = wrap.getBoundingClientRect(); if (!r.width) return;
        o.dpr = Math.min(2, window.devicePixelRatio || 1); o.w = Math.round(r.width); o.h = Math.round(r.height);
        cv.width = Math.round(o.w * o.dpr); cv.height = Math.round(o.h * o.dpr); ctx.setTransform(o.dpr, 0, 0, o.dpr, 0, 0);
        if (o.onresize) o.onresize();
      }
      var ro = window.ResizeObserver ? new ResizeObserver(fit) : null; if (ro) ro.observe(wrap); else window.addEventListener('resize', fit);
      o.fit = fit; fit();
      o.pos = function (e) { var r = cv.getBoundingClientRect(); return { x: (e.clientX - r.left) * (o.w / r.width), y: (e.clientY - r.top) * (o.h / r.height) }; };
      o.destroy = function () { if (ro) ro.disconnect(); else window.removeEventListener('resize', fit); wrap.remove(); };
      return o;
    },
    /* requestAnimationFrame loop with dt (seconds, clamped) that stops while the page is hidden and when destroyed. */
    loop: function (fn) {
      var id = 0, last = 0, dead = false;
      function frame(t) { if (dead) return; var dt = Math.min(0.05, (t - last) / 1000 || 0.016); last = t; try { fn(dt, t / 1000); } catch (e) { console.error(e); } id = requestAnimationFrame(frame); }
      function vis() { if (document.hidden) { cancelAnimationFrame(id); id = 0; } else if (!id && !dead) { last = performance.now(); id = requestAnimationFrame(frame); } }
      document.addEventListener('visibilitychange', vis); id = requestAnimationFrame(frame);
      return { stop: function () { dead = true; cancelAnimationFrame(id); document.removeEventListener('visibilitychange', vis); } };
    },
    /* One call wires mouse/touch/pen pointer events with capture. */
    drag: function (el, h) {
      var down = false;
      el.addEventListener('pointerdown', function (e) { down = true; try { el.setPointerCapture(e.pointerId); } catch (x) {} h.down && h.down(e); });
      el.addEventListener('pointermove', function (e) { if (down || h.hover) h.move && h.move(e, down); });
      function up(e) { if (!down) return; down = false; h.up && h.up(e); }
      el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
    },
    reducedMotion: function () { try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; } },
    keyHandler: function (fn) { function k(e) { var t = e.target && e.target.tagName; if (t === 'INPUT' || t === 'TEXTAREA' || t === 'SELECT') return; fn(e); } window.addEventListener('keydown', k); return function () { window.removeEventListener('keydown', k); }; },
    seg: function (host, opts, value, onchange) {
      var s = PG.el('div', 'pg-seg'); opts.forEach(function (o) { var b = PG.el('button', value === o[0] ? 'on' : '', o[1]); b.type = 'button'; b.onclick = function () { [].forEach.call(s.children, function (c) { c.classList.remove('on'); }); b.classList.add('on'); onchange(o[0]); }; s.appendChild(b); });
      host.appendChild(s); return s;
    },
    button: function (host, label, fn, pri) { var b = PG.el('button', 'pg-btn' + (pri ? ' pri' : ''), label); b.type = 'button'; b.onclick = fn; host.appendChild(b); return b; }
  };
  /* Section page: tabs + one mounted toy at a time. */
  PG.page = function (section, root) {
    var items = PG.list(section), tabs = PG.el('div', 'pg-tabs'), stage = PG.el('div', 'pg-stage'), cur = null;
    root.appendChild(tabs); root.appendChild(stage);
    function show(id) {
      if (cur && cur.inst && cur.inst.destroy) { try { cur.inst.destroy(); } catch (e) {} }
      var def = items.filter(function (i) { return i.id === id; })[0] || items[0]; cur = { def: def };
      [].forEach.call(tabs.children, function (b) { b.classList.toggle('on', b.dataset.id === def.id); });
      stage.innerHTML = ''; stage.appendChild(PG.el('h3', '', def.title)); stage.appendChild(PG.el('p', 'sub', def.sub || ''));
      var body = PG.el('div'); stage.appendChild(body);
      try { cur.inst = def.mount(body, {}); } catch (e) { console.error(e); body.textContent = 'This toy could not start: ' + e.message; }
      try { history.replaceState(null, '', '#' + def.id); } catch (e) {}
    }
    items.forEach(function (d) { var b = PG.el('button', 'pg-tab', d.title); b.type = 'button'; b.dataset.id = d.id; b.onclick = function () { show(d.id); }; tabs.appendChild(b); });
    show((location.hash || '').slice(1) || items[0].id);
    window.addEventListener('hashchange', function () { var id = (location.hash || '').slice(1); if (id && (!cur || cur.def.id !== id)) show(id); });
    window.addEventListener('beforeunload', function () { if (cur && cur.inst && cur.inst.destroy) cur.inst.destroy(); });
  };
})();
