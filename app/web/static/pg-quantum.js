/* Quantum Playground: Gravity Well, Particle Storm, Orbit Challenge, Signal Path, Light Tunnel. */
(function () {
  var PG = window.PG;
  var TAU = Math.PI * 2;

  /* ---------------------------------------------------------------- Gravity Well */
  PG.register('quantum', 'well', {
    title: 'Gravity Well', sub: 'Drag back from the ball and release to launch it. Bend its path around the wells and into the gold target. Touching a well ends the shot.',
    mount: function (host) {
      var C = PG.canvas(host, 1.7), ctx = C.ctx, level = PG.store.get('well-level', 1), best = PG.store.get('well-best', 1), shots = 0, ball, drag = null, wells = [], target, trail = [], state = 'aim', msg = PG.el('div', 'pg-msg'), start, dead = false;
      var top = PG.el('div', 'pg-bar'); host.appendChild(top); var sL = PG.el('span', 'pg-stat'), sS = PG.el('span', 'pg-stat'); top.appendChild(sL); top.appendChild(sS);
      PG.button(top, 'Reset shot', function () { resetShot(); }); PG.button(top, 'New layout', function () { PG.store.set('well-seed', Math.floor(Math.random() * 1e6)); build(); }); PG.button(top, 'Next level', function () { level++; PG.store.set('well-level', level); build(); }, true);
      host.appendChild(msg);
      function build() {
        var rnd = PG.rng(level * 7919 + (PG.store.get('well-seed', 0))); start = { x: .1, y: .2 + rnd() * .6 }; target = { x: .9, y: .2 + rnd() * .6, r: .035 }; wells = [];
        var n = Math.min(7, 1 + Math.floor((level + 1) / 2));
        for (var i = 0; i < n; i++) wells.push({ x: .3 + rnd() * .4, y: .12 + rnd() * .76, m: .6 + rnd() * .8 + level * .03, r: .03 });
        shots = 0; resetShot(); msg.textContent = 'Level ' + level + '. Pull back and release.'; hud();
      }
      function hud() { sL.innerHTML = 'Level <b>' + level + '</b>'; sS.innerHTML = 'Shots <b>' + shots + '</b>'; }
      function resetShot() { ball = { x: start.x, y: start.y, vx: 0, vy: 0 }; trail = []; state = 'aim'; }
      function S() { return C.h; }   // unit scale: normalised coords x*C.w, y*C.h
      function accel(x, y) { var ax = 0, ay = 0; wells.forEach(function (w) { var dx = w.x - x, dy = (w.y - y) / 1.7, d2 = dx * dx + dy * dy + .002, f = w.m * .09 / d2; var d = Math.sqrt(d2); ax += dx / d * f; ay += dy / d * f * 1.7; }); return [ax, ay]; }
      function simPath(vx, vy, steps) { var x = ball.x, y = ball.y, p = [[x, y]]; for (var i = 0; i < steps; i++) { var a = accel(x, y); vx += a[0] * .016; vy += a[1] * .016; x += vx * .016; y += vy * .016; if (i % 3 === 0) p.push([x, y]); if (x < -.1 || x > 1.1 || y < -.1 || y > 1.1) break; } return p; }
      PG.drag(C.cv, {
        down: function (e) { if (state !== 'aim') return; var p = C.pos(e); drag = { x: p.x / C.w, y: p.y / C.h }; },
        move: function (e, down) { if (drag && down) { var p = C.pos(e); drag.x = p.x / C.w; drag.y = p.y / C.h; } },
        up: function () { if (!drag || state !== 'aim') { drag = null; return; } var vx = (ball.x - drag.x) * 1.4, vy = (ball.y - drag.y) * 1.4, sp = Math.hypot(vx, vy); drag = null; if (sp < .03) return; var m = Math.min(1, .75 / sp); ball.vx = vx * m; ball.vy = vy * m; state = 'fly'; shots++; hud(); }
      });
      var L = PG.loop(function (dt, t) {
        if (state === 'fly') {
          for (var k = 0; k < 2; k++) { var a = accel(ball.x, ball.y); ball.vx += a[0] * dt; ball.vy += a[1] * dt; ball.x += ball.vx * dt; ball.y += ball.vy * dt; }
          trail.push([ball.x, ball.y]); if (trail.length > 400) trail.shift();
          if (Math.hypot((ball.x - target.x) * 1.7, ball.y - target.y) < target.r * 1.7 + .012) { state = 'won'; best = Math.max(best, level); PG.store.set('well-best', best); msg.textContent = 'Landed in ' + shots + ' shot' + (shots > 1 ? 's' : '') + '. Press Next level.'; }
          else if (wells.some(function (w) { return Math.hypot((ball.x - w.x) * 1.7, ball.y - w.y) < w.r * 1.7; })) { state = 'lost'; msg.textContent = 'Swallowed by a well. Reset the shot and try a wider arc.'; setTimeout(function () { if (!dead && state === 'lost') resetShot(); }, 900); }
          else if (ball.x < -.15 || ball.x > 1.15 || ball.y < -.15 || ball.y > 1.15) { state = 'lost'; msg.textContent = 'Lost in space.'; setTimeout(function () { if (!dead && state === 'lost') resetShot(); }, 700); }
        }
        var W = C.w, H = C.h; ctx.fillStyle = '#05070d'; ctx.fillRect(0, 0, W, H);
        // gravity field shading
        wells.forEach(function (w) { var g = ctx.createRadialGradient(w.x * W, w.y * H, 0, w.x * W, w.y * H, H * .3 * w.m); g.addColorStop(0, 'rgba(155,140,255,.5)'); g.addColorStop(1, 'rgba(155,140,255,0)'); ctx.fillStyle = g; ctx.fillRect(0, 0, W, H); });
        wells.forEach(function (w) { ctx.fillStyle = '#000'; ctx.strokeStyle = '#9b8cff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(w.x * W, w.y * H, w.r * H * 1.7 * .6 + 6, 0, TAU); ctx.fill(); ctx.stroke(); });
        ctx.fillStyle = '#f5b83d'; ctx.shadowColor = '#f5b83d'; ctx.shadowBlur = 16; ctx.beginPath(); ctx.arc(target.x * W, target.y * H, 11 + Math.sin(t * 4) * 1.5, 0, TAU); ctx.fill(); ctx.shadowBlur = 0;
        if (trail.length > 1) { ctx.strokeStyle = 'rgba(53,224,176,.5)'; ctx.lineWidth = 2; ctx.beginPath(); trail.forEach(function (p, i) { if (i) ctx.lineTo(p[0] * W, p[1] * H); else ctx.moveTo(p[0] * W, p[1] * H); }); ctx.stroke(); }
        if (drag && state === 'aim') { var vx = (ball.x - drag.x) * 1.4, vy = (ball.y - drag.y) * 1.4, sp = Math.hypot(vx, vy), m = Math.min(1, .75 / (sp || 1)); var pth = simPath(vx * m, vy * m, 140); ctx.setLineDash([3, 6]); ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1.5; ctx.beginPath(); pth.forEach(function (p, i) { if (i) ctx.lineTo(p[0] * W, p[1] * H); else ctx.moveTo(p[0] * W, p[1] * H); }); ctx.stroke(); ctx.setLineDash([]); ctx.strokeStyle = 'rgba(255,255,255,.3)'; ctx.beginPath(); ctx.moveTo(ball.x * W, ball.y * H); ctx.lineTo(drag.x * W, drag.y * H); ctx.stroke(); }
        ctx.fillStyle = '#35e0b0'; ctx.shadowColor = '#35e0b0'; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(ball.x * W, ball.y * H, 6, 0, TAU); ctx.fill(); ctx.shadowBlur = 0;
      });
      build();
      return { destroy: function () { dead = true; L.stop(); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Particle Storm */
  PG.register('quantum', 'storm', {
    title: 'Particle Storm', sub: 'Hold to pour particles in. Switch the force and watch them react: gravity, a black hole, a vortex or repulsion.',
    mount: function (host) {
      var C = PG.canvas(host, 1.7), ctx = C.ctx, small = window.innerWidth < 760, MAX = small ? 1500 : 3500, P = [], mode = 'gravity', ptr = null, down = false, hole = null;
      var bar = PG.el('div', 'pg-bar'); PG.seg(bar, [['gravity', 'Gravity'], ['hole', 'Black hole'], ['vortex', 'Vortex'], ['repel', 'Repel'], ['zero', 'Zero-g']], mode, function (m) { mode = m; hole = null; }); PG.button(bar, 'Clear', function () { P = []; }); host.appendChild(bar);
      PG.drag(C.cv, { hover: true, down: function (e) { down = true; ptr = C.pos(e); if (mode === 'hole') hole = ptr; }, move: function (e) { ptr = C.pos(e); if (down && mode === 'hole') hole = ptr; }, up: function () { down = false; } });
      C.cv.addEventListener('pointerleave', function () { ptr = null; down = false; });
      var L = PG.loop(function (dt, t) {
        var W = C.w, H = C.h;
        if (down && ptr) for (var k = 0; k < (small ? 12 : 24) && P.length < MAX; k++) { var a = Math.random() * TAU, s = Math.random() * 140; P.push({ x: ptr.x, y: ptr.y, vx: Math.cos(a) * s, vy: Math.sin(a) * s - 40, h: (t * 40 + Math.random() * 60) % 360 }); }
        ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = 'rgba(4,7,13,.22)'; ctx.fillRect(0, 0, W, H); ctx.globalCompositeOperation = 'lighter';
        var cx = ptr ? ptr.x : W / 2, cy = ptr ? ptr.y : H / 2, g = mode === 'gravity' ? 420 : mode === 'zero' ? 0 : 0;
        for (var i = P.length - 1; i >= 0; i--) {
          var p = P[i]; p.vy += g * dt;
          if (mode === 'hole' && hole) { var dx = hole.x - p.x, dy = hole.y - p.y, d2 = dx * dx + dy * dy + 900, d = Math.sqrt(d2), f = 2.2e6 / d2; p.vx += dx / d * f * dt; p.vy += dy / d * f * dt; if (d < 14) { P.splice(i, 1); continue; } }
          else if (mode === 'vortex' && ptr) { var dx2 = ptr.x - p.x, dy2 = ptr.y - p.y, d3 = Math.hypot(dx2, dy2) + 30, f2 = 1.4e5 / d3; p.vx += (-dy2 / d3 * f2 + dx2 / d3 * f2 * .35) * dt; p.vy += (dx2 / d3 * f2 + dy2 / d3 * f2 * .35) * dt; }
          else if (mode === 'repel' && ptr) { var dx3 = p.x - ptr.x, dy3 = p.y - ptr.y, d4 = Math.hypot(dx3, dy3) + 20; if (d4 < 220) { var f3 = 9e4 / (d4 * d4) * 30; p.vx += dx3 / d4 * f3 * dt; p.vy += dy3 / d4 * f3 * dt; } }
          p.x += p.vx * dt; p.y += p.vy * dt;
          if (p.x < 0) { p.x = 0; p.vx *= -.8; } else if (p.x > W) { p.x = W; p.vx *= -.8; } if (p.y < 0) { p.y = 0; p.vy *= -.8; } else if (p.y > H) { p.y = H; p.vy *= -.72; p.vx *= .985; }
          var sp = Math.min(1, Math.hypot(p.vx, p.vy) / 420); ctx.fillStyle = 'hsla(' + ((p.h + sp * 80) % 360) + ',95%,' + (50 + sp * 20) + '%,' + (.4 + sp * .5) + ')'; ctx.fillRect(p.x - 1, p.y - 1, 2.2, 2.2);
        }
        ctx.globalCompositeOperation = 'source-over';
        if (mode === 'hole' && hole) { ctx.strokeStyle = 'rgba(155,140,255,.8)'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(hole.x, hole.y, 12 + Math.sin(t * 6) * 2, 0, TAU); ctx.stroke(); }
        ctx.fillStyle = 'rgba(255,255,255,.35)'; ctx.font = '11px ui-monospace,monospace'; ctx.fillText(P.length + ' particles', 8, H - 8);
      });
      return { destroy: function () { L.stop(); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Orbit Challenge */
  PG.register('quantum', 'orbit', {
    title: 'Orbit Challenge', sub: 'You start in a low circular orbit. Burn prograde to climb, then burn again at the top to circularise on the gold ring. Fuel is limited.',
    mount: function (host) {
      var C = PG.canvas(host, 1.7), ctx = C.ctx, GM = 9e5, level = PG.store.get('orbit-level', 1), sat, trail = [], state = 'fly', ringR, planetR = 26, dead = false, holdT = 0, fuel = 100, thrust = 0, keys = {};
      var top = PG.el('div', 'pg-bar'); host.appendChild(top); var sL = PG.el('span', 'pg-stat'), sF = PG.el('span', 'pg-stat'), sE = PG.el('span', 'pg-stat'); top.appendChild(sL); top.appendChild(sF); top.appendChild(sE);
      var bar = PG.el('div', 'pg-bar'); host.appendChild(bar);
      function hold(label, dir) { var b = PG.el('button', 'pg-btn pri', label); b.type = 'button'; b.onpointerdown = function (e) { e.preventDefault(); thrust = dir; }; b.onpointerup = b.onpointerleave = b.onpointercancel = function () { if (thrust === dir) thrust = 0; }; bar.appendChild(b); return b; }
      hold('Burn prograde (hold)  ↑ / W', 1); hold('Burn retrograde (hold)  ↓ / S', -1); PG.button(bar, 'Retry', function () { reset(); }); PG.button(bar, 'Next orbit', function () { level++; PG.store.set('orbit-level', level); reset(); });
      var msg = PG.el('div', 'pg-msg'); host.appendChild(msg);
      function cen() { return { x: C.w / 2, y: C.h / 2 }; }
      function reset() { var m = Math.min(C.w, C.h); ringR = m * Math.min(.44, .2 + ((level * 3) % 7) * .035) ; var r0 = planetR + 24, c = cen(); sat = { x: c.x, y: c.y - r0, vx: Math.sqrt(GM / r0), vy: 0 }; trail = []; state = 'fly'; holdT = 0; fuel = 100; thrust = 0; msg.textContent = 'Orbit ' + level + ': reach a near-circular orbit at the gold ring and hold it for 2 seconds. The dotted line shows where you will go if you stop burning.'; sL.innerHTML = 'Orbit <b>' + level + '</b>'; }
      C.onresize = reset;
      var off = PG.keyHandler(function (e) { if (e.key === 'ArrowUp' || e.key === 'w' || e.key === 'W') { thrust = 1; e.preventDefault(); } else if (e.key === 'ArrowDown' || e.key === 's' || e.key === 'S') { thrust = -1; e.preventDefault(); } });
      function up(e) { if (['ArrowUp', 'w', 'W', 'ArrowDown', 's', 'S'].indexOf(e.key) >= 0) thrust = 0; } window.addEventListener('keyup', up);
      function step(s, dt, th) { var c = cen(), dx = c.x - s.x, dy = c.y - s.y, d2 = dx * dx + dy * dy, d = Math.sqrt(d2), a = GM / d2, v = Math.hypot(s.vx, s.vy) || 1; s.vx += dx / d * a * dt; s.vy += dy / d * a * dt; if (th) { s.vx += s.vx / v * th * 38 * dt; s.vy += s.vy / v * th * 38 * dt; } s.x += s.vx * dt; s.y += s.vy * dt; return d; }
      function elements(s) { var c = cen(), rx = s.x - c.x, ry = s.y - c.y, r = Math.hypot(rx, ry), v2 = s.vx * s.vx + s.vy * s.vy, h = rx * s.vy - ry * s.vx, E = v2 / 2 - GM / r, a = -GM / (2 * E), e = Math.sqrt(Math.max(0, 1 + 2 * E * h * h / (GM * GM))); return { e: e, a: a, E: E, r: r }; }
      var L = PG.loop(function (dt, t) {
        var W = C.w, H = C.h, c = cen(), el = elements(sat);
        if (state === 'fly') {
          var th = fuel > 0 ? thrust : 0; if (th) fuel = Math.max(0, fuel - 11 * dt);
          for (var i = 0; i < 4; i++) { var d = step(sat, dt / 4, th); if (d <= planetR) { state = 'crash'; msg.textContent = 'Crashed into the planet. Retry, and burn less retrograde.'; } }
          trail.push([sat.x, sat.y]); if (trail.length > 500) trail.shift(); el = elements(sat);
          if (el.E >= 0 && el.r > Math.max(W, H)) { state = 'escape'; msg.textContent = 'Escaped orbit. Retry with shorter burns.'; }
          if (state === 'fly' && el.E < 0 && el.e < .09 && Math.abs(el.a - ringR) < 13) { holdT += dt; if (holdT > 2) { state = 'won'; msg.textContent = 'Stable orbit on the ring (fuel left ' + Math.round(fuel) + '%). Press Next orbit.'; } } else holdT = Math.max(0, holdT - dt);
          if (state === 'fly' && fuel <= 0 && el.E < 0 && el.e > .09 && Math.abs(el.a - ringR) >= 13 && !thrust) msg.textContent = 'Out of fuel and not on the ring. Retry.';
        }
        sF.innerHTML = 'Fuel <b>' + Math.round(fuel) + '%</b>'; sE.innerHTML = 'Orbit <b>' + (el.E < 0 ? 'a=' + Math.round(el.a) + ' e=' + el.e.toFixed(2) : 'escape') + '</b>';
        ctx.fillStyle = '#04060b'; ctx.fillRect(0, 0, W, H);
        ctx.strokeStyle = state === 'won' ? '#35e0b0' : 'rgba(245,184,61,.7)'; ctx.lineWidth = 2; ctx.setLineDash([6, 6]); ctx.beginPath(); ctx.arc(c.x, c.y, ringR, 0, TAU); ctx.stroke(); ctx.setLineDash([]);
        var g = ctx.createRadialGradient(c.x - 8, c.y - 8, 4, c.x, c.y, planetR); g.addColorStop(0, '#6fa8ff'); g.addColorStop(1, '#17315a'); ctx.fillStyle = g; ctx.shadowColor = '#4d90ff'; ctx.shadowBlur = 24; ctx.beginPath(); ctx.arc(c.x, c.y, planetR, 0, TAU); ctx.fill(); ctx.shadowBlur = 0;
        if (trail.length > 1) { ctx.strokeStyle = 'rgba(53,224,176,.35)'; ctx.lineWidth = 1.5; ctx.beginPath(); trail.forEach(function (p, i) { if (i) ctx.lineTo(p[0], p[1]); else ctx.moveTo(p[0], p[1]); }); ctx.stroke(); }
        if (state === 'fly') { var s2 = { x: sat.x, y: sat.y, vx: sat.vx, vy: sat.vy }; ctx.setLineDash([2, 5]); ctx.strokeStyle = 'rgba(255,255,255,.5)'; ctx.lineWidth = 1.3; ctx.beginPath(); ctx.moveTo(s2.x, s2.y); for (var k = 0; k < 420; k++) { var dd = step(s2, .02, 0); if (dd <= planetR) break; if (k % 2 === 0) ctx.lineTo(s2.x, s2.y); } ctx.stroke(); ctx.setLineDash([]); }
        ctx.fillStyle = '#35e0b0'; ctx.shadowColor = '#35e0b0'; ctx.shadowBlur = 10; ctx.beginPath(); ctx.arc(sat.x, sat.y, 4.5, 0, TAU); ctx.fill(); ctx.shadowBlur = 0;
        if (thrust && fuel > 0 && state === 'fly') { var v = Math.hypot(sat.vx, sat.vy) || 1, ux = sat.vx / v * thrust, uy = sat.vy / v * thrust; ctx.strokeStyle = '#ffb347'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(sat.x - ux * 6, sat.y - uy * 6); ctx.lineTo(sat.x - ux * (14 + Math.random() * 6), sat.y - uy * (14 + Math.random() * 6)); ctx.stroke(); }
        if (holdT > 0) { ctx.fillStyle = 'rgba(53,224,176,.9)'; ctx.fillRect(c.x - 40, H - 14, 80 * Math.min(1, holdT / 2), 4); }
      });
      reset();
      return { destroy: function () { dead = true; L.stop(); off(); window.removeEventListener('keyup', up); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Signal Path */
  PG.register('quantum', 'signal', {
    title: 'Signal Path', sub: 'Tap mirrors to flip them between / and \\ so the light beam reaches the receiver. Fewer taps is a better score.',
    mount: function (host) {
      var GW = 9, GH = 6, level = PG.store.get('signal-level', 1), grid, src, tgt, taps = 0, won = false, C = PG.canvas(host, GW / GH), ctx = C.ctx, dead = false, flash = 0, par = 0;
      var top = PG.el('div', 'pg-bar'); host.appendChild(top); var sL = PG.el('span', 'pg-stat'), sT = PG.el('span', 'pg-stat'); top.appendChild(sL); top.appendChild(sT);
      PG.button(top, 'New puzzle', function () { build(); }); PG.button(top, 'Next level', function () { level++; PG.store.set('signal-level', level); build(); }, true); var msg = PG.el('div', 'pg-msg'); host.appendChild(msg);
      var DIRS = [[1, 0], [0, 1], [-1, 0], [0, -1]];
      function reflect(d, m) { return m === '/' ? [-d[1], -d[0]] : [d[1], d[0]]; }
      function build() {
        for (var attempt = 0; attempt < 200; attempt++) { if (make()) break; }
        taps = 0; won = false; msg.textContent = 'Level ' + level + '. Turn the mirrors to guide the beam into the receiver (the gold ring).'; hud(); trace();
      }
      function make() {
        var turns = Math.min(8, 2 + Math.floor(level / 2)), g = []; for (var y = 0; y < GH; y++) { g.push([]); for (var x = 0; x < GW; x++) g[y].push(null); }
        var used = {}, x = 0, y = Math.floor(Math.random() * GH), d = [1, 0], path = [[x, y]]; used[x + ',' + y] = 1; var mirrors = [];
        for (var t = 0; t < turns; t++) {
          var run = 1 + Math.floor(Math.random() * 4), ok = true;
          for (var s = 0; s < run; s++) { var nx = x + d[0], ny = y + d[1]; if (nx < 0 || ny < 0 || nx >= GW || ny >= GH || used[nx + ',' + ny]) { ok = s > 0; break; } x = nx; y = ny; used[x + ',' + y] = 1; }
          if (!ok) return false;
          var nd = Math.random() < .5 ? [-d[1], d[0]] : [d[1], -d[0]], m = reflect(d, '/')[0] === nd[0] && reflect(d, '/')[1] === nd[1] ? '/' : '\\';
          if (reflect(d, m)[0] !== nd[0] || reflect(d, m)[1] !== nd[1]) return false;
          var ax = x + nd[0], ay = y + nd[1]; if (ax < 0 || ay < 0 || ax >= GW || ay >= GH || used[ax + ',' + ay]) return false;
          g[y][x] = { m: m, fixed: false, want: m }; mirrors.push(g[y][x]); d = nd;
        }
        var run2 = 1 + Math.floor(Math.random() * 3); for (var s2 = 0; s2 < run2; s2++) { var nx2 = x + d[0], ny2 = y + d[1]; if (nx2 < 0 || ny2 < 0 || nx2 >= GW || ny2 >= GH || used[nx2 + ',' + ny2]) break; x = nx2; y = ny2; used[x + ',' + y] = 1; }
        // the receiver sits at the end of the solved path
        tgt = { x: x, y: y }; if (tgt.x === 0 && tgt.y === path[0][1]) return false;
        src = { y: path[0][1] };
        // decoy mirrors off the path
        var decoys = Math.min(10, 2 + level); for (var k = 0; k < decoys; k++) { var rx = Math.floor(Math.random() * GW), ry = Math.floor(Math.random() * GH); if (used[rx + ',' + ry] || g[ry][rx] || (rx === tgt.x && ry === tgt.y)) continue; g[ry][rx] = { m: Math.random() < .5 ? '/' : '\\', fixed: false, decoy: true }; }
        // scramble: flip a random non-empty subset of real mirrors until the beam does not solve it
        grid = g; par = 0; mirrors.forEach(function (mm) { if (Math.random() < .6) { mm.m = mm.m === '/' ? '\\' : '/'; par++; } });
        if (par === 0) { mirrors[0].m = mirrors[0].m === '/' ? '\\' : '/'; par = 1; }
        return !trace().hit;
      }
      function trace() {
        var x = -1, y = src.y, d = [1, 0], pts = [[x, y]], hit = false, n = 0;
        while (n++ < 200) { x += d[0]; y += d[1]; if (x < 0 || y < 0 || x >= GW || y >= GH) { pts.push([x, y]); break; } pts.push([x, y]); if (x === tgt.x && y === tgt.y) { hit = true; break; } var c = grid[y][x]; if (c) d = reflect(d, c.m); }
        trace.pts = pts; return { hit: hit, pts: pts };
      }
      function hud() { sL.innerHTML = 'Level <b>' + level + '</b>'; sT.innerHTML = 'Taps <b>' + taps + '</b> &middot; par ' + par; }
      C.cv.addEventListener('pointerdown', function (e) {
        if (won) return; var p = C.pos(e), cw = C.w / GW, x = Math.floor(p.x / cw), y = Math.floor(p.y / (C.h / GH)); if (x < 0 || y < 0 || x >= GW || y >= GH) return;
        var c = grid[y][x]; if (!c) return; c.m = c.m === '/' ? '\\' : '/'; taps++; flash = 1; var r = trace(); hud();
        if (r.hit) { won = true; msg.textContent = 'Signal received in ' + taps + ' tap' + (taps > 1 ? 's' : '') + (taps <= par ? ' (par!)' : '') + '. Press Next level.'; }
      });
      var L = PG.loop(function (dt, t) {
        var W = C.w, H = C.h, cw = W / GW, ch = H / GH; flash = Math.max(0, flash - dt * 3); ctx.fillStyle = '#05080d'; ctx.fillRect(0, 0, W, H);
        ctx.strokeStyle = 'rgba(255,255,255,.06)'; ctx.lineWidth = 1; for (var i = 1; i < GW; i++) { ctx.beginPath(); ctx.moveTo(i * cw, 0); ctx.lineTo(i * cw, H); ctx.stroke(); } for (i = 1; i < GH; i++) { ctx.beginPath(); ctx.moveTo(0, i * ch); ctx.lineTo(W, i * ch); ctx.stroke(); }
        var P = trace.pts; ctx.lineCap = 'round'; [[10, .12], [4, .4], [1.8, 1]].forEach(function (s) { ctx.strokeStyle = won ? 'rgba(53,224,176,' + s[1] + ')' : 'rgba(255,230,140,' + s[1] + ')'; ctx.lineWidth = s[0]; ctx.beginPath(); P.forEach(function (p, i) { var px = (p[0] + .5) * cw, py = (p[1] + .5) * ch; if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py); }); ctx.stroke(); });
        ctx.fillStyle = '#4d90ff'; ctx.fillRect(0, (src.y + .5) * ch - 8, 10, 16);
        ctx.strokeStyle = won ? '#35e0b0' : '#f5b83d'; ctx.lineWidth = 3; ctx.beginPath(); ctx.arc((tgt.x + .5) * cw, (tgt.y + .5) * ch, Math.min(cw, ch) * .28 + Math.sin(t * 4) * 1.5, 0, TAU); ctx.stroke();
        for (var y = 0; y < GH; y++) for (var x = 0; x < GW; x++) { var c = grid[y][x]; if (!c) continue; var cx = (x + .5) * cw, cy = (y + .5) * ch, r = Math.min(cw, ch) * .36; ctx.strokeStyle = '#9b8cff'; ctx.lineWidth = 5; ctx.shadowColor = '#9b8cff'; ctx.shadowBlur = 8; ctx.beginPath(); if (c.m === '/') { ctx.moveTo(cx - r, cy + r); ctx.lineTo(cx + r, cy - r); } else { ctx.moveTo(cx - r, cy - r); ctx.lineTo(cx + r, cy + r); } ctx.stroke(); ctx.shadowBlur = 0; }
      });
      build();
      return { destroy: function () { dead = true; L.stop(); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Light Tunnel */
  PG.register('quantum', 'tunnel', {
    title: 'Light Tunnel', sub: 'Fly through rings of light. Steer around the tunnel (arrows, drag, or tilt left/right on the pad) and slip through each gap.',
    mount: function (host) {
      var C = PG.canvas(host, 1.7), ctx = C.ctx, ang = -Math.PI / 2, targetAng = ang, rings = [], z0 = 0, speed = 1, score = 0, best = PG.store.get('tunnel-best', 0), over = false, started = false, hue = 180, keys = {}, dead = false;
      var top = PG.el('div', 'pg-bar'); host.appendChild(top); var sS = PG.el('span', 'pg-stat'), sB = PG.el('span', 'pg-stat'); top.appendChild(sS); top.appendChild(sB); PG.button(top, 'Restart', function () { reset(); }, true);
      var msg = PG.el('div', 'pg-msg', 'Press an arrow key, or touch and drag, to start.'); host.appendChild(msg);
      function reset() { rings = []; z0 = 0; speed = 1; score = 0; over = false; started = false; ang = -Math.PI / 2; targetAng = ang; for (var i = 0; i < 14; i++) addRing(i * 1.1 + 3); msg.textContent = 'Press an arrow key, or touch and drag, to start.'; hud(); }
      function addRing(z) { var gap = Math.max(.55, 1.15 - score / 3500); rings.push({ z: z, gapCenter: Math.random() * TAU, gap: gap, passed: false }); }
      function hud() { sS.innerHTML = 'Distance <b>' + Math.floor(score) + '</b>'; sB.innerHTML = 'Best <b>' + Math.floor(best) + '</b>'; }
      var off = PG.keyHandler(function (e) { if (['ArrowLeft', 'ArrowRight', 'a', 'd', 'A', 'D'].indexOf(e.key) >= 0) { e.preventDefault(); keys[e.key] = 1; started = true; msg.textContent = ''; } if (e.key === ' ' && over) reset(); });
      var offUp = (function () { function up(e) { delete keys[e.key]; } window.addEventListener('keyup', up); return function () { window.removeEventListener('keyup', up); }; })();
      PG.drag(C.cv, { down: function (e) { if (over) { reset(); return; } started = true; msg.textContent = ''; setA(e); }, move: function (e, dn) { if (dn) setA(e); }, up: function () {} });
      function setA(e) { var p = C.pos(e); targetAng = Math.atan2(p.y - C.h / 2, p.x - C.w / 2); }
      function angDiff(a, b) { var d = (a - b) % TAU; if (d > Math.PI) d -= TAU; if (d < -Math.PI) d += TAU; return d; }
      var L = PG.loop(function (dt, t) {
        var W = C.w, H = C.h, cx = W / 2, cy = H / 2, R0 = Math.min(W, H) * .46;
        if (started && !over) {
          var turn = (keys.ArrowRight || keys.d || keys.D ? 1 : 0) - (keys.ArrowLeft || keys.a || keys.A ? 1 : 0); if (turn) targetAng = ang + turn * 3.2 * dt * 2; var dA = angDiff(targetAng, ang); ang += Math.max(-7 * dt, Math.min(7 * dt, dA * 12 * dt));
          speed = 1 + Math.min(2.2, score / 900); var dz = speed * dt * 2.6; score += dz * 18; rings.forEach(function (r) { r.z -= dz; });
          rings.forEach(function (r) { if (!r.passed && r.z <= .5) { r.passed = true; if (Math.abs(angDiff(ang, r.gapCenter)) > r.gap / 2) { over = true; if (score > best) { best = score; PG.store.set('tunnel-best', best); } msg.textContent = 'Crashed at ' + Math.floor(score) + '. Tap or press Space to try again.'; } } });
          rings = rings.filter(function (r) { return r.z > -.3; }); while (rings.length < 14) addRing((rings.length ? rings[rings.length - 1].z : 3) + 1.1); hue = (hue + dt * 14) % 360; hud();
        }
        ctx.fillStyle = '#02040a'; ctx.fillRect(0, 0, W, H);
        var order = rings.slice().sort(function (a, b) { return b.z - a.z; }); ctx.lineCap = 'round';
        order.forEach(function (r) {
          var sc = 1 / Math.max(.18, r.z + .35), rad = R0 * sc * .42; if (rad > Math.max(W, H)) return; var a = Math.min(1, sc * 1.4), c = 'hsla(' + ((hue + r.z * 16) % 360) + ',95%,62%,' + Math.min(1, a) + ')';
          ctx.strokeStyle = c; ctx.lineWidth = Math.max(1.5, 7 * sc); ctx.shadowColor = c; ctx.shadowBlur = 10 * Math.min(1.5, sc);
          var gs = r.gapCenter - r.gap / 2, ge = r.gapCenter + r.gap / 2; ctx.beginPath(); ctx.arc(cx, cy, rad, ge, gs + TAU); ctx.stroke(); ctx.shadowBlur = 0;
          ctx.strokeStyle = 'rgba(255,255,255,' + a * .12 + ')'; ctx.lineWidth = 1; for (var k = 0; k < 8; k++) { var aa = k * TAU / 8; ctx.beginPath(); ctx.moveTo(cx + Math.cos(aa) * rad, cy + Math.sin(aa) * rad); ctx.lineTo(cx + Math.cos(aa) * rad * .86, cy + Math.sin(aa) * rad * .86); ctx.stroke(); }
        });
        var sx = cx + Math.cos(ang) * R0 * .42 * .86, sy = cy + Math.sin(ang) * R0 * .42 * .86; ctx.fillStyle = '#fff'; ctx.shadowColor = '#35e0b0'; ctx.shadowBlur = 16; ctx.beginPath(); ctx.arc(sx, sy, 7, 0, TAU); ctx.fill(); ctx.shadowBlur = 0;
        ctx.save(); ctx.translate(sx, sy); ctx.rotate(ang + Math.PI / 2); ctx.fillStyle = '#35e0b0'; ctx.beginPath(); ctx.moveTo(0, -12); ctx.lineTo(7, 7); ctx.lineTo(-7, 7); ctx.closePath(); ctx.fill(); ctx.restore();
      });
      reset();
      return { destroy: function () { dead = true; L.stop(); off(); offUp(); C.destroy(); } };
    }
  });
})();
