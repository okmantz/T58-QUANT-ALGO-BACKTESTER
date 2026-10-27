/* Fidgets: Infinity Cube, Liquid Playground, Particle Field. */
(function () {
  var PG = window.PG;

  /* ---------------------------------------------------------------- Infinity Cube (a 2x2x2 twisting cube you can orbit and snap) */
  PG.register('fidgets', 'cube', {
    title: 'Infinity Cube', sub: 'Drag a face to twist a layer. Drag the empty space to look around. It never ends.',
    mount: function (host) {
      var C = PG.canvas(host, 1.5), ctx = C.ctx;
      var COLORS = ['#35e0b0', '#4d90ff', '#9b8cff', '#f5b83d', '#ff6b81', '#8fe04a'];   // local face index -> colour
      var NORM = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
      function mmul(a, b) { var r = []; for (var i = 0; i < 3; i++) { r[i] = []; for (var j = 0; j < 3; j++) { r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j]; } } return r; }
      function mvec(m, v) { return [m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2], m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2], m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2]]; }
      function rotAxis(ax, a) {   // ax: 0,1,2
        var c = Math.cos(a), s = Math.sin(a);
        if (ax === 0) return [[1, 0, 0], [0, c, -s], [0, s, c]];
        if (ax === 1) return [[c, 0, s], [0, 1, 0], [-s, 0, c]];
        return [[c, -s, 0], [s, c, 0], [0, 0, 1]];
      }
      var I3 = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
      var cubies = [];
      [-1, 1].forEach(function (x) { [-1, 1].forEach(function (y) { [-1, 1].forEach(function (z) { cubies.push({ p: [x, y, z], R: I3, init: [x, y, z] }); }); }); });
      var yaw = 0.7, pitch = -0.5, vyaw = 0.25, vpitch = 0, twist = null, orbit = null, hit = null, last = null, idle = 0, snaps = PG.store.get('cube-twists', 0);
      var info = PG.el('div', 'pg-bar'); var stat = PG.el('span', 'pg-stat', 'Twists: <b>' + snaps + '</b>'); info.appendChild(stat);
      PG.button(info, 'Shuffle', function () { for (var i = 0; i < 6; i++) setTimeout(function () { if (!twist) startTwist(Math.floor(Math.random() * 3), Math.random() < .5 ? -1 : 1, Math.random() < .5 ? -1 : 1); }, i * 330); });
      host.appendChild(info);
      function view() { return mmul(rotAxis(0, pitch), rotAxis(1, yaw)); }
      function project(v, V) { var w = mvec(V, v), f = 7, s = (f / (f + w[2])) * (C.h * 0.135); return { x: C.w / 2 + w[0] * s, y: C.h / 2 + w[1] * s, z: w[2] }; }
      function faces(V) {
        var out = [];
        cubies.forEach(function (cb, ci) {
          var Rt = cb.R;
          for (var f = 0; f < 6; f++) {
            var n = NORM[f], ax = n[0] ? 0 : n[1] ? 1 : 2, u = (ax + 1) % 3, v = (ax + 2) % 3, sg = n[ax], pts = [];
            [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (q) {
              var l = [0, 0, 0]; l[ax] = sg * .93; l[u] = q[0] * .93; l[v] = q[1] * .93;
              var w = mvec(Rt, l); pts.push(project([w[0] + cb.p[0], w[1] + cb.p[1], w[2] + cb.p[2]], V));
            });
            var wn = mvec(V, mvec(Rt, n));
            if (wn[2] > 0.0001) continue;   // back-facing
            var outer = cb.init[ax] === sg;  // was this face on the outside at the start?
            out.push({ pts: pts, z: (pts[0].z + pts[1].z + pts[2].z + pts[3].z) / 4, color: outer ? COLORS[f] : '#0c141c', shade: .62 + .38 * Math.max(0, -wn[2] * .5 + wn[1] * -.5 + .2), ci: ci, f: f, n: mvec(Rt, n), outer: outer });
          }
        });
        out.sort(function (a, b) { return b.z - a.z; });
        return out;
      }
      function inPoly(p, pts) { var c = false; for (var i = 0, j = 3; i < 4; j = i++) { if (((pts[i].y > p.y) !== (pts[j].y > p.y)) && (p.x < (pts[j].x - pts[i].x) * (p.y - pts[i].y) / (pts[j].y - pts[i].y) + pts[i].x)) c = !c; } return c; }
      function startTwist(ax, layer, dir) { if (twist) return; twist = { ax: ax, layer: layer, dir: dir, t: 0, members: cubies.filter(function (c) { return Math.sign(c.p[ax]) === layer; }) }; }
      function endTwist() {
        var T = twist, M = rotAxis(T.ax, T.dir * Math.PI / 2);
        T.members.forEach(function (c) { c.p = mvec(M, c.p).map(Math.round); c.R = mmul(M, c.R).map(function (r) { return r.map(function (x) { return Math.round(x); }); }); });
        twist = null; snaps++; stat.innerHTML = 'Twists: <b>' + snaps + '</b>'; PG.store.set('cube-twists', snaps);
      }
      PG.drag(C.cv, {
        down: function (e) {
          var p = C.pos(e), V = view(), fs = faces(V); hit = null;
          for (var i = fs.length - 1; i >= 0; i--) { if (inPoly(p, fs[i].pts)) { hit = fs[i]; break; } }
          last = p; orbit = !hit; idle = 0; if (orbit) { vyaw = 0; vpitch = 0; }
          if (hit) hit.start = p;
        },
        move: function (e) {
          var p = C.pos(e); if (!last) return;
          if (orbit) { var dx = p.x - last.x, dy = p.y - last.y; yaw += dx * 0.01; pitch = Math.max(-1.4, Math.min(1.4, pitch + dy * 0.01)); vyaw = dx * 0.01 * 30; last = p; return; }
          if (hit && !twist) {
            var dx2 = p.x - hit.start.x, dy2 = p.y - hit.start.y;
            if (dx2 * dx2 + dy2 * dy2 > 26 * 26) {
              var V = view(), n = hit.n, best = null;
              for (var a = 0; a < 3; a++) {
                if (Math.abs(n[a]) > .5) continue;
                var ax3 = [0, 0, 0]; ax3[a] = 1;
                var t = [ax3[1] * n[2] - ax3[2] * n[1], ax3[2] * n[0] - ax3[0] * n[2], ax3[0] * n[1] - ax3[1] * n[0]];   // direction the surface moves under +rotation about a
                var c0 = project([0, 0, 0], V), c1 = project(t, V), sx = c1.x - c0.x, sy = c1.y - c0.y, len = Math.hypot(sx, sy) || 1, dot = (sx * dx2 + sy * dy2) / len / Math.hypot(dx2, dy2);
                if (!best || Math.abs(dot) > Math.abs(best.dot)) best = { a: a, dot: dot };
              }
              if (best) {
                var cb = cubies[hit.ci], layer = Math.sign(cb.p[best.a]);
                startTwist(best.a, layer, best.dot >= 0 ? 1 : -1);
                hit = null;
              }
            }
          }
        },
        up: function () { last = null; hit = null; orbit = null; }
      });
      var L = PG.loop(function (dt) {
        if (!orbit) { yaw += vyaw * dt; vyaw += (0.25 - vyaw) * Math.min(1, dt * 0.8); }
        if (twist) { twist.t += dt / 0.22; if (twist.t >= 1) endTwist(); }
        var V = view(); ctx.clearRect(0, 0, C.w, C.h);
        var g = ctx.createRadialGradient(C.w / 2, C.h / 2, 10, C.w / 2, C.h / 2, C.h); g.addColorStop(0, '#101c2a'); g.addColorStop(1, '#04070b'); ctx.fillStyle = g; ctx.fillRect(0, 0, C.w, C.h);
        var save = null;
        if (twist) {   // temporarily apply the in-progress rotation to the moving layer
          var e = twist.t < .5 ? 2 * twist.t * twist.t : 1 - Math.pow(-2 * twist.t + 2, 2) / 2, M = rotAxis(twist.ax, twist.dir * e * Math.PI / 2);
          save = twist.members.map(function (c) { return [c.p, c.R]; });
          twist.members.forEach(function (c) { c.p = mvec(M, c.p); c.R = mmul(M, c.R); });
        }
        var fs = faces(V);
        if (save) twist.members.forEach(function (c, i) { c.p = save[i][0]; c.R = save[i][1]; });
        fs.forEach(function (f) {
          ctx.beginPath(); ctx.moveTo(f.pts[0].x, f.pts[0].y); for (var i = 1; i < 4; i++) ctx.lineTo(f.pts[i].x, f.pts[i].y); ctx.closePath();
          ctx.fillStyle = f.color; ctx.fill(); if (f.outer) { ctx.fillStyle = 'rgba(0,0,0,' + (1 - f.shade) * .75 + ')'; ctx.fill(); }
          ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.stroke();
        });
      });
      return { destroy: function () { L.stop(); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Liquid Playground (neon ring on springs) */
  PG.register('fidgets', 'liquid', {
    title: 'Liquid Playground', sub: 'Poke, drag and swirl a neon ring of liquid light.',
    mount: function (host) {
      var C = PG.canvas(host, 1.6), ctx = C.ctx, N = 140, off = new Float32Array(N), vel = new Float32Array(N), ptr = null, prev = null, sparks = [], hue = 170, mode = 'rainbow';
      var bar = PG.el('div', 'pg-bar'); PG.seg(bar, [['rainbow', 'Rainbow'], ['neon', 'Neon cyan'], ['ember', 'Ember']], mode, function (m) { mode = m; }); host.appendChild(bar);
      PG.drag(C.cv, { hover: true, down: function (e) { ptr = C.pos(e); prev = ptr; pushRing(ptr, 1.8); }, move: function (e, down) { prev = ptr; ptr = C.pos(e); if (down || true) pushRing(ptr, down ? 1.6 : .55); }, up: function () {} });
      C.cv.addEventListener('pointerleave', function () { ptr = null; });
      function geo() { var R = Math.min(C.w, C.h) * .3; return { cx: C.w / 2, cy: C.h / 2, R: R }; }
      function pushRing(p, k) {
        var g = geo(), dx = p.x - g.cx, dy = p.y - g.cy, d = Math.hypot(dx, dy), ang = Math.atan2(dy, dx), idx = ((ang / (Math.PI * 2) + 1) % 1) * N, delta = d - g.R;
        if (Math.abs(delta) > g.R * .55) return;
        var dir = delta > 0 ? -1 : 1, spread = 9, speed = prev && p !== prev ? Math.min(40, Math.hypot(p.x - prev.x, p.y - prev.y)) : 6;
        for (var i = -spread; i <= spread; i++) { var j = (Math.round(idx) + i + N * 2) % N, w = Math.exp(-i * i / 26); vel[j] += dir * (4 + speed * .5) * w * k; }
        if (sparks.length < 160 && Math.random() < .6) { var a2 = ang + (Math.random() - .5) * .3; sparks.push({ x: g.cx + Math.cos(a2) * g.R, y: g.cy + Math.sin(a2) * g.R, vx: Math.cos(a2) * 30 * dir * -1 + (Math.random() - .5) * 40, vy: Math.sin(a2) * 30 * dir * -1 + (Math.random() - .5) * 40, life: 1 }); }
      }
      function col(i, a) { var h = mode === 'rainbow' ? (hue + i / N * 360) % 360 : mode === 'neon' ? 180 + Math.sin(i / N * 6.28 * 2) * 18 : 18 + Math.sin(i / N * 6.28 * 2) * 22; return 'hsla(' + h + ',100%,' + (mode === 'ember' ? 58 : 62) + '%,' + a + ')'; }
      var L = PG.loop(function (dt, t) {
        var g = geo(); hue = (hue + dt * 22) % 360;
        var tmp = new Float32Array(N);
        for (var i = 0; i < N; i++) { var l = off[(i + N - 1) % N], r = off[(i + 1) % N]; tmp[i] = (l + r - 2 * off[i]) * 160 - off[i] * 22 - vel[i] * 2.6; }
        for (i = 0; i < N; i++) { vel[i] += tmp[i] * dt; off[i] += vel[i] * dt * 6; if (off[i] > g.R * .6) off[i] = g.R * .6; if (off[i] < -g.R * .6) off[i] = -g.R * .6; }
        // gentle idle ripple so it is never dead
        var idle = Math.sin(t * 1.1) * 1.6; for (i = 0; i < N; i++) off[i] += Math.sin(i / N * 6.283 * 3 + t * 1.3) * idle * dt * 4;
        ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = 'rgba(3,6,12,.34)'; ctx.fillRect(0, 0, C.w, C.h);
        ctx.globalCompositeOperation = 'lighter'; ctx.lineCap = 'round';
        var pts = []; for (i = 0; i < N; i++) { var a = i / N * Math.PI * 2, rr = g.R + off[i]; pts.push([g.cx + Math.cos(a) * rr, g.cy + Math.sin(a) * rr]); }
        [[14, .06], [8, .12], [4, .5], [1.6, 1]].forEach(function (s) {
          ctx.lineWidth = s[0]; for (var i = 0; i < N; i++) { var a = pts[i], b = pts[(i + 1) % N]; ctx.strokeStyle = col(i, s[1]); ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke(); }
        });
        for (i = sparks.length - 1; i >= 0; i--) { var s = sparks[i]; s.life -= dt * 1.4; s.x += s.vx * dt; s.y += s.vy * dt; s.vy += 40 * dt; if (s.life <= 0) { sparks.splice(i, 1); continue; } ctx.fillStyle = col(i * 3, s.life * .9); ctx.beginPath(); ctx.arc(s.x, s.y, 1.2 + s.life * 1.6, 0, 6.283); ctx.fill(); }
      });
      return { destroy: function () { L.stop(); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Particle Field */
  PG.register('fidgets', 'field', {
    title: 'Particle Field', sub: 'Thousands of particles drift on a flow field. Move or hold to attract, repel or swirl them.',
    mount: function (host) {
      var C = PG.canvas(host, 1.6), ctx = C.ctx, small = window.innerWidth < 760, N = small ? 520 : 1100, mode = 'attract', ptr = null, held = false, P = [];
      var bar = PG.el('div', 'pg-bar'); PG.seg(bar, [['attract', 'Attract'], ['repel', 'Repel'], ['swirl', 'Swirl']], mode, function (m) { mode = m; }); host.appendChild(bar);
      function reset() { P = []; for (var i = 0; i < N; i++) P.push({ x: Math.random() * C.w, y: Math.random() * C.h, vx: 0, vy: 0 }); ctx.fillStyle = '#04080e'; ctx.fillRect(0, 0, C.w, C.h); }
      C.onresize = reset; reset();
      PG.drag(C.cv, { hover: true, down: function (e) { held = true; ptr = C.pos(e); }, move: function (e) { ptr = C.pos(e); }, up: function () { held = false; } });
      C.cv.addEventListener('pointerleave', function () { ptr = null; });
      var L = PG.loop(function (dt, t) {
        ctx.fillStyle = 'rgba(4,8,14,.14)'; ctx.fillRect(0, 0, C.w, C.h); ctx.globalCompositeOperation = 'lighter';
        var k = held ? 2.4 : 1;
        for (var i = 0; i < N; i++) {
          var p = P[i], a = Math.sin(p.x * .008 + t * .35) + Math.cos(p.y * .009 - t * .3), fx = Math.cos(a * 2.2) * 26, fy = Math.sin(a * 2.2) * 26;
          if (ptr) {
            var dx = ptr.x - p.x, dy = ptr.y - p.y, d2 = dx * dx + dy * dy + 400, f = 90000 * k / d2, d = Math.sqrt(d2);
            if (mode === 'attract') { fx += dx / d * f; fy += dy / d * f; } else if (mode === 'repel') { fx -= dx / d * f; fy -= dy / d * f; } else { fx += -dy / d * f + dx / d * f * .25; fy += dx / d * f + dy / d * f * .25; }
          }
          p.vx = (p.vx + fx * dt) * .965; p.vy = (p.vy + fy * dt) * .965; var ox = p.x, oy = p.y; p.x += p.vx * dt * 6; p.y += p.vy * dt * 6;
          if (p.x < 0) p.x += C.w; else if (p.x > C.w) p.x -= C.w; if (p.y < 0) p.y += C.h; else if (p.y > C.h) p.y -= C.h;
          if (Math.abs(p.x - ox) > 50 || Math.abs(p.y - oy) > 50) continue;
          var sp = Math.min(1, Math.hypot(p.vx, p.vy) / 90); ctx.strokeStyle = 'hsla(' + (190 + sp * 120) + ',90%,' + (48 + sp * 22) + '%,' + (.5 + sp * .5) + ')'; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(p.x, p.y); ctx.stroke();
        }
        ctx.globalCompositeOperation = 'source-over';
      });
      return { destroy: function () { L.stop(); C.destroy(); } };
    }
  });
})();
