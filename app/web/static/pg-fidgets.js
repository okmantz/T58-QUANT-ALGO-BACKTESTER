/* Fidgets: Infinity Cube, Liquid Playground, Particle Field. */
(function () {
  var PG = window.PG;

  /* ---------------------------------------------------------------- Infinity Cube (a 2x2x2 twisting cube: a fidget with hidden combos) */
  PG.register('fidgets', 'cube', {
    title: 'Infinity Cube', sub: 'Twist it, spin it, mash it. Chain moves in the right order and the cube does something you did not expect.',
    mount: function (host) {
      var C = PG.canvas(host, 1.5), ctx = C.ctx, dead = false;
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
      var yaw = 0.7, pitch = -0.5, vyaw = 0.25, twist = null, orbit = null, hit = null, last = null, queue = [], hist = [], seq = [], taps = [];
      var snaps = PG.store.get('cube-twists', 0), found = PG.store.get('cube-secrets', []), soundOn = PG.store.get('cube-sound', false);
      var fx = { rainbow: 0, wobble: 0, party: 0, neon: false, ex: 0, exv: 0, bits: [] }, clock = 0, actx = null, toastT = 0, toastMsg = '';
      var AX = ['X', 'Y', 'Z'];
      /* The secrets. Moves are written axis + direction; the move trail under the cube shows what you just did. */
      var SECRETS = [
        { id: 'prism', name: 'Prism', hint: 'Make the same twist four times in a row.' },
        { id: 'bang', name: 'Big Bang', hint: 'Twist X↻ Y↻ X↺ Y↺, in that order.' },
        { id: 'wobble', name: 'Jelly', hint: 'Twist X↻ X↺ X↻ X↺ (back and forth).' },
        { id: 'confetti', name: 'Confetti', hint: 'Twist X↻ Y↻ Z↻ twice around.' },
        { id: 'neon', name: 'Neon', hint: 'Tap the cube five times fast, without dragging. Tap five more to turn it off.' },
        { id: 'party', name: 'Party', hint: 'The old cheat code: ↑ ↑ ↓ ↓ ← → ← → B A on your keyboard.' }
      ];
      function tok(m) { return AX[m.ax] + (m.dir > 0 ? '+' : '-'); }
      function sound(kind) {
        if (!soundOn) return;
        try {
          actx = actx || new (window.AudioContext || window.webkitAudioContext)();
          var o = actx.createOscillator(), g = actx.createGain(), t = actx.currentTime;
          o.type = 'triangle'; o.frequency.setValueAtTime(kind === 'win' ? 520 : 180 + Math.random() * 60, t);
          if (kind === 'win') o.frequency.exponentialRampToValueAtTime(1040, t + .18);
          g.gain.setValueAtTime(.0001, t); g.gain.exponentialRampToValueAtTime(kind === 'win' ? .12 : .05, t + .008); g.gain.exponentialRampToValueAtTime(.0001, t + (kind === 'win' ? .3 : .07));
          o.connect(g); g.connect(actx.destination); o.start(t); o.stop(t + .32);
        } catch (e) {}
      }
      function buzz(ms) { if (soundOn) { try { navigator.vibrate && navigator.vibrate(ms); } catch (e) {} } }

      var info = PG.el('div', 'pg-bar'); host.appendChild(info);
      var stat = PG.el('span', 'pg-stat', ''); info.appendChild(stat);
      PG.button(info, 'Shuffle', function () { shuffle(); });
      PG.button(info, 'Spin', function () { vyaw = 7; fx.spin = 1; sound(); });
      PG.button(info, 'Undo', function () { var m = hist.pop(); if (m) { queue.push({ ax: m.ax, layer: m.layer, dir: -m.dir, quiet: true, nolog: true }); } });
      PG.button(info, 'Solve', function () { var rev = hist.slice().reverse(); hist = []; rev.forEach(function (m) { queue.push({ ax: m.ax, layer: m.layer, dir: -m.dir, quiet: true, nolog: true, fast: true }); }); });
      var snd = PG.button(info, '', function () { soundOn = !soundOn; PG.store.set('cube-sound', soundOn); snd.textContent = 'Sound & buzz: ' + (soundOn ? 'on' : 'off'); sound(); }); snd.textContent = 'Sound & buzz: ' + (soundOn ? 'on' : 'off');
      var trail = PG.el('div', 'cube-trail'); host.appendChild(trail);
      var sec = PG.el('div', 'cube-secrets'); host.appendChild(sec);
      var showHints = false;
      function drawStats() { stat.innerHTML = 'Twists: <b>' + snaps + '</b>'; }
      function drawTrail() { trail.innerHTML = seq.length ? '<span>Move trail</span>' + seq.slice(-10).map(function (t) { return '<i>' + t.replace('+', '↻').replace('-', '↺') + '</i>'; }).join('') : '<span>Move trail appears here as you twist.</span>'; }
      function drawSecrets() {
        sec.innerHTML = '<div class="hd"><b>Secrets ' + found.length + ' / ' + SECRETS.length + '</b><button type="button" class="pg-btn" id="cube-hints">' + (showHints ? 'Hide hints' : 'Show hints') + '</button></div><div class="chips">' +
          SECRETS.map(function (x) { var ok = found.indexOf(x.id) >= 0; return '<div class="chip' + (ok ? ' ok' : '') + '"><b>' + (ok ? x.name : '???') + '</b>' + (ok || showHints ? '<small>' + x.hint + '</small>' : '') + '</div>'; }).join('') + '</div>';
        sec.querySelector('#cube-hints').onclick = function () { showHints = !showHints; drawSecrets(); };
      }
      function unlock(id) {
        var x = SECRETS.filter(function (s) { return s.id === id; })[0];
        if (found.indexOf(id) < 0) { found.push(id); PG.store.set('cube-secrets', found); drawSecrets(); toastMsg = 'Secret unlocked: ' + x.name; toastT = 3; sound('win'); buzz([30, 40, 30]); }
        else { toastMsg = x.name; toastT = 1.4; }
      }
      function confetti(n) { for (var i = 0; i < n; i++) fx.bits.push({ x: C.w / 2, y: C.h / 2, vx: (Math.random() - .5) * 520, vy: -Math.random() * 520 - 120, h: Math.random() * 360, life: 1.6 + Math.random() }); }
      function check() {
        var t = seq.slice(-6), n = t.length, j = t.join(' ');
        if (n >= 4 && t.slice(-4).every(function (x) { return x === t[n - 1]; })) { fx.rainbow = 9; unlock('prism'); }
        if (/X\+ Y\+ X- Y-$/.test(j)) { fx.exv = 7; unlock('bang'); }
        if (/X\+ X- X\+ X-$/.test(j)) { fx.wobble = 3; unlock('wobble'); }
        if (/X\+ Y\+ Z\+ X\+ Y\+ Z\+$/.test(j)) { confetti(150); unlock('confetti'); }
      }
      function nextMove(m) { if (twist) return false; startTwist(m.ax, m.layer, m.dir, m); return true; }
      function shuffle() { for (var i = 0; i < 7; i++) queue.push({ ax: Math.floor(Math.random() * 3), layer: Math.random() < .5 ? -1 : 1, dir: Math.random() < .5 ? -1 : 1, quiet: true, fast: true }); }
      drawStats(); drawTrail(); drawSecrets();

      function view() { var w = fx.wobble > 0 ? Math.sin(clock * 38) * .06 * Math.min(1, fx.wobble) : 0; return mmul(rotAxis(0, pitch + w), rotAxis(1, yaw + w * .6)); }
      function project(v, V) { var w = mvec(V, v), f = 7, s = (f / (f + w[2])) * (C.h * 0.135); return { x: C.w / 2 + w[0] * s, y: C.h / 2 + w[1] * s, z: w[2] }; }
      function faceColor(f, ci, outer) {
        if (!outer) return '#0c141c';
        if (fx.rainbow > 0 || fx.party > 0) return 'hsl(' + ((clock * (fx.party > 0 ? 220 : 70) + ci * 47 + f * 61) % 360) + ',82%,56%)';
        return COLORS[f];
      }
      function faces(V) {
        var out = [], sc = 1 + fx.ex;
        cubies.forEach(function (cb, ci) {
          var Rt = cb.R;
          for (var f = 0; f < 6; f++) {
            var n = NORM[f], ax = n[0] ? 0 : n[1] ? 1 : 2, u = (ax + 1) % 3, v = (ax + 2) % 3, sg = n[ax], pts = [];
            [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (q) {
              var l = [0, 0, 0]; l[ax] = sg * .93; l[u] = q[0] * .93; l[v] = q[1] * .93;
              var w = mvec(Rt, l); pts.push(project([w[0] + cb.p[0] * sc, w[1] + cb.p[1] * sc, w[2] + cb.p[2] * sc], V));
            });
            var wn = mvec(V, mvec(Rt, n));
            if (wn[2] > 0.0001) continue;   // back-facing
            var outer = cb.init[ax] === sg;  // was this face on the outside at the start?
            out.push({ pts: pts, z: (pts[0].z + pts[1].z + pts[2].z + pts[3].z) / 4, color: faceColor(f, ci, outer), shade: .62 + .38 * Math.max(0, -wn[2] * .5 + wn[1] * -.5 + .2), ci: ci, f: f, n: mvec(Rt, n), outer: outer });
          }
        });
        out.sort(function (a, b) { return b.z - a.z; });
        return out;
      }
      function inPoly(p, pts) { var c = false; for (var i = 0, j = 3; i < 4; j = i++) { if (((pts[i].y > p.y) !== (pts[j].y > p.y)) && (p.x < (pts[j].x - pts[i].x) * (p.y - pts[i].y) / (pts[j].y - pts[i].y) + pts[i].x)) c = !c; } return c; }
      function startTwist(ax, layer, dir, o) {
        if (twist) return; o = o || {};
        twist = { ax: ax, layer: layer, dir: dir, t: 0, dur: o.fast ? .1 : .22, quiet: !!o.quiet, nolog: !!o.nolog, members: cubies.filter(function (c) { return Math.sign(c.p[ax]) === layer; }) };
      }
      function endTwist() {
        var T = twist, M = rotAxis(T.ax, T.dir * Math.PI / 2);
        T.members.forEach(function (c) { c.p = mvec(M, c.p).map(Math.round); c.R = mmul(M, c.R).map(function (r) { return r.map(function (x) { return Math.round(x); }); }); });
        twist = null; snaps++; PG.store.set('cube-twists', snaps); drawStats(); sound(); buzz(6);
        if (!T.nolog) { hist.push({ ax: T.ax, layer: T.layer, dir: T.dir }); if (hist.length > 300) hist.shift(); }
        if (!T.quiet) { seq.push(tok(T)); if (seq.length > 24) seq.shift(); drawTrail(); check(); }
      }
      PG.drag(C.cv, {
        down: function (e) {
          var p = C.pos(e), V = view(), fs = faces(V); hit = null;
          for (var i = fs.length - 1; i >= 0; i--) { if (inPoly(p, fs[i].pts)) { hit = fs[i]; break; } }
          last = p; orbit = !hit; if (orbit) { vyaw = 0; }
          if (hit) { hit.start = p; hit.t0 = performance.now(); }
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
        up: function () {
          if (hit && !twist && hit.t0 && performance.now() - hit.t0 < 320) {   // a tap, not a drag
            var now = performance.now(); taps = taps.filter(function (x) { return now - x < 2200; }); taps.push(now); sound();
            fx.pop = 1;
            if (taps.length >= 5) { taps = []; fx.neon = !fx.neon; unlock('neon'); }
          }
          last = null; hit = null; orbit = null;
        }
      });
      /* keyboard: U/J = X, I/K = Y, O/L = Z; arrows + B + A feed the cheat code */
      var KONAMI = ['arrowup', 'arrowup', 'arrowdown', 'arrowdown', 'arrowleft', 'arrowright', 'arrowleft', 'arrowright', 'b', 'a'], kpos = 0;
      var offKeys = PG.keyHandler(function (e) {
        var k = (e.key || '').toLowerCase();
        if (k === KONAMI[kpos]) { kpos++; if (kpos === KONAMI.length) { kpos = 0; fx.party = 12; confetti(200); unlock('party'); } e.preventDefault && k.indexOf('arrow') === 0 && e.preventDefault(); } else kpos = k === KONAMI[0] ? 1 : 0;
        var map = { u: [0, 1], j: [0, -1], i: [1, 1], k: [1, -1], o: [2, 1], l: [2, -1] };
        if (map[k] && !twist) startTwist(map[k][0], 1, map[k][1]);
      });
      var L = PG.loop(function (dt) {
        clock += dt;
        if (!orbit) { yaw += vyaw * dt; vyaw += (0.25 - vyaw) * Math.min(1, dt * 0.8); }
        if (!twist && queue.length) nextMove(queue.shift());
        if (twist) { twist.t += dt / twist.dur; if (twist.t >= 1) endTwist(); }
        ['rainbow', 'wobble', 'party', 'pop'].forEach(function (k) { if (fx[k] > 0) fx[k] = Math.max(0, fx[k] - dt); });
        fx.exv += -fx.ex * 55 * dt - fx.exv * 5 * dt; fx.ex += fx.exv * dt; if (Math.abs(fx.ex) < .0008 && Math.abs(fx.exv) < .01) { fx.ex = 0; fx.exv = 0; }
        if (toastT > 0) toastT -= dt;
        var V = view(); ctx.clearRect(0, 0, C.w, C.h);
        var g = ctx.createRadialGradient(C.w / 2, C.h / 2, 10, C.w / 2, C.h / 2, C.h);
        g.addColorStop(0, fx.party > 0 ? 'hsl(' + ((clock * 90) % 360) + ',45%,16%)' : '#101c2a'); g.addColorStop(1, '#04070b'); ctx.fillStyle = g; ctx.fillRect(0, 0, C.w, C.h);
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
          if (fx.neon && f.outer) { ctx.save(); ctx.shadowColor = '#7ffff0'; ctx.shadowBlur = 14; ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(190,255,248,.95)'; ctx.stroke(); ctx.restore(); }
          else { ctx.lineWidth = 1.5; ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.stroke(); }
        });
        for (var i2 = fx.bits.length - 1; i2 >= 0; i2--) {
          var b = fx.bits[i2]; b.life -= dt; b.vy += 900 * dt; b.x += b.vx * dt; b.y += b.vy * dt;
          if (b.life <= 0 || b.y > C.h + 20) { fx.bits.splice(i2, 1); continue; }
          ctx.fillStyle = 'hsla(' + b.h + ',90%,60%,' + Math.min(1, b.life) + ')'; ctx.fillRect(b.x, b.y, 6, 4);
        }
        if (toastT > 0) { ctx.save(); ctx.globalAlpha = Math.min(1, toastT); ctx.font = '700 ' + Math.round(C.h * .06) + 'px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.fillStyle = '#fff'; ctx.shadowColor = '#000'; ctx.shadowBlur = 8; ctx.fillText(toastMsg, C.w / 2, C.h * .1); ctx.restore(); }
      });
      return { destroy: function () { dead = true; queue = []; L.stop(); offKeys(); C.destroy(); try { actx && actx.close(); } catch (e) {} } };
    }
  });

  /* ---------------------------------------------------------------- Liquid Playground (a ring of liquid light on stable springs) */
  PG.register('fidgets', 'liquid', {
    title: 'Liquid Playground', sub: 'Poke, drag and swirl a ring of liquid light. Quick flicks make big waves; a slow touch makes ripples.',
    mount: function (host) {
      var C = PG.canvas(host, 1.6), ctx = C.ctx, N = 180, off = new Float32Array(N), vel = new Float32Array(N), acc = new Float32Array(N);
      var ptr = null, prev = null, held = false, sparks = [], hue = 170, mode = 'rainbow', hasConic = typeof ctx.createConicGradient === 'function', acc_t = 0;
      var bar = PG.el('div', 'pg-bar'); PG.seg(bar, [['rainbow', 'Rainbow'], ['neon', 'Neon cyan'], ['ember', 'Ember']], mode, function (m) { mode = m; });
      PG.button(bar, 'Calm', function () { off.fill(0); vel.fill(0); sparks = []; }); host.appendChild(bar);
      function geo() { var R = Math.min(C.w, C.h) * .3; return { cx: C.w / 2, cy: C.h / 2, R: R }; }
      /* Physics runs on a fixed 1/120 s step so a slow frame can never make the springs blow up (the old version used the frame time directly). */
      var H = 1 / 120, WAVE = 2600, PULL = 14, DAMP = 2.4;
      function push(p, strength, down) {
        var g = geo(), dx = p.x - g.cx, dy = p.y - g.cy, d = Math.hypot(dx, dy), ang = Math.atan2(dy, dx), idx = ((ang / (Math.PI * 2) + 1) % 1) * N, delta = d - g.R;
        if (Math.abs(delta) > g.R * .5) return;
        var dir = delta > 0 ? -1 : 1, w0 = down ? 11 : 7;
        for (var i = -14; i <= 14; i++) { var j = (Math.round(idx) + i + N * 4) % N, w = Math.exp(-i * i / (2 * w0)); vel[j] += dir * strength * w; }
        if (sparks.length < 120 && strength > 40 && Math.random() < .5) { var a2 = ang + (Math.random() - .5) * .25; sparks.push({ x: g.cx + Math.cos(a2) * (g.R + delta * .2), y: g.cy + Math.sin(a2) * (g.R + delta * .2), vx: Math.cos(a2) * -dir * (40 + Math.random() * 60), vy: Math.sin(a2) * -dir * (40 + Math.random() * 60), life: 1 }); }
      }
      PG.drag(C.cv, {
        hover: true,
        down: function (e) { held = true; ptr = prev = C.pos(e); push(ptr, 260, true); },
        move: function (e, down) {
          prev = ptr; ptr = C.pos(e); var sp = prev ? Math.hypot(ptr.x - prev.x, ptr.y - prev.y) : 0;
          push(ptr, Math.min(220, sp * (down ? 9 : 4)), down);
        },
        up: function () { held = false; }
      });
      C.cv.addEventListener('pointerleave', function () { ptr = null; prev = null; });
      function hueAt(i) { return mode === 'rainbow' ? (hue + i / N * 360) % 360 : mode === 'neon' ? 180 + Math.sin(i / N * 6.2832 * 2) * 18 : 18 + Math.sin(i / N * 6.2832 * 2) * 22; }
      function col(i, a) { return 'hsla(' + hueAt(i) + ',100%,' + (mode === 'ember' ? 58 : 62) + '%,' + a + ')'; }
      function step() {
        for (var i = 0; i < N; i++) { var l = off[(i + N - 1) % N], r = off[(i + 1) % N]; acc[i] = (l + r - 2 * off[i]) * WAVE - off[i] * PULL - vel[i] * DAMP; }
        for (i = 0; i < N; i++) { vel[i] += acc[i] * H; off[i] += vel[i] * H; if (!isFinite(off[i]) || !isFinite(vel[i])) { off[i] = 0; vel[i] = 0; } }
      }
      var L = PG.loop(function (dt, t) {
        var g = geo(), lim = g.R * .55; hue = (hue + dt * 22) % 360;
        acc_t += dt; var n = 0; while (acc_t >= H && n < 12) { step(); acc_t -= H; n++; } if (n === 12) acc_t = 0;
        var mean = 0, i; for (i = 0; i < N; i++) { mean += off[i]; } mean /= N;      // keep the ring's overall size steady
        for (i = 0; i < N; i++) { off[i] -= mean * .08; if (off[i] > lim) { off[i] = lim; vel[i] *= .5; } else if (off[i] < -lim) { off[i] = -lim; vel[i] *= .5; } }
        var idle = Math.sin(t * 0.9) * 1.1;
        ctx.globalCompositeOperation = 'source-over'; ctx.clearRect(0, 0, C.w, C.h);
        var bg = ctx.createRadialGradient(g.cx, g.cy, 0, g.cx, g.cy, Math.max(C.w, C.h) * .7); bg.addColorStop(0, '#0b1626'); bg.addColorStop(1, '#03060c'); ctx.fillStyle = bg; ctx.fillRect(0, 0, C.w, C.h);
        var pts = []; for (i = 0; i < N; i++) { var a = i / N * Math.PI * 2, rr = g.R + off[i] + Math.sin(a * 3 + t * 1.3) * idle; pts.push([g.cx + Math.cos(a) * rr, g.cy + Math.sin(a) * rr]); }
        function ring() { ctx.beginPath(); var m0 = [(pts[N - 1][0] + pts[0][0]) / 2, (pts[N - 1][1] + pts[0][1]) / 2]; ctx.moveTo(m0[0], m0[1]); for (var k = 0; k < N; k++) { var p1 = pts[k], p2 = pts[(k + 1) % N]; ctx.quadraticCurveTo(p1[0], p1[1], (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2); } ctx.closePath(); }
        // liquid body
        ring(); var fill = ctx.createRadialGradient(g.cx, g.cy, g.R * .1, g.cx, g.cy, g.R * 1.1); fill.addColorStop(0, col(0, 0)); fill.addColorStop(1, col(N / 3, .13)); ctx.fillStyle = fill; ctx.fill();
        ctx.globalCompositeOperation = 'lighter'; ctx.lineJoin = 'round';
        var stroke = null;
        if (hasConic && mode === 'rainbow') { stroke = ctx.createConicGradient(0, g.cx, g.cy); for (i = 0; i <= 12; i++) stroke.addColorStop(i / 12, 'hsl(' + ((hue + i / 12 * 360) % 360) + ',100%,62%)'); }
        else if (hasConic) { stroke = ctx.createConicGradient(0, g.cx, g.cy); for (i = 0; i <= 12; i++) stroke.addColorStop(i / 12, 'hsl(' + hueAt((i % 12) / 12 * N) + ',100%,' + (mode === 'ember' ? 58 : 62) + '%)'); }
        [[16, .07], [8, .14], [4, .5], [1.8, 1]].forEach(function (s) { ctx.lineWidth = s[0]; ctx.globalAlpha = s[1]; ctx.strokeStyle = stroke || 'hsl(' + hueAt(0) + ',100%,62%)'; ring(); ctx.stroke(); });
        ctx.globalAlpha = 1;
        for (i = sparks.length - 1; i >= 0; i--) { var s = sparks[i]; s.life -= dt * 1.5; s.x += s.vx * dt; s.y += s.vy * dt; s.vy += 60 * dt; if (s.life <= 0) { sparks.splice(i, 1); continue; } ctx.fillStyle = col(i * 5, s.life * .9); ctx.beginPath(); ctx.arc(s.x, s.y, 1 + s.life * 1.8, 0, 6.2832); ctx.fill(); }
        ctx.globalCompositeOperation = 'source-over';
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
