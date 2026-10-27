/* Daily Puzzles: Infinite Maze, Daily chess puzzle, Chart analyzer. Same puzzle for everyone on the same date (seeded by the local date). */
(function () {
  var PG = window.PG, CH = PG.chess;
  var PUZZLES = [["rnb1kbr1/p2ppppp/1p6/1P6/2p1n1P1/7B/P1PP3P/RNBQ1qNK b q - 0 1",1],["rnb5/ppppp3/3k4/8/3PP1R1/5Q2/PPP2KP1/2R2BN1 w - - 0 1",2],["3k1bnr/1bpp1Qp1/2n1p3/8/4P3/8/r1PNKPPP/R5R1 w - - 0 1",1],["r1b5/ppppp3/n2k4/8/3PP1R1/2P2Q2/PP3KP1/2R2BN1 w - - 0 1",2],["3k1bnr/1bpp2Q1/2n1p3/8/4P3/8/2rNKPPP/R5R1 w - - 0 1",1],["6nk/3Q4/6p1/8/2Pp4/5P2/R3P1Pr/1N2KBNR w K - 0 1",2],["r3kbnr/4q2p/5pp1/1Pnp2N1/6b1/1PP3P1/3PPP1P/1NBQKB1R b Kkq - 0 1",1],["8/5k2/8/4Q3/2P5/5P2/3RP1PR/1N1K1BN1 w - - 0 1",2],["1rb1k1nr/p2R3p/n1p5/1p2P3/8/PQP4q/P2PP3/R1B1KBN1 w Qk - 0 1",1],["1nb2Bnr/1p1ppk2/7p/8/3P4/r5q1/2P1PPPP/R2QKB1R b KQ - 0 1",2],["1rb1k1nr/p6R/n1p5/1p2P3/8/P1Pq4/P2PP3/R1BQKBN1 b Qk - 0 1",1],["rnb3nr/p1p2ppp/4k3/1p4B1/1P1Q4/N5P1/P1P1PK1P/R4B1R w - - 0 1",2],["rnbq1knr/p1pQ1ppp/1p6/8/2P5/N5PP/PP2PP2/R1B1KBNR w KQ - 0 1",1],["rnb1kbQ1/3pppp1/p1p5/8/3P4/N7/PPP1P1P1/1RB1KBqR w Kq - 0 1",2],["rnbqk3/pp3p2/2pQ4/2P2p2/8/6P1/PPP1PK2/RNB2B1R w q - 0 1",1],["2bqkbnr/p2np1p1/2pp3p/1P3p2/PP2P2P/8/2RPBPP1/1NB1K1NR w Kk - 0 1",2],["3n4/1p2k1bN/6pp/1bp5/2P1Pq1P/8/r5P1/5BKR b - - 0 1",1],["rn2kb2/1pp2pQR/p7/8/P7/1PN1p1P1/R1PP4/2B1K1N1 w q - 0 1",2],["3n4/1p2k1bN/6pp/1Pp5/4Pq1P/8/1r4P1/5BKR b - - 0 1",1],["1rb5/1p1kp3/1Q4p1/1P3pP1/RP6/5N2/4PP2/1Nb1KB1R w - - 0 1",2],["1n2kb1r/rpQ1pppp/p4n2/8/1P6/3P4/P1P1PP1P/RNB1KbNR w KQk - 0 1",1],["1rb5/1p1kp3/1Q4p1/1P3pb1/RP6/5N2/4PP1R/1N2KB2 w - - 0 1",2],["1n2kb1r/r2p3p/b6n/p3pp2/2p4q/5PP1/PPPPP3/1RBQKBN1 b k - 0 1",1],["Bnbqkbr1/p1ppp2n/5p2/3Q3p/P4p2/6PP/3PP3/RNB2KNR w - - 0 1",2],["1n2kb1r/r2p3p/b7/p3pp2/2p2Pnq/6P1/PPPPP3/1RBQKBN1 b k - 0 1",1],["Bnbqkbr1/2ppp2n/5p2/p2Q3p/P4P2/7P/3PP3/RNB2KNR w - - 0 1",2],["1nb1kb1r/r2p3p/8/p3Pp2/2p3nq/6P1/PPPPP3/1RBQKBN1 b k - 0 1",1],["r2qkbRr/ppp2p1p/2n1p3/8/1P1p1P2/N7/P1PPP3/R1BQKBN1 b Qkq - 0 1",2],["1nb1kb1r/r2p3p/8/p3Pp2/2p4q/4n1P1/PPPPP1B1/1RBQK1N1 b k - 0 1",1],["2b1k2r/2Rp1pbp/3q3n/1pB1p3/1P5p/5P2/4N1P1/1N1QKB2 w k - 0 1",2],["1n2kb1r/r2p3p/b3P3/p4p2/2p4q/4n1P1/PPPPP1B1/1RBQK1N1 b k - 0 1",1],["2b2k1r/7p/1Bp1p2P/2P2p2/8/p1PQ1N2/P3P1K1/R4R2 w - - 0 1",2],["7k/3Q4/6pn/8/2Pp4/5P2/R3P1PR/1N2KBN1 w - - 0 1",1],["2b3kr/7p/1Bp1p2P/2P2p2/8/p1PQ1N2/P3PK2/R4R2 w - - 0 1",2],["6k1/3Q4/6pR/8/2Pp4/5P2/R3P1P1/1N2KBN1 w - - 0 1",1],["rnbqk1nr/pp1p3p/2p3p1/4Pp2/1b3P2/P6P/RPPNP1P1/2BQKBNR b Kkq - 0 1",2],["6k1/3Q4/6p1/8/2P5/3p1P1R/R3P1P1/1N2KBN1 w - - 0 1",1],["rnbq2nr/pp1p1k1p/2pP2p1/5p2/P4P2/7P/RPPNP1P1/2BQKBNR b K - 0 1",2],["1n2kbnr/2p1pp1p/6p1/1N4P1/7P/3R4/RPPP1q2/2BK2N1 b k - 0 1",1],["2r5/pppPkpp1/5p2/8/P1N5/8/RPK1BP2/2BQ4 w - - 0 1",2],["1n2kbnr/2p1pp1p/6p1/1N4P1/7q/3R4/RPPP4/2BK2N1 w k - 0 1",1],["r2k3r/p6p/p7/4pp2/Q7/bR4P1/P2BK1R1/5BN1 w - - 0 1",2],["rnbqkb2/1p1pn3/2p1p1p1/p7/P1PPN3/1R6/1P3P2/R1Q1KBN1 w Qq - 0 1",1],["rn1qkbnr/p5P1/bppp3p/5p2/8/P6P/RPPP2P1/1NBQKBNR b Kkq - 0 1",2],["rnb1kb1r/pp1p1ppp/2p1p3/8/P4Pnq/N5P1/RPPPP3/2BQKBNR b Kkq - 0 1",1],["rn1qk1nr/p5b1/bppp3p/5p2/1P6/P6P/R1PP2P1/1NBQKBNR b Kkq - 0 1",2],["2b3qr/2p1p2Q/3pk3/1p6/1p2P3/2NP1N2/R1P1P1PP/4KB1R w K - 0 1",1],["2kr1bnr/p1p5/5p1p/4p2q/1Pp5/P3K3/5P2/1R4N1 b - - 0 1",2],["2B5/n7/4pkp1/p5N1/6P1/P2R4/2q2r2/4K2R b - - 0 1",1],["2k2bnr/p1p5/5p1p/4p2q/PPpr4/4K3/5P2/1R4N1 b - - 0 1",2],["2bqk1n1/r2n4/2p5/3ppp2/8/B3P2N/P2NBPPR/R3K3 w Q - 0 1",1],["2k2bnr/p1pr4/5p1p/4p2q/PPp5/4K3/5P2/4R1N1 b - - 0 1",2],["rNbq2nr/8/pp2p2Q/5k1P/5p2/8/P1P1PPP1/R1B1KBR1 w Q - 0 1",1],["Q7/3Pk1p1/B7/1p1ppP2/8/2P5/1P1P1PPP/2B1K1NR w K - 0 1",2],["2b2bnr/rpp1p1pp/p4pk1/1B6/1P2P3/N7/P1P2KPP/R1B3NR w - - 0 1",1],["Q7/3Pk1p1/8/1B2pP2/3p4/2P5/1P1P1PPP/2B1K1NR w K - 0 1",2],["rnb5/1pppk2r/4p2Q/p7/P1P5/5RK1/4P3/RNB3N1 w - - 0 1",1],["3q1bnr/2p1k1pp/4B3/4n3/8/r1P3PN/PP2KP1P/R1B4R b - - 0 1",2],["rn2k1n1/p1p2p2/8/1p1Q2Bp/PPP5/2N2P2/8/R4K2 w - - 0 1",1],["3q1bnr/2p3pp/4k3/4n3/6P1/r1P4N/PP2KP1P/R1B4R b - - 0 1",2],["rn2k1n1/p1p2p2/8/3Q2Bp/PPp5/5P2/8/RN3K2 w - - 0 1",1],["5k2/3P2p1/4p3/p4p2/8/PPP1BN2/6K1/R7 w - - 0 1",2],["rnbqk1nr/pppp1p1p/8/8/P4b2/8/1PPPP1P1/RNBQKBN1 b kq - 0 1",1],["2bqk2r/r1pp1ppp/1R6/4p2P/1P6/7n/2PPP1B1/1NBQK1N1 b k - 0 1",2],["rnbqk1nr/1ppp1p1p/p7/8/P4b2/8/RPPPP1P1/1NBQKBN1 b kq - 0 1",1],["4k3/8/3B4/4P3/5N2/7p/PP2PK1P/6R1 w - - 0 1",2],["rnb1k1nr/pppp1pp1/4p3/7R/8/N1P2qP1/PP1PP3/1RBQKB2 b kq - 0 1",1],["2bqk1nr/1p6/n1p1p1p1/4Pp1p/1P6/N1N5/1b3PPP/5KR1 b k - 0 1",2],["2r1kbnr/p1p2ppR/3pp3/1pn3q1/P4P2/1P4P1/2PPP3/RNBQKBN1 b Qk - 0 1",1],["rn3b1r/p3pk2/3q3R/1Bp5/8/N4P2/PPP2P2/1RK5 b - - 0 1",2],["2r1kbnr/p1p2ppR/3pp3/1pn1q3/P4P2/1PP3P1/3PP3/RNBQKBN1 b Qk - 0 1",1],["rn3b1r/p3pk2/3qR3/1B6/2p5/N4P2/PPP2P2/1RK5 b - - 0 1",2]];   // [fen, mateIn] -- verified by the engine: exactly one first move forces mate

  /* ---------------------------------------------------------------- Infinite Maze */
  PG.register('puzzles', 'maze', {
    title: 'Infinite Maze', sub: 'Find the exit, then do it again in a bigger maze. Arrow keys, WASD, swipe or the pad. Daily challenge: same maze for everyone today.',
    mount: function (host) {
      var diff = PG.store.get('maze-diff', 'medium'), mode = 'run', level = 1, W, H, cells, px, py, goal, trail, ex, started, t0, elapsed = 0, dead = false, seen, total = 0, won = false, anim = null;
      var CFG = { easy: { start: 7, grow: 1, fog: 0 }, medium: { start: 9, grow: 2, fog: 0 }, hard: { start: 11, grow: 2, fog: 4.5 } };
      var top = PG.el('div', 'pg-bar'); host.appendChild(top);
      var sLevel = PG.el('span', 'pg-stat'), sTime = PG.el('span', 'pg-stat'), sBest = PG.el('span', 'pg-stat'); top.appendChild(sLevel); top.appendChild(sTime); top.appendChild(sBest);
      var bar2 = PG.el('div', 'pg-bar'); host.appendChild(bar2);
      PG.seg(bar2, [['easy', 'Easy'], ['medium', 'Medium'], ['hard', 'Hard (fog)']], diff, function (v) { diff = v; PG.store.set('maze-diff', v); startRun(); });
      PG.button(bar2, 'Daily challenge', function () { startDaily(); }); PG.button(bar2, 'New run', function () { startRun(); }, true);
      var C = PG.canvas(host, 1); C.wrap.style.maxWidth = '560px'; C.wrap.style.margin = '10px auto 0';
      var msg = PG.el('div', 'pg-msg'); host.appendChild(msg);
      var pad = PG.el('div', 'tt-pad'); [['←', 3], ['↑', 0], ['↓', 2], ['→', 1]].forEach(function (b) { var x = PG.el('button', '', b[0]); x.type = 'button'; x.style.cssText = 'color:var(--teal);border-color:var(--border-light);background:var(--panel-3)'; x.onpointerdown = function (e) { e.preventDefault(); step(b[1]); }; pad.appendChild(x); }); host.appendChild(pad);
      function bestKey() { return 'maze-best-' + diff; }
      function gen(w, h, rnd) {
        var c = []; for (var i = 0; i < w * h; i++) c.push(15);   // bits: 1 up, 2 right, 4 down, 8 left = walls present
        var vis = new Uint8Array(w * h), st = [0]; vis[0] = 1; var D = [[0, -1, 1, 4], [1, 0, 2, 8], [0, 1, 4, 1], [-1, 0, 8, 2]];
        while (st.length) { var cur = st[st.length - 1], x = cur % w, y = (cur / w) | 0, opts = []; D.forEach(function (d) { var nx = x + d[0], ny = y + d[1]; if (nx >= 0 && ny >= 0 && nx < w && ny < h && !vis[ny * w + nx]) opts.push(d); });
          if (!opts.length) { st.pop(); continue; } var d = opts[Math.floor(rnd() * opts.length)], n = (y + d[1]) * w + x + d[0]; c[cur] &= ~d[2]; c[n] &= ~d[3]; vis[n] = 1; st.push(n); }
        return c;
      }
      function farthest(c, w, h) { var dist = new Int32Array(w * h).fill(-1), q = [0], best = 0; dist[0] = 0; var D = [[0, -1, 1], [1, 0, 2], [0, 1, 4], [-1, 0, 8]]; while (q.length) { var cur = q.shift(), x = cur % w, y = (cur / w) | 0; D.forEach(function (d) { if (c[cur] & d[2]) return; var n = (y + d[1]) * w + x + d[0]; if (dist[n] < 0) { dist[n] = dist[cur] + 1; q.push(n); if (dist[n] >= dist[best]) best = n; } }); } return best; }
      function setup(size, seed) {
        W = H = size; var rnd = seed ? PG.rng(seed) : Math.random; cells = gen(W, H, rnd); px = 0; py = 0; goal = farthest(cells, W, H); trail = [0]; seen = new Uint8Array(W * H); seen[0] = 1; started = false; won = false; anim = null; layout();
      }
      function sizeFor(l) { var c = CFG[diff]; return Math.min(31, c.start + (l - 1) * c.grow); }
      function startRun() { mode = 'run'; level = 1; total = 0; elapsed = 0; setup(sizeFor(1)); msg.textContent = 'Reach the glowing exit. The timer starts on your first move.'; hud(); }
      function startDaily() { mode = 'daily'; level = 1; elapsed = 0; setup(19, PG.dailySeed() * 7 + 3); msg.textContent = 'Daily challenge ' + PG.today() + ': one 19x19 maze, same for everyone. Best time is saved.'; hud(); }
      var cs, ox, oy;
      function layout() { cs = Math.floor(Math.min(C.w, C.h) / (W + 0.4)); ox = Math.floor((C.w - cs * W) / 2); oy = Math.floor((C.h - cs * H) / 2); }
      C.onresize = layout;
      function hud() {
        sLevel.innerHTML = mode === 'daily' ? 'Daily <b>' + PG.today() + '</b>' : 'Level <b>' + level + '</b> (' + W + 'x' + W + ')'; sTime.innerHTML = 'Time <b>' + PG.fmtTime(elapsed) + '</b>';
        var b = mode === 'daily' ? PG.store.get('maze-daily-' + PG.today(), null) : PG.store.get(bestKey(), 0); sBest.innerHTML = mode === 'daily' ? 'Today\'s best <b>' + (b ? PG.fmtTime(b) : '-') + '</b>' : 'Best level <b>' + b + '</b>';
      }
      function step(d) {
        if (won) return; var DX = [0, 1, 0, -1], DY = [-1, 0, 1, 0], bit = [1, 2, 4, 8][d], i = py * W + px; if (cells[i] & bit) return;
        if (!started) { started = true; t0 = performance.now() - elapsed * 1000; }
        var nx = px + DX[d], ny = py + DY[d]; anim = { fx: px, fy: py, t: 0 }; px = nx; py = ny; seen[py * W + px] = 1;
        var ti = trail.lastIndexOf(py * W + px); if (ti >= 0) trail.length = ti + 1; else trail.push(py * W + px);
        if (py * W + px === goal) finish();
      }
      function finish() {
        won = true;
        if (mode === 'daily') { var k = 'maze-daily-' + PG.today(), b = PG.store.get(k, null); if (!b || elapsed < b) PG.store.set(k, Math.round(elapsed * 10) / 10); msg.textContent = 'Daily maze solved in ' + PG.fmtTime(elapsed) + '. Come back tomorrow for a new one.'; hud(); return; }
        total += elapsed; if (level > PG.store.get(bestKey(), 0)) PG.store.set(bestKey(), level); msg.textContent = 'Level ' + level + ' cleared in ' + PG.fmtTime(elapsed) + '. Next maze is bigger...'; hud();
        setTimeout(function () { if (dead) return; level++; elapsed = 0; setup(sizeFor(level)); msg.textContent = 'Level ' + level + '. ' + (CFG[diff].fog ? 'You can only see nearby cells.' : 'Find the exit.'); hud(); }, 900);
      }
      var off = PG.keyHandler(function (e) { var m = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3, w: 0, d: 1, s: 2, a: 3, W: 0, D: 1, S: 2, A: 3 }[e.key]; if (m !== undefined) { e.preventDefault(); step(m); } });
      var sx, sy; C.cv.addEventListener('pointerdown', function (e) { sx = e.clientX; sy = e.clientY; });
      C.cv.addEventListener('pointerup', function (e) { var dx = e.clientX - sx, dy = e.clientY - sy; if (Math.max(Math.abs(dx), Math.abs(dy)) < 18) return; var d = Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0); step(d); });
      var held = null, rep = 0; C.cv.addEventListener('pointermove', function (e) { if (e.buttons !== 1 || sx == null) return; var dx = e.clientX - sx, dy = e.clientY - sy, th = Math.max(14, cs * .7); if (Math.max(Math.abs(dx), Math.abs(dy)) > th) { step(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0)); sx = e.clientX; sy = e.clientY; } });
      var L = PG.loop(function (dt, t) {
        if (started && !won) { elapsed = (performance.now() - t0) / 1000; } var ctx = C.ctx; ctx.clearRect(0, 0, C.w, C.h); ctx.fillStyle = '#060a10'; ctx.fillRect(0, 0, C.w, C.h);
        var fog = mode === 'daily' ? 0 : CFG[diff].fog; if (anim) { anim.t = Math.min(1, anim.t + dt * 14); if (anim.t >= 1) anim = null; }
        var ax = anim ? anim.fx + (px - anim.fx) * anim.t : px, ay = anim ? anim.fy + (py - anim.fy) * anim.t : py;
        ctx.save(); ctx.lineCap = 'round'; ctx.lineWidth = Math.max(1.5, cs * .1);
        for (var y = 0; y < H; y++) for (var x = 0; x < W; x++) {
          var i = y * W + x, vis = 1; if (fog) { var d = Math.hypot(x - ax, y - ay); vis = d <= fog ? 1 - Math.max(0, d - fog + 1.5) / 1.5 * .8 : (seen[i] ? .22 : 0); } if (vis <= 0) continue;
          ctx.strokeStyle = 'rgba(120,200,255,' + (.85 * vis) + ')'; var X = ox + x * cs, Y = oy + y * cs, c = cells[i];
          ctx.beginPath(); if (c & 1) { ctx.moveTo(X, Y); ctx.lineTo(X + cs, Y); } if (c & 2) { ctx.moveTo(X + cs, Y); ctx.lineTo(X + cs, Y + cs); } if (c & 4) { ctx.moveTo(X, Y + cs); ctx.lineTo(X + cs, Y + cs); } if (c & 8) { ctx.moveTo(X, Y); ctx.lineTo(X, Y + cs); } ctx.stroke();
        }
        ctx.restore();
        ctx.strokeStyle = 'rgba(53,224,176,.35)'; ctx.lineWidth = Math.max(2, cs * .22); ctx.lineJoin = 'round'; ctx.beginPath(); trail.forEach(function (n, k) { var X = ox + (n % W + .5) * cs, Y = oy + (((n / W) | 0) + .5) * cs; if (k) ctx.lineTo(X, Y); else ctx.moveTo(X, Y); }); ctx.stroke();
        var gx = ox + (goal % W + .5) * cs, gy = oy + (((goal / W) | 0) + .5) * cs, pulse = 1 + Math.sin(t * 4) * .18, gv = fog ? Math.max(.25, 1 - Math.max(0, Math.hypot(goal % W - ax, ((goal / W) | 0) - ay) - fog) / 6) : 1;
        ctx.fillStyle = 'rgba(245,184,61,' + gv + ')'; ctx.shadowColor = '#f5b83d'; ctx.shadowBlur = 14 * gv; ctx.beginPath(); ctx.arc(gx, gy, cs * .3 * pulse, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0;
        ctx.fillStyle = '#35e0b0'; ctx.shadowColor = '#35e0b0'; ctx.shadowBlur = 12; ctx.beginPath(); ctx.arc(ox + (ax + .5) * cs, oy + (ay + .5) * cs, cs * .26, 0, 6.283); ctx.fill(); ctx.shadowBlur = 0;
        if (((t * 4) | 0) % 2 === 0 || true) sTime.innerHTML = 'Time <b>' + PG.fmtTime(elapsed) + '</b>';
      });
      startRun();
      return { destroy: function () { dead = true; L.stop(); off(); C.destroy(); } };
    }
  });

  /* ---------------------------------------------------------------- Daily chess puzzle */
  PG.register('puzzles', 'chess', {
    title: 'Daily chess puzzle', sub: 'Mate in one or two. A new puzzle every day, plus extra practice whenever you like.',
    mount: function (host) {
      var epoch = Math.floor(new Date(new Date().getFullYear(), new Date().getMonth(), new Date().getDate()).getTime() / 864e5), idx = ((epoch * 7) % PUZZLES.length + PUZZLES.length) % PUZZLES.length, daily = true, state, start, last = null, solved = false, step = 1, hints = 0, dead = false, firstTry = true;
      var row = PG.el('div', 'pg-row'), left = PG.el('div', 'pg-grow'), side = PG.el('div', 'pg-side'); row.appendChild(left); row.appendChild(side); host.appendChild(row);
      var info = PG.el('div', 'pg-bar'); left.appendChild(info);
      var ui = PG.mountChessBoard(left, { getState: function () { return state; }, flipped: function () { return start.t === 'b'; }, lastMove: function () { return last; }, canMove: function () { return !solved && state.t === start.t; }, selectable: function () { return true; }, onMove: try_ });
      var msg = PG.el('div', 'pg-msg'); left.appendChild(msg);
      var bar = PG.el('div', 'pg-bar'); left.appendChild(bar);
      PG.button(bar, 'Hint', function () { hint(); }); PG.button(bar, 'Show solution', function () { solution(); }); PG.button(bar, 'Reset', function () { load(idx, daily); }); PG.button(bar, 'Another puzzle', function () { idx = (idx + 1 + Math.floor(Math.random() * (PUZZLES.length - 1))) % PUZZLES.length; load(idx, false); }, true);
      var rec = PG.el('div', 'pg-side'); side.appendChild(rec);
      function load(i, isDaily) {
        idx = i; daily = isDaily; var p = PUZZLES[i]; state = CH.fromFEN(p[0]); start = state; last = null; solved = false; step = 1; hints = 0; firstTry = true; ui.clear(); ui.render();
        var n = p[1]; info.innerHTML = '<span class="pg-stat">' + (isDaily ? 'Today\'s puzzle' : 'Practice') + '</span><span class="pg-stat"><b>' + (state.t === 'w' ? 'White' : 'Black') + '</b> to play, mate in <b>' + n + '</b></span>';
        msg.textContent = n === 1 ? 'Find the checkmate.' : 'Find the first move that forces checkmate in two.'; stats();
      }
      function stats() { var s = PG.store.get('chess-puzzle-stats', { solved: 0, firstTry: 0, days: {} }); rec.innerHTML = '<p><b>' + s.solved + '</b> puzzles solved, <b>' + s.firstTry + '</b> on the first try.</p><p>' + (s.days[PG.today()] ? 'Today\'s puzzle: <b>done</b>.' : 'Today\'s puzzle is waiting.') + '</p><p class="pg-hint">No streaks and no reminders: it is just here when you want it. Positions come from engine-checked games, so each has exactly one first move that works.</p>'; }
      function done() {
        solved = true; var s = PG.store.get('chess-puzzle-stats', { solved: 0, firstTry: 0, days: {} }); s.solved++; if (firstTry) s.firstTry++; if (daily) s.days[PG.today()] = 1; PG.store.set('chess-puzzle-stats', s);
        msg.textContent = 'Checkmate. ' + (firstTry ? 'Solved on the first try.' : 'Solved.'); stats();
      }
      function try_(m) {
        var n = PUZZLES[idx][1] - (step - 1), good = CH.mateMoves(state, n), ok = good.some(function (g) { return g.from === m.from && g.to === m.to && (g.promo || '') === (m.promo || ''); });
        if (!ok) { firstTry = false; msg.textContent = 'Not quite. That does not force mate. Try again.'; ui.render(); return; }
        state = CH.apply(state, m); last = m; ui.clear(); ui.render();
        var st = CH.status(state); if (st.over && st.reason === 'checkmate') { done(); return; }
        step++; msg.textContent = 'Good. Black replies...'.replace('Black', state.t === 'b' ? 'Black' : 'White');
        setTimeout(function () { if (dead) return; var rs = CH.legal(state), rm = rs[0], deep = -1; rs.forEach(function (r) { var d = CH.mateMoves(CH.apply(state, r), 1).length; if (deep < 0 || d < deep) { deep = d; rm = r; } }); state = CH.apply(state, rm); last = rm; ui.render(); msg.textContent = 'Now deliver checkmate.'; }, 550);
      }
      function hint() { var n = PUZZLES[idx][1] - (step - 1), g = CH.mateMoves(state, n)[0]; if (!g) return; hints++; firstTry = false; msg.textContent = hints === 1 ? 'Hint: look at the ' + ({ p: 'pawn', n: 'knight', b: 'bishop', r: 'rook', q: 'queen', k: 'king' })[state.b[g.from].toLowerCase()] + ' on ' + CH.sqName(g.from) + '.' : 'Move ' + CH.sqName(g.from) + ' to ' + CH.sqName(g.to) + '.'; }
      function solution() { var n = PUZZLES[idx][1] - (step - 1), g = CH.mateMoves(state, n)[0]; if (!g) return; firstTry = false; msg.textContent = 'Solution: ' + CH.san(state, g) + '.'; }
      side.insertAdjacentHTML('afterbegin', '<p>Tap a piece, then its destination square.</p>');
      load(idx, true);
      return { destroy: function () { dead = true; } };
    }
  });

  /* ---------------------------------------------------------------- Chart analyzer (synthetic charts, no market data) */
  PG.register('puzzles', 'chart', {
    title: 'Chart analyzer', sub: 'Read an invented price chart and call the next 10 candles: Buy, Sell or Skip. Charts are generated on your device; nothing here is real market data.',
    mount: function (host) {
      var diff = PG.store.get('chart-diff', 'normal'), mode = 'free', seed, data, hidden, reveal = 0, answered = false, dead = false, setN = 0, setScore = 0, dailySet = 0, rnd;
      var top = PG.el('div', 'pg-bar'); host.appendChild(top);
      var sScore = PG.el('span', 'pg-stat'), sAcc = PG.el('span', 'pg-stat'); top.appendChild(sScore); top.appendChild(sAcc);
      PG.seg(top, [['easy', 'Clear trends'], ['normal', 'Normal'], ['hard', 'Choppy']], diff, function (v) { diff = v; PG.store.set('chart-diff', v); next(); });
      PG.button(top, 'Daily set of 5', function () { mode = 'daily'; dailySet = 0; setN = 0; setScore = 0; next(); });
      var C = PG.canvas(host, 1.9); var bar = PG.el('div', 'pg-bar'); host.appendChild(bar);
      var bBuy = PG.button(bar, 'Buy (up)', function () { answer('buy'); }, true), bSell = PG.button(bar, 'Sell (down)', function () { answer('sell'); }), bSkip = PG.button(bar, 'Skip', function () { answer('skip'); }), bNext = PG.button(bar, 'Next chart', function () { next(); }); bNext.style.display = 'none';
      var msg = PG.el('div', 'pg-msg'); host.appendChild(msg); host.appendChild(PG.el('p', 'pg-hint', 'Charts mix trends, pullbacks and noise. Even skilled traders are wrong a lot, so treat a 55-60% hit rate as good. Stats stay on this device.'));
      function st() { return PG.store.get('chart-stats', { right: 0, wrong: 0, skip: 0 }); }
      function stats() { var s = st(), n = s.right + s.wrong; sScore.innerHTML = mode === 'daily' ? 'Daily set <b>' + Math.min(setN + (answered ? 0 : 1), 5) + '/5</b> &middot; ' + setScore + ' correct' : 'Called <b>' + n + '</b>'; sAcc.innerHTML = 'Hit rate <b>' + (n ? Math.round(100 * s.right / n) + '%' : '-') + '</b>'; }
      function gen() {
        var N = 70, M = 10, p = 100, out = [], drift = (rnd() - .5) * (diff === 'easy' ? .9 : diff === 'normal' ? .5 : .22), vol = diff === 'hard' ? 1.25 : 1, turn = 20 + Math.floor(rnd() * 30), reg = drift;
        for (var i = 0; i < N + M; i++) {
          if (i === turn) reg = (i < N ? -drift * (diff === 'easy' ? .2 : 1) : drift) * (rnd() < .5 ? 1 : .6); if (i === N) reg = drift * (diff === 'hard' ? (rnd() < .5 ? 1 : -1) : 1) * (diff === 'easy' ? 1.3 : .8);
          var o = p, ch = reg * .35 + (rnd() - .5) * 2.2 * vol, c = o + ch, hi = Math.max(o, c) + rnd() * 1.1 * vol, lo = Math.min(o, c) - rnd() * 1.1 * vol; out.push({ o: o, h: hi, l: lo, c: c }); p = c;
        }
        return { shown: out.slice(0, N), future: out.slice(N) };
      }
      function next() {
        seed = mode === 'daily' ? PG.dailySeed() * 31 + dailySet * 977 : Math.floor(Math.random() * 1e9); rnd = PG.rng(seed); var g = gen(); data = g.shown; hidden = g.future; reveal = 0; answered = false;
        bBuy.disabled = bSell.disabled = bSkip.disabled = false; bNext.style.display = 'none'; msg.textContent = 'What happens over the next 10 candles?'; stats(); draw();
      }
      function answer(a) {
        if (answered) return; answered = true; var up = hidden[hidden.length - 1].c > data[data.length - 1].c, s = st(), ok = (a === 'buy' && up) || (a === 'sell' && !up);
        if (a === 'skip') s.skip++; else if (ok) s.right++; else s.wrong++; PG.store.set('chart-stats', s);
        var mv = hidden[hidden.length - 1].c - data[data.length - 1].c; msg.textContent = (a === 'skip' ? 'Skipped. ' : ok ? 'Correct. ' : 'Not this time. ') + 'Price ' + (up ? 'rose' : 'fell') + ' ' + Math.abs(mv).toFixed(1) + ' points over the next 10 candles.';
        if (mode === 'daily') { setN++; if (ok) setScore++; if (setN >= 5) { msg.textContent += ' Daily set finished: ' + setScore + ' of 5 correct.'; mode = 'free'; } else dailySet++; }
        bBuy.disabled = bSell.disabled = bSkip.disabled = true; bNext.style.display = ''; bNext.textContent = 'Next chart'; revealAnim(); stats();
      }
      function revealAnim() { var i = 0; (function tick() { if (dead) return; reveal = ++i; draw(); if (i < hidden.length) setTimeout(tick, 110); })(); }
      function draw() {
        var ctx = C.ctx, all = data.concat(hidden.slice(0, reveal)), lo = Infinity, hi = -Infinity; ctx.fillStyle = '#05080d'; ctx.fillRect(0, 0, C.w, C.h);
        (answered ? data.concat(hidden) : data).forEach(function (b) { lo = Math.min(lo, b.l); hi = Math.max(hi, b.h); }); var pad = (hi - lo) * .08; lo -= pad; hi += pad;
        var m = { l: 10, r: 56, t: 10, b: 14 }, n = data.length + hidden.length, bw = (C.w - m.l - m.r) / n, Y = function (v) { return m.t + (hi - v) / (hi - lo) * (C.h - m.t - m.b); };
        ctx.strokeStyle = 'rgba(255,255,255,.06)'; ctx.fillStyle = 'rgba(255,255,255,.4)'; ctx.font = '10px ui-monospace,monospace'; ctx.lineWidth = 1;
        for (var g = 0; g <= 4; g++) { var v = lo + (hi - lo) * g / 4, y = Y(v); ctx.beginPath(); ctx.moveTo(m.l, y); ctx.lineTo(C.w - m.r + 6, y); ctx.stroke(); ctx.fillText(v.toFixed(1), C.w - m.r + 10, y + 3); }
        all.forEach(function (b, i) { var x = m.l + i * bw + bw / 2, up = b.c >= b.o, col = up ? '#35d878' : '#ff5d73'; ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(x, Y(b.h)); ctx.lineTo(x, Y(b.l)); ctx.stroke(); var y1 = Y(Math.max(b.o, b.c)), y2 = Y(Math.min(b.o, b.c)); ctx.fillRect(x - bw * .34, y1, bw * .68, Math.max(1.5, y2 - y1)); });
        if (!answered) { var lx = m.l + data.length * bw; ctx.strokeStyle = 'rgba(255,255,255,.25)'; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(lx, m.t); ctx.lineTo(lx, C.h - m.b); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = 'rgba(255,255,255,.4)'; ctx.fillText('next 10 candles?', lx + 6, m.t + 12); }
      }
      C.onresize = draw; next();
      return { destroy: function () { dead = true; C.destroy(); } };
    }
  });
})();
