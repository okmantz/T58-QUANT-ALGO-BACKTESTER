/* Arcade: Chess vs computer, 2048 Quant Edition, Tetris (terminal). */
(function () {
  var PG = window.PG, CH = PG.chess;

  /* ---------------------------------------------------------------- Chess */
  PG.register('arcade', 'chess', {
    title: 'Chess vs computer', sub: 'Full rules: castling, en passant, promotion, draws. Pick a level and a colour.',
    mount: function (host) {
      var state, hist, keys, last = null, level = PG.store.get('chess-level', 3), human = PG.store.get('chess-side', 'w'), thinking = false, over = null, dead = false, flipped = false, moves = [];
      var row = PG.el('div', 'pg-row'), left = PG.el('div', 'pg-grow'), side = PG.el('div', 'pg-side'); row.appendChild(left); row.appendChild(side); host.appendChild(row);
      var msg = PG.el('div', 'pg-msg'), ui = PG.mountChessBoard(left, {
        getState: function () { return state; }, flipped: function () { return flipped; }, lastMove: function () { return last; }, canMove: function () { return !thinking && !over && state.t === human; },
        selectable: function () { return true; }, onMove: function (m) { play(m); }
      });
      left.appendChild(msg);
      var bar = PG.el('div', 'pg-bar'); left.appendChild(bar);
      PG.button(bar, 'New game', function () { newGame(); }, true); PG.button(bar, 'Undo', function () { undo(); }); PG.button(bar, 'Flip board', function () { flipped = !flipped; ui.render(); });
      var lv = PG.el('div', 'pg-bar'); lv.appendChild(PG.el('span', '', 'Level')); PG.seg(lv, [[1, 'Beginner'], [2, 'Casual'], [3, 'Club'], [4, 'Strong']], level, function (v) { level = v; PG.store.set('chess-level', v); });
      lv.appendChild(PG.el('span', '', 'You play')); PG.seg(lv, [['w', 'White'], ['b', 'Black']], human, function (v) { human = v; PG.store.set('chess-side', v); newGame(); });
      side.appendChild(lv); var list = PG.el('div', '', ''); list.style.cssText = 'margin-top:12px;max-height:300px;overflow:auto;font-family:ui-monospace,Menlo,monospace;font-size:12.5px;line-height:1.7'; side.appendChild(list);
      function newGame() { state = CH.start(); hist = [CH.key(state)]; keys = []; last = null; over = null; moves = []; thinking = false; flipped = human === 'b'; ui.clear(); ui.render(); list.innerHTML = ''; msg.textContent = human === 'w' ? 'Your move (white).' : 'Computer is thinking...'; if (state.t !== human) setTimeout(engine, 350); }
      function play(m) {
        var txt = CH.san(state, m); keys.push({ s: state, last: last, hist: hist.slice(), moves: moves.slice() }); state = CH.apply(state, m); hist.push(CH.key(state)); last = m;
        moves.push(txt); renderList(); ui.clear(); ui.render(); var st = CH.status(state, hist);
        if (st.over) { over = st; msg.textContent = st.reason === 'checkmate' ? (st.winner === human ? 'Checkmate. You win.' : 'Checkmate. The computer wins.') : 'Draw by ' + ({ stalemate: 'stalemate', fifty: 'the 50-move rule', material: 'insufficient material', repetition: 'repetition' })[st.reason] + '.'; return; }
        if (state.t === human) msg.textContent = st.check ? 'Check! Your move.' : 'Your move.'; else { msg.textContent = 'Computer is thinking...'; setTimeout(engine, 250); }
      }
      function renderList() { var h = ''; for (var i = 0; i < moves.length; i += 2) h += (i / 2 + 1) + '. ' + moves[i] + ' ' + (moves[i + 1] || '') + '<br>'; list.innerHTML = h; list.scrollTop = 1e6; }
      function engine() { if (dead || over) return; thinking = true; setTimeout(function () { if (dead) return; var m = CH.best(state, level); thinking = false; if (m) play(m); }, 30); }
      function undo() {
        if (thinking || !keys.length) return; var n = state.t === human ? 2 : 1; while (n-- > 0 && keys.length) { var k = keys.pop(); state = k.s; last = k.last; hist = k.hist; moves = k.moves; }
        over = null; ui.clear(); ui.render(); renderList(); msg.textContent = 'Your move.';
      }
      side.insertAdjacentHTML('beforeend', '<p class="pg-hint">Tip: tap a piece to see where it can go, then tap the square. The engine plays instantly at lower levels and thinks up to about 2 seconds at Strong.</p>');
      newGame();
      return { destroy: function () { dead = true; } };
    }
  });

  /* ---------------------------------------------------------------- 2048 Quant Edition */
  var QNAMES = { 2: 'Tick', 4: 'Pip', 8: 'Bar', 16: 'Candle', 32: 'Setup', 64: 'Trade', 128: 'Strategy', 256: 'Edge', 512: 'Portfolio', 1024: 'Fund', 2048: 'T58', 4096: 'Alpha', 8192: 'Legend' };
  PG.register('arcade', '2048', {
    title: '2048 Quant Edition', sub: 'Slide with arrow keys, WASD or swipes. Merge matching tiles. Toggle plain powers of two if you prefer numbers.',
    mount: function (host) {
      var plain = PG.store.get('g48-plain', false), best = PG.store.get('g48-best', 0), score = 0, tiles = [], nid = 1, undoStack = [], over = false, won = false;
      var top = PG.el('div', 'pg-bar'); host.appendChild(top);
      var sScore = PG.el('span', 'pg-stat'), sBest = PG.el('span', 'pg-stat'); top.appendChild(sScore); top.appendChild(sBest);
      PG.button(top, 'New game', function () { reset(); }, true); PG.button(top, 'Undo', function () { undo(); });
      PG.seg(top, [[false, 'Quant names'], [true, 'Powers of 2']], plain, function (v) { plain = v; PG.store.set('g48-plain', v); tiles.forEach(paint); });
      var box = PG.el('div', 'g48'), grid = PG.el('div', 'g48-grid'); for (var i = 0; i < 16; i++) grid.appendChild(PG.el('div', 'g48-cell')); box.appendChild(grid);
      var ov = PG.el('div', 'g48-over'); box.appendChild(ov); host.appendChild(box);
      host.appendChild(PG.el('p', 'pg-hint', 'Best score is saved on this device. Reach <b>T58</b> (2048) and keep going: there are tiles beyond it.'));
      function colorFor(v) { var e = Math.log2(v); return 'hsl(' + ((165 + e * 22) % 360) + ',' + (62 + e) + '%,' + Math.max(48, 74 - e * 2.2) + '%)'; }
      function paint(t) {
        t.el.style.setProperty('--r', t.r); t.el.style.setProperty('--c', t.c); t.el.style.background = colorFor(t.v); t.el.classList.toggle('big', plain && t.v < 1000);
        t.el.innerHTML = plain ? String(t.v) : (QNAMES[t.v] || String(t.v)) + '<i>' + t.v + '</i>';
      }
      function addTile(r, c, v, pop) { var el = PG.el('div', 'g48-tile'); box.appendChild(el); var t = { id: nid++, r: r, c: c, v: v, el: el }; if (pop) el.classList.add('pop'); tiles.push(t); paint(t); return t; }
      function spawn() { var free = []; for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++) if (!at(r, c)) free.push([r, c]); if (!free.length) return; var f = free[Math.floor(Math.random() * free.length)]; addTile(f[0], f[1], Math.random() < .9 ? 2 : 4, true); }
      function at(r, c) { for (var i = 0; i < tiles.length; i++) if (tiles[i].r === r && tiles[i].c === c && !tiles[i].dying) return tiles[i]; return null; }
      function snapshot() { return { s: score, t: tiles.filter(function (t) { return !t.dying; }).map(function (t) { return [t.r, t.c, t.v]; }) }; }
      function restore(sn) { tiles.forEach(function (t) { t.el.remove(); }); tiles = []; score = sn.s; sn.t.forEach(function (a) { addTile(a[0], a[1], a[2]); }); refresh(); }
      function refresh() { sScore.innerHTML = 'Score <b>' + score + '</b>'; sBest.innerHTML = 'Best <b>' + best + '</b>'; }
      function reset() { tiles.forEach(function (t) { t.el.remove(); }); tiles = []; score = 0; undoStack = []; over = false; won = false; ov.classList.remove('on'); spawn(); spawn(); refresh(); }
      function undo() { if (undoStack.length) { restore(undoStack.pop()); over = false; ov.classList.remove('on'); } }
      function move(dir) {   // 0 up 1 right 2 down 3 left
        if (over) return; var before = snapshot(), moved = false, gain = 0;
        for (var line = 0; line < 4; line++) {
          var cells = []; for (var k = 0; k < 4; k++) { var r = dir % 2 === 0 ? (dir === 0 ? k : 3 - k) : line, c = dir % 2 === 0 ? line : (dir === 3 ? k : 3 - k); cells.push([r, c]); }
          var seq = cells.map(function (p) { return at(p[0], p[1]); }).filter(Boolean), out = [], idx = 0;
          while (idx < seq.length) {
            var a = seq[idx], b = seq[idx + 1];
            if (b && a.v === b.v) { out.push({ keep: a, gone: b, v: a.v * 2 }); gain += a.v * 2; idx += 2; } else { out.push({ keep: a }); idx++; }
          }
          out.forEach(function (o, n) {
            var p = cells[n]; if (o.keep.r !== p[0] || o.keep.c !== p[1]) moved = true; o.keep.r = p[0]; o.keep.c = p[1];
            if (o.gone) { moved = true; o.gone.r = p[0]; o.gone.c = p[1]; o.gone.dying = true; o.keep.v = o.v; o.keep.merged = true; }
          });
        }
        if (!moved) return; undoStack.push(before); if (undoStack.length > 20) undoStack.shift();
        tiles.forEach(paint); score += gain; if (score > best) { best = score; PG.store.set('g48-best', best); }
        setTimeout(function () {
          tiles = tiles.filter(function (t) { if (t.dying) { t.el.remove(); return false; } return true; });
          tiles.forEach(function (t) { if (t.merged) { t.merged = false; paint(t); t.el.classList.remove('pop'); void t.el.offsetWidth; t.el.classList.add('pop'); if (t.v === 2048 && !won) { won = true; flash('You reached T58! Keep going for Alpha.'); } } });
          spawn(); refresh(); if (!canMove()) { over = true; ov.innerHTML = '<div>Game over<br><small>Score ' + score + '</small><br><br><button class="pg-btn pri" type="button">Play again</button></div>'; ov.querySelector('button').onclick = reset; ov.classList.add('on'); }
        }, 120);
      }
      function flash(t) { var m = PG.el('div', 'pg-msg', t); host.appendChild(m); setTimeout(function () { m.remove(); }, 4000); }
      function canMove() { if (tiles.length < 16) return true; for (var r = 0; r < 4; r++) for (var c = 0; c < 4; c++) { var t = at(r, c); if ((c < 3 && at(r, c + 1).v === t.v) || (r < 3 && at(r + 1, c).v === t.v)) return true; } return false; }
      var off = PG.keyHandler(function (e) { var m = { ArrowUp: 0, ArrowRight: 1, ArrowDown: 2, ArrowLeft: 3, w: 0, d: 1, s: 2, a: 3, W: 0, D: 1, S: 2, A: 3 }[e.key]; if (m !== undefined) { e.preventDefault(); move(m); } });
      var sx = 0, sy = 0; box.addEventListener('touchstart', function (e) { sx = e.touches[0].clientX; sy = e.touches[0].clientY; }, { passive: true });
      box.addEventListener('touchend', function (e) { var dx = e.changedTouches[0].clientX - sx, dy = e.changedTouches[0].clientY - sy; if (Math.max(Math.abs(dx), Math.abs(dy)) < 24) return; move(Math.abs(dx) > Math.abs(dy) ? (dx > 0 ? 1 : 3) : (dy > 0 ? 2 : 0)); });
      box.addEventListener('touchmove', function (e) { e.preventDefault(); }, { passive: false });
      reset();
      return { destroy: function () { off(); } };
    }
  });

  /* ---------------------------------------------------------------- Tetris, terminal edition */
  var SHAPES = {
    I: [[0, 1], [1, 1], [2, 1], [3, 1]], O: [[1, 0], [2, 0], [1, 1], [2, 1]], T: [[1, 0], [0, 1], [1, 1], [2, 1]], S: [[1, 0], [2, 0], [0, 1], [1, 1]],
    Z: [[0, 0], [1, 0], [1, 1], [2, 1]], J: [[0, 0], [0, 1], [1, 1], [2, 1]], L: [[2, 0], [0, 1], [1, 1], [2, 1]]
  };
  var KINDS = 'IOTSZJL';
  PG.register('arcade', 'tetris', {
    title: 'Tetris', sub: 'Terminal edition. Arrows or WASD to move/rotate, Space to hard drop, C to hold, P to pause.',
    mount: function (host) {
      var W = 10, H = 20, grid, cur, next, hold, held, bag = [], score, lines, level, over, paused, acc, started, t0, elapsed, dirty = true, best = PG.store.get('tetris-best', { scores: [], lines: 0, level: 0, time: 0 });
      var wrap = PG.el('div', 'tt-wrap'), pre = PG.el('pre'); wrap.appendChild(pre); var row = PG.el('div', 'tt-row'); row.appendChild(wrap); host.appendChild(row);
      var bar = PG.el('div', 'pg-bar'); host.appendChild(bar);
      var bStart = PG.button(bar, 'New game', function () { reset(); }, true), bPause = PG.button(bar, 'Pause', function () { togglePause(); });
      var pad = PG.el('div', 'tt-pad'); [['↺', 'rot'], ['←', 'left'], ['↓', 'down'], ['→', 'right'], ['⤓', 'drop'], ['H', 'hold']].forEach(function (b) { var x = PG.el('button', '', b[0]); x.type = 'button'; x.onpointerdown = function (e) { e.preventDefault(); act(b[1]); }; pad.appendChild(x); }); host.appendChild(pad);
      var pbs = PG.el('div', 'pg-list'); host.appendChild(PG.el('h4', '', 'Personal bests')); host.lastChild.style.cssText = 'margin:14px 0 0;font-size:13px'; host.appendChild(pbs);
      function bag7() { if (!bag.length) { bag = KINDS.split(''); for (var i = bag.length - 1; i > 0; i--) { var j = Math.floor(Math.random() * (i + 1)), t = bag[i]; bag[i] = bag[j]; bag[j] = t; } } return bag.pop(); }
      function piece(k) { return { k: k, cells: SHAPES[k].map(function (a) { return a.slice(); }), x: 3, y: 0, rot: 0 }; }
      function rotated(p, dir) { if (p.k === 'O') return p.cells; var size = p.k === 'I' ? 4 : 3; return p.cells.map(function (a) { return dir > 0 ? [size - 1 - a[1], a[0]] : [a[1], size - 1 - a[0]]; }); }
      function fits(cells, x, y) { for (var i = 0; i < cells.length; i++) { var cx = cells[i][0] + x, cy = cells[i][1] + y; if (cx < 0 || cx >= W || cy >= H) return false; if (cy >= 0 && grid[cy][cx]) return false; } return true; }
      function reset() { grid = []; for (var i = 0; i < H; i++) grid.push(new Array(W).fill('')); score = 0; lines = 0; level = 1; over = false; paused = false; acc = 0; hold = null; held = false; bag = []; next = bag7(); spawn(); t0 = performance.now(); elapsed = 0; started = true; bPause.textContent = 'Pause'; dirty = true; renderPB(); }
      function spawn() { cur = piece(next); next = bag7(); held = false; if (!fits(cur.cells, cur.x, cur.y)) gameOver(); }
      function gameOver() { over = true; var b = best; b.scores.push(score); b.scores.sort(function (a, c) { return c - a; }); b.scores = b.scores.slice(0, 5); b.lines = Math.max(b.lines, lines); b.level = Math.max(b.level, level); b.time = Math.max(b.time, Math.floor(elapsed)); PG.store.set('tetris-best', b); renderPB(); }
      function renderPB() { var b = best; pbs.innerHTML = '<div class="pg-pb"><b>' + (b.scores[0] || 0) + '</b>best score</div><div class="pg-pb"><b>' + b.lines + '</b>most lines</div><div class="pg-pb"><b>' + b.level + '</b>highest level</div><div class="pg-pb"><b>' + PG.fmtTime(b.time) + '</b>longest game</div>' + (b.scores.length > 1 ? '<div class="pg-pb"><b>' + b.scores.join(' / ') + '</b>top 5 scores</div>' : ''); }
      function lock() {
        cur.cells.forEach(function (a) { var y = a[1] + cur.y, x = a[0] + cur.x; if (y >= 0) grid[y][x] = cur.k; });
        var n = 0; for (var y = H - 1; y >= 0; y--) { if (grid[y].every(Boolean)) { grid.splice(y, 1); grid.unshift(new Array(W).fill('')); n++; y++; } }
        if (n) { lines += n; score += [0, 100, 300, 500, 800][n] * level; level = 1 + Math.floor(lines / 10); }
        spawn(); dirty = true;
      }
      function ghostY() { var y = cur.y; while (fits(cur.cells, cur.x, y + 1)) y++; return y; }
      function act(a) {
        if (over || paused || !started) return;
        if (a === 'left' && fits(cur.cells, cur.x - 1, cur.y)) cur.x--; else if (a === 'right' && fits(cur.cells, cur.x + 1, cur.y)) cur.x++;
        else if (a === 'down') { if (fits(cur.cells, cur.x, cur.y + 1)) { cur.y++; score += 1; } else lock(); acc = 0; }
        else if (a === 'rot') { var r = rotated(cur, 1); [0, -1, 1, -2, 2].some(function (k) { if (fits(r, cur.x + k, cur.y)) { cur.cells = r; cur.x += k; return true; } return false; }); }
        else if (a === 'drop') { var y = ghostY(); score += 2 * (y - cur.y); cur.y = y; lock(); acc = 0; }
        else if (a === 'hold' && !held) { var k = cur.k; if (hold) { cur = piece(hold); } else { cur = piece(next); next = bag7(); } hold = k; held = true; }
        dirty = true;
      }
      function togglePause() { if (over) return; paused = !paused; bPause.textContent = paused ? 'Resume' : 'Pause'; dirty = true; }
      var off = PG.keyHandler(function (e) {
        var m = { ArrowLeft: 'left', a: 'left', A: 'left', ArrowRight: 'right', d: 'right', D: 'right', ArrowDown: 'down', s: 'down', S: 'down', ArrowUp: 'rot', w: 'rot', W: 'rot', x: 'rot', ' ': 'drop', c: 'hold', C: 'hold' }[e.key];
        if (e.key === 'p' || e.key === 'P' || e.key === 'Escape') { togglePause(); return; } if (m) { e.preventDefault(); act(m); }
      });
      var BLK = { I: '[]', O: '[]', T: '[]', S: '[]', Z: '[]', J: '[]', L: '[]' };
      function draw() {
        var v = grid.map(function (r) { return r.slice(); }), gy = over ? cur.y : ghostY();
        if (!over) { cur.cells.forEach(function (a) { var y = a[1] + gy, x = a[0] + cur.x; if (y >= 0 && !v[y][x]) v[y][x] = '.'; }); }
        cur.cells.forEach(function (a) { var y = a[1] + cur.y, x = a[0] + cur.x; if (y >= 0) v[y][x] = cur.k; });
        function mini(k) { var rows = ['        ', '        ']; if (k) SHAPES[k].forEach(function (a) { var rr = a[1] > 1 ? 1 : a[1]; var r = k === 'I' ? 0 : a[1]; if (k === 'I') { rows[0] = '[][][][]'; } else { var s = rows[r]; var px = a[0] * 2 + (k === 'O' ? 0 : 0); rows[r] = s.substr(0, px) + '[]' + s.substr(px + 2); } }); return rows; }
        var nx = mini(next), hd = mini(hold), side = [
          'NEXT', nx[0], nx[1], '', 'HOLD', hd[0], hd[1], '', 'SCORE', String(score), '', 'LINES ' + lines, 'LEVEL ' + level, 'TIME  ' + PG.fmtTime(elapsed), '', 'BEST  ' + (best.scores[0] || 0)
        ];
        var out = 't58@quant:~$ ./tetris' + ((Date.now() % 1000) < 500 ? ' _' : '  ') + '\n\n';
        for (var y = 0; y < H; y++) {
          var line = '<!'; for (var x = 0; x < W; x++) { var c = v[y][x]; line += c === '.' ? ' .' : c ? BLK[c] : ' .'.replace('.', ' '); } line += '!>';
          out += line + '  ' + (side[y] || '') + '\n';
        }
        out += '<!' + '=='.repeat(W) + '!>\n' + '  ' + '\\/'.repeat(W) + '\n';
        if (paused) out += '\n  [ PAUSED ] press P to resume\n'; else if (over) out += '\n  GAME OVER. score ' + score + '. press New game\n'; else out += '\n  arrows/WASD move  space drop  C hold  P pause\n';
        pre.textContent = out;
      }
      var L = PG.loop(function (dt) {
        if (started && !over && !paused) {
          elapsed = (performance.now() - t0) / 1000 - 0; acc += dt; var step = Math.max(0.07, 0.8 - (level - 1) * 0.07);
          while (acc >= step) { acc -= step; if (fits(cur.cells, cur.x, cur.y + 1)) cur.y++; else lock(); dirty = true; }
          dirty = true;
        }
        if (dirty) { draw(); dirty = false; }
      });
      var pauseVis = function () { if (document.hidden && started && !over && !paused) { paused = true; bPause.textContent = 'Resume'; dirty = true; } };
      document.addEventListener('visibilitychange', pauseVis);
      reset(); paused = true; bPause.textContent = 'Resume'; dirty = true;   // starts paused so nothing falls before you are ready
      return { destroy: function () { L.stop(); off(); document.removeEventListener('visibilitychange', pauseVis); } };
    }
  });
})();
