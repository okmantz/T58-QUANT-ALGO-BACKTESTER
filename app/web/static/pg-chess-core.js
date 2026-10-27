/* Compact, complete chess rules + a small alpha-beta engine (castling, en passant, promotion, check/mate/stalemate, 50-move,
   repetition, insufficient material). Board index 0 = a8 ... 63 = h1. Upper-case = white, lower-case = black, '' = empty. */
(function () {
  var VAL = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
  var KN = [[-2, -1], [-2, 1], [-1, -2], [-1, 2], [1, -2], [1, 2], [2, -1], [2, 1]], KG = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]];
  var DIAG = [[-1, -1], [-1, 1], [1, -1], [1, 1]], ORTH = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  function isW(p) { return p !== '' && p === p.toUpperCase(); }
  function fromFEN(fen) {
    var f = fen.split(' '), b = [];
    f[0].split('/').forEach(function (row) { for (var i = 0; i < row.length; i++) { var ch = row[i]; if (/\d/.test(ch)) for (var k = 0; k < +ch; k++) b.push(''); else b.push(ch); } });
    var c = f[2] || '-';
    return { b: b, t: f[1] || 'w', c: { K: c.indexOf('K') >= 0, Q: c.indexOf('Q') >= 0, k: c.indexOf('k') >= 0, q: c.indexOf('q') >= 0 }, ep: f[3] && f[3] !== '-' ? ('abcdefgh'.indexOf(f[3][0]) + (8 - +f[3][1]) * 8) : -1, hm: +f[4] || 0, fm: +f[5] || 1 };
  }
  function start() { return fromFEN('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'); }
  function attacked(b, sq, byW) {
    var r = (sq >> 3), c = sq & 7, i, d, p, rr, cc;
    var pr = byW ? r + 1 : r - 1, pp = byW ? 'P' : 'p';
    if (pr >= 0 && pr < 8) { if (c > 0 && b[pr * 8 + c - 1] === pp) return true; if (c < 7 && b[pr * 8 + c + 1] === pp) return true; }
    var kn = byW ? 'N' : 'n', kg = byW ? 'K' : 'k';
    for (i = 0; i < 8; i++) { rr = r + KN[i][0]; cc = c + KN[i][1]; if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && b[rr * 8 + cc] === kn) return true; }
    for (i = 0; i < 8; i++) { rr = r + KG[i][0]; cc = c + KG[i][1]; if (rr >= 0 && rr < 8 && cc >= 0 && cc < 8 && b[rr * 8 + cc] === kg) return true; }
    var bq = byW ? 'BQ' : 'bq', rq = byW ? 'RQ' : 'rq';
    for (d = 0; d < 4; d++) { rr = r + DIAG[d][0]; cc = c + DIAG[d][1]; while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) { p = b[rr * 8 + cc]; if (p) { if (bq.indexOf(p) >= 0) return true; break; } rr += DIAG[d][0]; cc += DIAG[d][1]; } }
    for (d = 0; d < 4; d++) { rr = r + ORTH[d][0]; cc = c + ORTH[d][1]; while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) { p = b[rr * 8 + cc]; if (p) { if (rq.indexOf(p) >= 0) return true; break; } rr += ORTH[d][0]; cc += ORTH[d][1]; } }
    return false;
  }
  function kingSq(b, w) { var k = w ? 'K' : 'k'; for (var i = 0; i < 64; i++) if (b[i] === k) return i; return -1; }
  function inCheck(s, w) { if (w === undefined) w = s.t === 'w'; var k = kingSq(s.b, w); return k >= 0 && attacked(s.b, k, !w); }
  function pseudo(s) {
    var m = [], b = s.b, w = s.t === 'w', i, r, c, p, d, rr, cc, to, t;
    function add(from, to, extra) { var mv = { from: from, to: to, cap: b[to] || '' }; if (extra) for (var k in extra) mv[k] = extra[k]; m.push(mv); }
    for (i = 0; i < 64; i++) {
      p = b[i]; if (!p || isW(p) !== w) continue; r = i >> 3; c = i & 7; t = p.toLowerCase();
      if (t === 'p') {
        var dir = w ? -1 : 1, sr = w ? 6 : 1, pr = w ? 0 : 7, nr = r + dir;
        if (nr < 0 || nr > 7) continue;
        if (!b[nr * 8 + c]) { if (nr === pr) ['q', 'r', 'b', 'n'].forEach(function (x) { add(i, nr * 8 + c, { promo: x }); }); else { add(i, nr * 8 + c); if (r === sr && !b[(r + 2 * dir) * 8 + c]) add(i, (r + 2 * dir) * 8 + c, { dbl: true }); } }
        [-1, 1].forEach(function (dc) {
          cc = c + dc; if (cc < 0 || cc > 7) return; to = nr * 8 + cc;
          if (b[to] && isW(b[to]) !== w) { if (nr === pr) ['q', 'r', 'b', 'n'].forEach(function (x) { add(i, to, { promo: x }); }); else add(i, to); }
          else if (to === s.ep) add(i, to, { ep: true, cap: w ? 'p' : 'P' });
        });
      } else if (t === 'n' || t === 'k') {
        var arr = t === 'n' ? KN : KG;
        for (d = 0; d < 8; d++) { rr = r + arr[d][0]; cc = c + arr[d][1]; if (rr < 0 || rr > 7 || cc < 0 || cc > 7) continue; to = rr * 8 + cc; if (!b[to] || isW(b[to]) !== w) add(i, to); }
        if (t === 'k') {
          var home = w ? 60 : 4, opp = !w;
          if (i === home && !attacked(b, i, opp)) {
            if ((w ? s.c.K : s.c.k) && b[home + 3] === (w ? 'R' : 'r') && !b[home + 1] && !b[home + 2] && !attacked(b, home + 1, opp) && !attacked(b, home + 2, opp)) add(i, home + 2, { castle: 'K' });
            if ((w ? s.c.Q : s.c.q) && b[home - 4] === (w ? 'R' : 'r') && !b[home - 1] && !b[home - 2] && !b[home - 3] && !attacked(b, home - 1, opp) && !attacked(b, home - 2, opp)) add(i, home - 2, { castle: 'Q' });
          }
        }
      } else {
        var dirs = t === 'b' ? DIAG : t === 'r' ? ORTH : DIAG.concat(ORTH);
        for (d = 0; d < dirs.length; d++) { rr = r + dirs[d][0]; cc = c + dirs[d][1]; while (rr >= 0 && rr < 8 && cc >= 0 && cc < 8) { to = rr * 8 + cc; if (!b[to]) add(i, to); else { if (isW(b[to]) !== w) add(i, to); break; } rr += dirs[d][0]; cc += dirs[d][1]; } }
      }
    }
    return m;
  }
  function apply(s, m) {
    var b = s.b.slice(), w = s.t === 'w', p = b[m.from], c = { K: s.c.K, Q: s.c.Q, k: s.c.k, q: s.c.q }, ep = -1;
    b[m.from] = ''; b[m.to] = m.promo ? (w ? m.promo.toUpperCase() : m.promo) : p;
    if (m.ep) b[m.to + (w ? 8 : -8)] = '';
    if (m.castle === 'K') { b[m.to - 1] = b[m.to + 1]; b[m.to + 1] = ''; } else if (m.castle === 'Q') { b[m.to + 1] = b[m.to - 2]; b[m.to - 2] = ''; }
    if (m.dbl) ep = (m.from + m.to) >> 1;
    if (p === 'K') { c.K = c.Q = false; } else if (p === 'k') { c.k = c.q = false; }
    if (m.from === 63 || m.to === 63) c.K = false; if (m.from === 56 || m.to === 56) c.Q = false; if (m.from === 7 || m.to === 7) c.k = false; if (m.from === 0 || m.to === 0) c.q = false;
    return { b: b, t: w ? 'b' : 'w', c: c, ep: ep, hm: (p.toLowerCase() === 'p' || m.cap) ? 0 : s.hm + 1, fm: s.fm + (w ? 0 : 1) };
  }
  function legal(s) {
    var w = s.t === 'w', out = [], ms = pseudo(s);
    for (var i = 0; i < ms.length; i++) { var n = apply(s, ms[i]), k = kingSq(n.b, w); if (k >= 0 && !attacked(n.b, k, !w)) out.push(ms[i]); }
    return out;
  }
  function key(s) { return s.b.join('') + s.t + (s.c.K ? 'K' : '') + (s.c.Q ? 'Q' : '') + (s.c.k ? 'k' : '') + (s.c.q ? 'q' : '') + s.ep; }
  function insufficient(b) {
    var minors = 0, other = false; for (var i = 0; i < 64; i++) { var t = b[i].toLowerCase(); if (!t || t === 'k') continue; if (t === 'n' || t === 'b') minors++; else other = true; }
    return !other && minors <= 1;
  }
  /* result: null | {over:true, reason:'checkmate'|'stalemate'|'fifty'|'material'|'repetition', winner:'w'|'b'|null} */
  function status(s, hist) {
    var l = legal(s), chk = inCheck(s);
    if (!l.length) return chk ? { over: true, reason: 'checkmate', winner: s.t === 'w' ? 'b' : 'w', check: true } : { over: true, reason: 'stalemate', winner: null };
    if (s.hm >= 100) return { over: true, reason: 'fifty', winner: null };
    if (insufficient(s.b)) return { over: true, reason: 'material', winner: null };
    if (hist) { var k = key(s), n = 0; for (var i = 0; i < hist.length; i++) if (hist[i] === k) n++; if (n >= 3) return { over: true, reason: 'repetition', winner: null }; }
    return { over: false, check: chk, moves: l };
  }
  // ---- engine
  var PST_P = [0, 0, 0, 0, 0, 0, 0, 0, 50, 50, 50, 50, 50, 50, 50, 50, 10, 10, 20, 30, 30, 20, 10, 10, 5, 5, 10, 25, 25, 10, 5, 5, 0, 0, 0, 20, 20, 0, 0, 0, 5, -5, -10, 0, 0, -10, -5, 5, 5, 10, 10, -20, -20, 10, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0];
  function cen(i) { var r = i >> 3, c = i & 7, dr = Math.min(r, 7 - r), dc = Math.min(c, 7 - c); return (dr + dc) * 6 - 12; }
  function evaluate(s) {
    var score = 0, mat = 0, i, p, t, v;
    for (i = 0; i < 64; i++) { p = s.b[i]; if (!p) continue; t = p.toLowerCase(); if (t !== 'k' && t !== 'p') mat += VAL[t]; }
    for (i = 0; i < 64; i++) {
      p = s.b[i]; if (!p) continue; t = p.toLowerCase(); var w = isW(p); v = VAL[t]; var idx = w ? i : 63 - i ^ 7 ^ 0;
      if (!w) idx = (7 - (i >> 3)) * 8 + (i & 7);
      if (t === 'p') v += PST_P[idx]; else if (t === 'n' || t === 'b') v += cen(i); else if (t === 'q') v += cen(i) / 3;
      else if (t === 'k') { var r = i >> 3, c = i & 7; v += mat > 2400 ? (w ? (r - 7) * -8 : (r) * -8) + (Math.abs(c - 3.5) > 2 ? 12 : 0) : -cen(i) * 1.5; }
      score += w ? v : -v;
    }
    return s.t === 'w' ? score : -score;
  }
  var nodes, deadline, stopped;
  function order(ms) { return ms.sort(function (a, b) { return ((VAL[b.cap.toLowerCase()] || 0) * 10 + (b.promo ? 800 : 0)) - ((VAL[a.cap.toLowerCase()] || 0) * 10 + (a.promo ? 800 : 0)); }); }
  function qsearch(s, alpha, beta, depth) {
    var stand = evaluate(s); if (stand >= beta) return beta; if (stand > alpha) alpha = stand; if (depth <= 0) return alpha;
    var ms = order(pseudo(s).filter(function (m) { return m.cap || m.promo; })), w = s.t === 'w';
    for (var i = 0; i < ms.length; i++) { var n = apply(s, ms[i]), k = kingSq(n.b, w); if (k < 0 || attacked(n.b, k, !w)) continue; var sc = -qsearch(n, -beta, -alpha, depth - 1); if (sc >= beta) return beta; if (sc > alpha) alpha = sc; }
    return alpha;
  }
  function search(s, depth, alpha, beta, ply) {
    if ((++nodes & 1023) === 0 && performance.now() > deadline) stopped = true; if (stopped) return 0;
    var chk = inCheck(s); if (chk) depth++;
    if (depth <= 0) return qsearch(s, alpha, beta, 4);
    var ms = order(pseudo(s)), w = s.t === 'w', any = false;
    for (var i = 0; i < ms.length; i++) {
      var n = apply(s, ms[i]), k = kingSq(n.b, w); if (k < 0 || attacked(n.b, k, !w)) continue; any = true;
      var sc = -search(n, depth - 1, -beta, -alpha, ply + 1); if (stopped) return 0;
      if (sc >= beta) return beta; if (sc > alpha) alpha = sc;
    }
    if (!any) return chk ? -100000 + ply : 0;
    if (s.hm >= 100) return 0;
    return alpha;
  }
  /* best move for the side to move. level 1 = random-ish, 2 = depth 2, 3 = depth 3, 4 = iterative to depth 5 (time capped) */
  function best(s, level, msCap) {
    var ms = legal(s); if (!ms.length) return null;
    if (level <= 1) { var caps = ms.filter(function (m) { return m.cap; }); return Math.random() < .35 && caps.length ? caps[Math.floor(Math.random() * caps.length)] : ms[Math.floor(Math.random() * ms.length)]; }
    nodes = 0; stopped = false; deadline = performance.now() + (msCap || (level >= 4 ? 1800 : level === 3 ? 900 : 350));
    var maxD = level >= 4 ? 5 : level === 3 ? 4 : 2, bestM = ms[0];
    order(ms);
    for (var d = 1; d <= maxD; d++) {
      var alpha = -1e9, cand = null, scores = [];
      for (var i = 0; i < ms.length; i++) {
        var sc = -search(apply(s, ms[i]), d - 1, -1e9, -alpha, 1); if (stopped) break;
        scores.push(sc); if (sc > alpha) { alpha = sc; cand = ms[i]; }
      }
      if (stopped) break; bestM = cand || bestM;
      if (cand) { ms.splice(ms.indexOf(cand), 1); ms.unshift(cand); }
      if (alpha > 90000) break;
    }
    if (level === 2) { var pool = ms.slice(0, 1); bestM = bestM; }
    return bestM;
  }
  /* can the side to move force mate within n of its own moves? (used by the daily puzzles) */
  function mateIn(s, n) {
    var ms = legal(s);
    for (var i = 0; i < ms.length; i++) {
      var ns = apply(s, ms[i]), rep = legal(ns);
      if (!rep.length) { if (inCheck(ns)) return ms[i]; continue; }
      if (n <= 1) continue;
      var all = true; for (var j = 0; j < rep.length; j++) { if (!mateIn(apply(ns, rep[j]), n - 1)) { all = false; break; } }
      if (all) return ms[i];
    }
    return null;
  }
  function mateMoves(s, n) {   // every first move that forces mate in <= n
    var out = [], ms = legal(s);
    ms.forEach(function (m) {
      var ns = apply(s, m), rep = legal(ns);
      if (!rep.length) { if (inCheck(ns)) out.push(m); return; }
      if (n <= 1) return;
      for (var j = 0; j < rep.length; j++) if (!mateIn(apply(ns, rep[j]), n - 1)) return;
      out.push(m);
    });
    return out;
  }
  function sqName(i) { return 'abcdefgh'[i & 7] + (8 - (i >> 3)); }
  function san(s, m) {   // short algebraic, good enough for a move list
    var p = s.b[m.from], t = p.toLowerCase(), txt;
    if (m.castle) txt = m.castle === 'K' ? 'O-O' : 'O-O-O';
    else {
      txt = t === 'p' ? (m.cap ? 'abcdefgh'[m.from & 7] + 'x' : '') : t.toUpperCase();
      if (t !== 'p') { var amb = legal(s).filter(function (o) { return o !== m && s.b[o.from] === p && o.to === m.to; }); if (amb.length) txt += (amb.some(function (o) { return (o.from & 7) === (m.from & 7); }) ? (8 - (m.from >> 3)) : 'abcdefgh'[m.from & 7]); if (m.cap) txt += 'x'; }
      txt += sqName(m.to); if (m.promo) txt += '=' + m.promo.toUpperCase();
    }
    var n = apply(s, m), st = status(n); return txt + (st.over && st.reason === 'checkmate' ? '#' : inCheck(n) ? '+' : '');
  }
  window.PG = window.PG || {};
  window.PG.chess = { fromFEN: fromFEN, start: start, legal: legal, apply: apply, inCheck: inCheck, status: status, key: key, best: best, mateIn: mateIn, mateMoves: mateMoves, san: san, sqName: sqName, isW: isW, evaluate: evaluate, kingSq: kingSq };
  var PG = window.PG, CH = window.PG.chess;
  var GLYPH = { k: '♚︎', q: '♛︎', r: '♜︎', b: '♝︎', n: '♞︎', p: '♟︎' };
  window.PG.mountChessBoard = function (host, o) {
    /* o: {getState, onMove(m), flipped(), lastMove(), selectable(sq)} -> {render(), setPromo} ; shared by Arcade chess and Daily puzzle. */
    var board = PG.el('div', 'chess-board'); host.appendChild(board);
    var sel = -1, targets = [], promoBox = PG.el('div', 'pg-promo'); host.appendChild(promoBox);
    function render() {
      var s = o.getState(), flip = o.flipped(), last = o.lastMove(), chkSq = CH.inCheck(s) ? CH.kingSq(s.b, s.t === 'w') : -1; board.innerHTML = '';
      for (var vi = 0; vi < 64; vi++) {
        var i = flip ? 63 - vi : vi, r = i >> 3, c = i & 7, p = s.b[i], d = PG.el('div', 'chess-sq ' + (((r + c) & 1) ? 'd' : 'l'));
        d.dataset.sq = i; if (p) { d.innerHTML = '<span class="' + (CH.isW(p) ? 'chess-w' : 'chess-b') + '">' + GLYPH[p.toLowerCase()] + '</span>'; }
        if (i === sel) d.classList.add('sel'); if (last && (i === last.from || i === last.to)) d.classList.add('last'); if (i === chkSq) d.classList.add('chk');
        var tg = targets.filter(function (m) { return m.to === i; })[0]; if (tg) d.classList.add(tg.cap ? 'cap' : 'dot');
        if (vi % 8 === 0) { var rk = PG.el('span', '', 8 - r); rk.style.cssText = 'position:absolute;left:3px;top:1px;font-size:10px;opacity:.55;color:#000'; d.appendChild(rk); }
        if (vi >= 56) { var fl = PG.el('span', '', 'abcdefgh'[c]); fl.style.cssText = 'position:absolute;right:3px;bottom:0;font-size:10px;opacity:.55;color:#000'; d.appendChild(fl); }
        board.appendChild(d);
      }
    }
    board.addEventListener('click', function (e) {
      var d = e.target.closest('.chess-sq'); if (!d || !o.canMove()) return; var i = +d.dataset.sq, s = o.getState();
      var mv = targets.filter(function (m) { return m.to === i; });
      if (sel >= 0 && mv.length) {
        if (mv[0].promo) {
          promoBox.innerHTML = ''; promoBox.classList.add('on');
          ['q', 'r', 'b', 'n'].forEach(function (pc) { var b = PG.el('button', 'pg-btn', GLYPH[pc]); b.style.fontSize = '24px'; b.onclick = function () { promoBox.classList.remove('on'); var m = mv.filter(function (x) { return x.promo === pc; })[0]; sel = -1; targets = []; o.onMove(m); }; promoBox.appendChild(b); });
          return;
        }
        sel = -1; targets = []; o.onMove(mv[0]); return;
      }
      var p = s.b[i];
      if (p && (CH.isW(p) === (s.t === 'w')) && o.selectable(i)) { sel = i; targets = CH.legal(s).filter(function (m) { return m.from === i; }); } else { sel = -1; targets = []; }
      promoBox.classList.remove('on'); render();
    });
    return { render: render, clear: function () { sel = -1; targets = []; promoBox.classList.remove('on'); } };
  };

})();
