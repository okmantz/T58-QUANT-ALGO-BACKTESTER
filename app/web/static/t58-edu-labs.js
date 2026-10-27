/* T58 Education -- interactive labs (2026-10-17). Learn by playing with the numbers instead of only reading about them.
 * Pure DOM + SVG, no dependencies; every lab is self-contained and seeded so results are reproducible.
 * Labs: 1 Edge calculator, 2 Risk-per-trade fan, 3 Trailing-drawdown game, 4 Overfit-your-own-strategy,
 *       5 Gamma regime explorer.   (The 6-question knowledge check is the last module of the Education course: see mountQuiz.)
 *
 * Every lab can be mounted two ways: ALL of them together on the Interactive labs tab (mount), or ONE at a time
 * inside the lesson it belongs to (mountOne) so a module reads in chunks instead of one long wall of text.
 */
(function (root) {
  'use strict';
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function rng(seed) { var a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { var u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function money(n) { return (n < 0 ? '-$' : '$') + Math.abs(Math.round(n)).toLocaleString(); }
  var NS = 'http://www.w3.org/2000/svg';
  function svgEl(tag, attrs) { var e = document.createElementNS(NS, tag); Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); }); return e; }

  function slider(host, label, min, max, step, value, fmt, onChange) {
    var w = el('div', 'lab-ctl'), l = el('label'), out = el('b'); l.appendChild(document.createTextNode(label + ': ')); l.appendChild(out); w.appendChild(l);
    var i = el('input'); i.type = 'range'; i.min = min; i.max = max; i.step = step; i.value = value; w.appendChild(i);
    function upd() { out.textContent = fmt(parseFloat(i.value)); }
    i.addEventListener('input', function () { upd(); onChange(parseFloat(i.value)); }); upd(); host.appendChild(w); return i;
  }
  function stat(host, label, metric) { var d = el('div', 'lab-stat'), s = el('span', null, label); if (metric) s.setAttribute('data-metric', metric); d.appendChild(s); var b = el('b', null, '--'); d.appendChild(b); host.appendChild(d); return b; }

  /* -------- Lab 1: edge calculator -------- */
  function lab1(host) {
    var S = { w: 45, win: 300, loss: 150 };
    var ctl = el('div', 'lab-ctls'), out = el('div', 'lab-stats'); host.appendChild(ctl); host.appendChild(out);
    var exp = stat(out, 'Expectancy per trade', 'expectancy'), pf = stat(out, 'Profit factor', 'profit_factor'), be = stat(out, 'Break-even win rate'), pay = stat(out, 'Payoff ratio (avg win ÷ avg loss)'), v = el('p', 'lab-note'); host.appendChild(v);
    function calc() {
      var w = S.w / 100, e = w * S.win - (1 - w) * S.loss, gp = w * S.win, gl = (1 - w) * S.loss;
      exp.textContent = money(e); exp.className = e >= 0 ? 'pos' : 'neg'; pf.textContent = gl > 0 ? (gp / gl).toFixed(2) : '∞'; be.textContent = (S.loss / (S.win + S.loss) * 100).toFixed(1) + '%'; pay.textContent = (S.win / S.loss).toFixed(2);
      v.textContent = e > 0 ? 'You make ' + money(e) + ' on an average trade. Over 100 trades that is about ' + money(e * 100) + ' — if the win rate and sizes are real and hold out of sample.' : 'This loses ' + money(-e) + ' per trade. Raise the win rate above ' + (S.loss / (S.win + S.loss) * 100).toFixed(1) + '% or widen winners relative to losers.';
    }
    slider(ctl, 'Win rate', 10, 90, 1, S.w, function (x) { return x + '%'; }, function (x) { S.w = x; calc(); });
    slider(ctl, 'Average win', 50, 1000, 10, S.win, money, function (x) { S.win = x; calc(); });
    slider(ctl, 'Average loss', 50, 1000, 10, S.loss, money, function (x) { S.loss = x; calc(); });
    calc();
  }

  /* -------- Lab 2: risk-per-trade fan -------- */
  function lab2(host) {
    var S = { w: 45, payoff: 1.8, risk: 1, n: 150, seed: 3 };
    var ctl = el('div', 'lab-ctls'), box = el('div'), out = el('div', 'lab-stats'); host.appendChild(ctl); host.appendChild(box); host.appendChild(out);
    var med = stat(out, 'Median ending balance (start $100k)'), dd = stat(out, 'Paths that touch -10% drawdown', 'max_drawdown'), prof = stat(out, 'Paths ending in profit');
    function run() {
      var N = 40, paths = [], r = rng(S.seed * 131 + 7), ends = [], hit = 0, win = 0;
      for (var p = 0; p < N; p++) { var bal = 100000, peak = bal, path = [bal], touched = false; for (var i = 0; i < S.n; i++) { var risk = 100000 * S.risk / 100; bal += r() < S.w / 100 ? risk * S.payoff : -risk; if (bal > peak) peak = bal; if (peak - bal >= 10000) touched = true; path.push(bal); } paths.push(path); ends.push(bal); if (touched) hit++; if (bal > 100000) win++; }
      var all = [].concat.apply([], paths), mn = Math.min.apply(null, all), mx = Math.max.apply(null, all), rg = (mx - mn) || 1, W = 640, H = 220;
      var svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%' });
      var y = function (v) { return H - 8 - (v - mn) / rg * (H - 16); };
      svg.appendChild(svgEl('line', { x1: 0, x2: W, y1: y(100000), y2: y(100000), stroke: '#2a3242', 'stroke-dasharray': '4 4' }));
      svg.appendChild(svgEl('line', { x1: 0, x2: W, y1: y(90000), y2: y(90000), stroke: '#ff6f6f', 'stroke-dasharray': '2 4', opacity: .6 }));
      paths.forEach(function (pt, k) { svg.appendChild(svgEl('polyline', { points: pt.map(function (v, i) { return (i / S.n * W).toFixed(1) + ',' + y(v).toFixed(1); }).join(' '), fill: 'none', stroke: ends[k] >= 100000 ? '#35e0b0' : '#ff6f6f', 'stroke-width': 1.2, opacity: .6 })); });
      box.textContent = ''; box.appendChild(svg);
      ends.sort(function (a, b) { return a - b; }); med.textContent = money(ends[Math.floor(N / 2)]); dd.textContent = Math.round(hit / N * 100) + '%'; prof.textContent = Math.round(win / N * 100) + '%';
    }
    slider(ctl, 'Win rate', 20, 80, 1, S.w, function (x) { return x + '%'; }, function (x) { S.w = x; run(); });
    slider(ctl, 'Average win in R', 0.5, 4, 0.1, S.payoff, function (x) { return x.toFixed(1) + 'R'; }, function (x) { S.payoff = x; run(); });
    slider(ctl, 'Risk per trade', 0.1, 4, 0.1, S.risk, function (x) { return x.toFixed(1) + '%'; }, function (x) { S.risk = x; run(); });
    var b = el('button', 'lab-btn', 'New luck'); b.type = 'button'; b.addEventListener('click', function () { S.seed++; run(); }); ctl.appendChild(b);
    host.appendChild(el('p', 'lab-note', '40 simulated runs of ' + S.n + ' trades. Same edge, bigger bets: watch more paths cross the red -10% line. Risk per trade is the one dial that decides whether a prop-firm account survives.'));
    run();
  }

  /* -------- Lab 3: trailing drawdown game -------- */
  function lab3(host) {
    var start = 50000, dd = 2500, bal, peak, floor, over;
    var panel = el('div'), btns = el('div', 'lab-ctls'), msg = el('p', 'lab-note'); host.appendChild(panel); host.appendChild(btns); host.appendChild(msg);
    function draw() {
      panel.textContent = '';
      [['Balance', money(bal)], ['Highest balance', money(peak)], ['Drawdown floor', money(floor)], ['Remaining loss allowance', money(bal - floor)]].forEach(function (r) { var d = el('div', 'lab-stat'); d.appendChild(el('span', null, r[0])); d.appendChild(el('b', null, r[1])); panel.appendChild(d); });
      panel.className = 'lab-stats';
      var bar = el('div', 'lab-bar'), i = el('i'); i.style.width = Math.max(0, Math.min(100, (bal - floor) / dd * 100)) + '%'; i.style.background = (bal - floor) / dd < .35 ? '#ff6f6f' : '#35e0b0'; bar.appendChild(i); panel.appendChild(bar);
    }
    function reset() { bal = peak = start; floor = start - dd; over = false; msg.textContent = 'Starting balance ' + money(start) + ', trailing drawdown ' + money(dd) + '. Click Win / Lose and watch the floor.'; draw(); }
    function trade(amt) {
      if (over) return; bal += amt; if (bal > peak) peak = bal; floor = Math.min(start, peak - dd); if (floor < start - dd) floor = start - dd; draw();
      if (bal <= floor) { over = true; msg.textContent = 'Account breached! Your peak was ' + money(peak) + ', so the floor had trailed up to ' + money(floor) + '. Banked profit is NOT extra room — it is what the floor is chasing.'; }
      else if (floor >= start) msg.textContent = 'The floor has locked at your starting balance — from here the account can no longer be breached by giving back profits above the start.'; else msg.textContent = 'Floor ' + money(floor) + '. Remaining allowance ' + money(bal - floor) + '.';
    }
    [['Win +$500', 500], ['Win +$1,000', 1000], ['Lose −$500', -500], ['Lose −$1,000', -1000]].forEach(function (b) { var x = el('button', 'lab-btn', b[0]); x.type = 'button'; x.addEventListener('click', function () { trade(b[1]); }); btns.appendChild(x); });
    var rs = el('button', 'lab-btn', 'Reset'); rs.type = 'button'; rs.addEventListener('click', reset); btns.appendChild(rs); reset();
  }

  /* -------- Lab 4: overfit your own strategy -------- */
  function sma(a, n) { var out = new Array(a.length).fill(null), s = 0; for (var i = 0; i < a.length; i++) { s += a[i]; if (i >= n) s -= a[i - n]; if (i >= n - 1) out[i] = s / n; } return out; }
  function backtest(px, f, s, from, to) { var F = sma(px, f), Sl = sma(px, s), eq = 0; for (var i = Math.max(from, s); i < to - 1; i++) { if (F[i] === null || Sl[i] === null) continue; var pos = F[i] > Sl[i] ? 1 : -1; eq += pos * (px[i + 1] - px[i]); } return eq; }
  function lab4(host) {
    var seed = 11, tally = { runs: 0, worse: 0 };
    var btn = el('button', 'lab-btn', 'Generate random prices and optimize'); btn.type = 'button'; var out = el('div', 'lab-stats'), box = el('div'), note = el('p', 'lab-note'); host.appendChild(btn); host.appendChild(out); host.appendChild(box); host.appendChild(note);
    btn.addEventListener('click', function () {
      seed++; var r = rng(seed * 977), px = [100]; for (var i = 1; i < 1500; i++) px.push(px[i - 1] + gauss(r)); var split = 900, best = null, tested = 0;
      for (var f = 3; f <= 30; f += 3) for (var s = 20; s <= 120; s += 10) { if (f >= s) continue; tested++; var v = backtest(px, f, s, 0, split); if (!best || v > best.v) best = { f: f, s: s, v: v }; }
      var oos = backtest(px, best.f, best.s, split, px.length);
      tally.runs++; if (oos < best.v) tally.worse++;
      out.textContent = ''; out.className = 'lab-stats';
      [['Combinations tried', tested], ['Best in-sample profit (points)', best.v.toFixed(1)], ['Best parameters', 'fast ' + best.f + ' / slow ' + best.s], ['Same parameters out-of-sample', oos.toFixed(1)]].forEach(function (a, k) { var d = el('div', 'lab-stat'); d.appendChild(el('span', null, a[0])); var b = el('b', null, String(a[1])); if (k === 3) b.className = oos >= 0 ? 'pos' : 'neg'; d.appendChild(b); out.appendChild(d); });
      var svg = svgEl('svg', { viewBox: '0 0 640 150', width: '100%' }), mn = Math.min.apply(null, px), mx = Math.max.apply(null, px);
      svg.appendChild(svgEl('polyline', { points: px.map(function (v, i) { return (i / 1499 * 640).toFixed(1) + ',' + (146 - (v - mn) / (mx - mn) * 140).toFixed(1); }).join(' '), fill: 'none', stroke: '#8a93a6', 'stroke-width': 1.3 }));
      svg.appendChild(svgEl('rect', { x: split / 1500 * 640, y: 0, width: 640 - split / 1500 * 640, height: 150, fill: 'rgba(240,180,41,.10)' })); var t = svgEl('text', { x: split / 1500 * 640 + 6, y: 14, fill: '#f0b429', 'font-size': 11 }); t.textContent = 'out-of-sample'; svg.appendChild(t);
      box.textContent = ''; box.appendChild(svg);
      note.textContent = 'This is pure random noise — there is no edge to find. Yet the best of ' + tested + ' combinations looks good in-sample. Out of sample it was worse than in-sample ' + tally.worse + ' of ' + tally.runs + ' times you tried. That gap is exactly what walk-forward, CPCV and the T58 robustness score exist to catch.';
    });
    note.textContent = 'Press the button: the lab invents random prices, searches moving-average pairs on the first 60%, then tests the winner on the unseen 40%.';
  }

  /* -------- Lab 5: gamma regime explorer -------- */
  function lab5(host) {
    var PUT = 6700, FLIP = 6780, CALL = 6900, S = { spot: 6820 };
    var ctl = el('div', 'lab-ctls'), box = el('div'), out = el('div', 'lab-stats'), note = el('p', 'lab-note'); host.appendChild(ctl); host.appendChild(box); host.appendChild(out); host.appendChild(note);
    var reg = stat(out, 'Regime', 'net_gex'), dealers = stat(out, 'Dealer hedging'), fam = stat(out, 'Strategy ideas');
    function draw() {
      var W = 640, H = 90, lo = 6640, hi = 6960, x = function (v) { return (v - lo) / (hi - lo) * W; }, svg = svgEl('svg', { viewBox: '0 0 ' + W + ' ' + H, width: '100%' });
      svg.appendChild(svgEl('rect', { x: 0, y: 30, width: x(FLIP), height: 24, fill: 'rgba(255,111,111,.22)' })); svg.appendChild(svgEl('rect', { x: x(FLIP), y: 30, width: W - x(FLIP), height: 24, fill: 'rgba(53,224,176,.20)' }));
      [[PUT, 'Put wall', '#ff6f6f'], [FLIP, 'Gamma flip', '#f0b429'], [CALL, 'Call wall', '#35e0b0']].forEach(function (l) { svg.appendChild(svgEl('line', { x1: x(l[0]), x2: x(l[0]), y1: 22, y2: 62, stroke: l[2], 'stroke-width': 2 })); var t = svgEl('text', { x: x(l[0]), y: 16, fill: l[2], 'font-size': 11, 'text-anchor': 'middle' }); t.textContent = l[1] + ' ' + l[0]; svg.appendChild(t); });
      svg.appendChild(svgEl('circle', { cx: x(S.spot), cy: 42, r: 8, fill: '#e7ebf2' })); var st = svgEl('text', { x: x(S.spot), y: 82, fill: '#e7ebf2', 'font-size': 12, 'text-anchor': 'middle' }); st.textContent = 'spot ' + S.spot; svg.appendChild(st);
      box.textContent = ''; box.appendChild(svg);
      var pos = S.spot >= FLIP;
      reg.textContent = pos ? 'Positive gamma' : 'Negative gamma'; reg.className = pos ? 'pos' : 'neg';
      dealers.textContent = pos ? 'Sell rallies, buy dips (dampens)' : 'Sell falls, buy rises (amplifies)';
      fam.textContent = pos ? 'Wall fade, pin fade' : 'Breakout, flip trend, convexity';
      note.textContent = pos ? 'Above the flip, dealers are long gamma: hedging pushes price back toward where it was, so ranges and fades work better. Near the call wall (' + CALL + ') rallies tend to stall.' : 'Below the flip, dealers are short gamma: hedging adds fuel to the move, so ranges break and trends run further. Near the put wall (' + PUT + ') selling can accelerate or snap back.';
    }
    slider(ctl, 'Spot price', 6650, 6950, 5, S.spot, function (x) { return String(x); }, function (x) { S.spot = x; draw(); }); draw();
  }

  /* -------- Lab 6: knowledge check -------- */
  var QUIZ = [
    ['A strategy has a 40% win rate, average win $300 and average loss $150. Is expectancy positive?', ['Yes, +$30 per trade', 'No, −$30 per trade', 'Zero'], 0, '0.4×300 − 0.6×150 = 120 − 90 = +$30 per trade.'],
    ['Your trailing drawdown is $2,500 and your peak balance rises by $1,000. The floor...', ['Stays where it was', 'Rises by $1,000', 'Falls by $1,000'], 1, 'A trailing floor follows the highest balance up (until it locks at the starting balance).'],
    ['You test 500 parameter combinations and the best has profit factor 2.4 in-sample. The most likely reason to distrust it?', ['Too many trades', 'Selection bias — best of many noisy tries', 'Profit factor is unreliable'], 1, 'The best of many random tries always looks good; only out-of-sample results tell you if it is real.'],
    ['Net gamma is negative and price breaks below the flip. Dealer hedging will most likely...', ['Dampen the move', 'Amplify the move', 'Have no effect'], 1, 'Short-gamma dealers sell into falls, feeding the move.'],
    ['Doubling risk per trade while keeping the same edge will most likely...', ['Double pass probability', 'Leave odds unchanged', 'Raise both profit speed and breach risk'], 2, 'More risk means faster progress to the target and a much higher chance of hitting drawdown limits.'],
    ['Which number says how many dollars you win per dollar you lose?', ['Win rate', 'Profit factor', 'Sharpe ratio'], 1, 'Profit factor = gross profit ÷ gross loss.'],
  ];
  function lab6(host) {
    var score = 0, answered = 0, sc = el('p', 'lab-note', 'Score: 0 / ' + QUIZ.length);
    QUIZ.forEach(function (q, qi) {
      var card = el('div', 'lab-q'); card.appendChild(el('b', null, (qi + 1) + '. ' + q[0])); var fb = el('div', 'lab-fb');
      q[1].forEach(function (opt, oi) { var b = el('button', 'lab-opt', opt); b.type = 'button'; b.addEventListener('click', function () {
        if (card.getAttribute('data-done')) return; card.setAttribute('data-done', '1'); answered++; var ok = oi === q[2]; if (ok) score++; b.classList.add(ok ? 'right' : 'wrong'); if (!ok) card.querySelectorAll('.lab-opt')[q[2]].classList.add('right');
        fb.textContent = (ok ? 'Correct. ' : 'Not quite. ') + q[3]; sc.textContent = 'Score: ' + score + ' / ' + QUIZ.length + (answered === QUIZ.length ? (score === QUIZ.length ? ' — perfect.' : ' — revisit the modules above for the ones you missed.') : ''); }); card.appendChild(b); });
      card.appendChild(fb); host.appendChild(card);
    }); host.appendChild(sc);
  }

  var LABS = [
    ['1. Edge calculator', 'Drag win rate and trade sizes: see expectancy, profit factor and the break-even win rate move together.', lab1],
    ['2. Risk-per-trade fan', 'Same edge, different bet size. Watch 40 possible futures fan out and cross the drawdown line.', lab2],
    ['3. Trailing-drawdown game', 'Win and lose trades by hand and see exactly how the prop-firm floor trails your peak.', lab3],
    ['4. Overfit a strategy on noise', 'Optimize a moving-average pair on random prices and watch it fail on unseen data.', lab4],
    ['5. Gamma regime explorer', 'Move price around the gamma flip and walls to see how dealer hedging changes the market.', lab5],
  ];
  function css() {
    var s = document.createElement('style');
    s.textContent = '.lab{border:1px solid var(--border);border-radius:12px;background:var(--panel-3);padding:14px;margin:0 0 14px}.lab h3{margin:0 0 2px;font-size:15px}.lab > p.desc{margin:0 0 10px;color:var(--text-muted);font-size:12.5px}' +
      '.lab-ctls{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;align-items:end;margin-bottom:10px}.lab-ctl label{display:block;font-size:12.5px;color:var(--text-muted);margin-bottom:4px}.lab-ctl input{width:100%}' +
      '.lab-stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:8px 0}.lab-stat{background:var(--panel-2);border:1px solid var(--border);border-radius:9px;padding:8px 10px}.lab-stat span{display:block;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--text-muted)}.lab-stat b{font-size:17px}' +
      '.lab .pos{color:var(--teal)}.lab .neg{color:var(--coral)}.lab-note{font-size:12.5px;color:var(--text-muted);margin:8px 0 0;line-height:1.55}.lab-btn{padding:9px 14px;border-radius:9px;border:1px solid var(--border-light);background:var(--panel-2);color:var(--text);font-weight:600;font-size:13px;cursor:pointer}.lab-btn:hover{border-color:var(--teal)}' +
      '.lab-bar{grid-column:1/-1;height:12px;border-radius:6px;background:var(--panel-2);overflow:hidden;border:1px solid var(--border)}.lab-bar i{display:block;height:100%;transition:width .2s}' +
      '.lab-q{margin:0 0 12px}.lab-opt{display:block;width:100%;text-align:left;margin:5px 0;padding:9px 12px;border-radius:8px;border:1px solid var(--border-light);background:var(--panel-2);color:var(--text);font-size:13px;cursor:pointer}.lab-opt.right{border-color:var(--teal);background:rgba(var(--teal-rgb), .14)}.lab-opt.wrong{border-color:var(--coral);background:rgba(255,111,111,.12)}.lab-fb{font-size:12.5px;color:var(--text-muted);margin-top:4px}';
    document.head.appendChild(s);
  }
  function labBox(l) {
    var box = el('div', 'lab'); box.appendChild(el('h3', null, l[0])); box.appendChild(el('p', 'desc', l[1])); var body = el('div'); box.appendChild(body); l[2](body); return box;
  }
  root.T58EduLabs = { labs: LABS, quiz: QUIZ, rng: rng, backtest: backtest, sma: sma,
    /* every lab, stacked: the Interactive labs tab */
    mount: function (host) {
      css(); host.textContent = '';
      LABS.forEach(function (l) { host.appendChild(labBox(l)); });
    },
    /* one lab (1-based number) inside the lesson it belongs to */
    mountOne: function (host, n) {
      var l = LABS[(parseInt(n, 10) || 0) - 1]; if (!l) return false;
      css(); host.textContent = ''; host.appendChild(labBox(l)); return true;
    },
    /* the knowledge check, on its own page */
    mountQuiz: function (host) { css(); host.textContent = ''; var box = el('div', 'lab'); lab6(box); host.appendChild(box); }
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = root.T58EduLabs;
})(typeof window !== 'undefined' ? window : globalThis);
