/* T58 prop-firm evaluation simulator (2026-10-17).
 * A visual, interactive second version of the replay: pick a firm's rules and a strategy's edge, drag the risk-per-trade
 * slider and the whole picture changes at once -- profit-target progress, daily-loss and trailing-drawdown gauges, the
 * current drawdown floor and remaining loss allowance, trading days, consistency progress, and the probabilities of
 * passing / failing / reaching a payout milestone (a 500-path Monte Carlo that re-runs on every slider move).
 *
 * The engine (T58PropSim.simulatePath / monteCarlo) is pure and seeded: the same inputs give the same answer, and a
 * slider change reuses the same random draws, so the odds move smoothly instead of jittering.
 * Rules modelled: profit target, daily loss limit, max drawdown (trailing from the intrabar peak, locking at the starting
 * balance, or static), minimum trading days, consistency rule (best day <= X% of total profit), and a payout milestone
 * (extra profit within N funded days without a breach). It is a model of the rules, not a firm's live rulebook.
 */
(function (root) {
  'use strict';
  function rng(seed) { var a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; var t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  function newState(p) {
    var start = p.account;
    return { day: 0, balance: start, peak: start, floor: start - p.account * p.maxDdPct / 100, dayPnl: 0, bestDay: 0, dayPnls: [], status: 'running', reason: '', passed: false, trades: 0, passDay: null };
  }
  function floorFor(p, st) {
    var base = p.account - p.account * p.maxDdPct / 100;
    if (p.ddType === 'static') return base;
    return Math.min(p.account, st.peak - p.account * p.maxDdPct / 100) < base ? base : Math.min(p.account, st.peak - p.account * p.maxDdPct / 100);
  }
  function profitOf(st, p) { return st.balance - p.account; }
  function consistencyOk(p, st) {
    var prof = profitOf(st, p); if (p.consistencyPct == null || prof <= 0) return true;
    return st.bestDay / prof <= p.consistencyPct / 100 + 1e-9;
  }
  function consistencyUsed(p, st) {
    var prof = profitOf(st, p); if (prof <= 0) return 0; return st.bestDay / prof * 100;
  }
  /* advance one trading day; rand() supplies the draws */
  function stepDay(p, st, rand) {
    if (st.status !== 'running') return st;
    st.day += 1; st.dayPnl = 0;
    var risk = p.account * p.riskPct / 100, dailyLimit = p.account * p.dailyLossPct / 100;
    for (var i = 0; i < p.tradesPerDay; i++) {
      var win = rand() < p.winRate, pnl = win ? risk * p.payoffR : -risk;
      st.balance += pnl; st.dayPnl += pnl; st.trades += 1;
      if (st.balance > st.peak) st.peak = st.balance;
      st.floor = floorFor(p, st);
      if (st.balance <= st.floor) { st.status = 'failed'; st.reason = 'Drawdown breached'; break; }
      if (st.dayPnl <= -dailyLimit) { st.status = 'failed'; st.reason = 'Daily loss limit hit'; break; }
    }
    st.dayPnls.push(st.dayPnl);
    if (st.dayPnl > st.bestDay) st.bestDay = st.dayPnl;
    if (st.status === 'running' && !st.passed && profitOf(st, p) >= p.account * p.targetPct / 100 && st.day >= p.minDays && consistencyOk(p, st)) {
      st.passed = true; st.passDay = st.day; st.status = 'passed'; st.reason = 'Profit target reached';
    }
    if (st.status === 'running' && st.day >= p.maxDays) { st.status = 'timeout'; st.reason = 'Ran out of days'; }
    return st;
  }
  /* after passing: funded phase until the payout milestone, a breach, or fundedDays */
  function fundedPhase(p, st, rand) {
    var fs = { balance: p.account, peak: p.account, floor: p.account - p.account * p.maxDdPct / 100, dayPnl: 0, day: 0 };
    var risk = p.account * p.riskPct / 100, dailyLimit = p.account * p.dailyLossPct / 100, goal = p.account * p.payoutPct / 100;
    for (var d = 0; d < p.fundedDays; d++) {
      fs.dayPnl = 0;
      for (var i = 0; i < p.tradesPerDay; i++) {
        var pnl = rand() < p.winRate ? risk * p.payoffR : -risk;
        fs.balance += pnl; fs.dayPnl += pnl; if (fs.balance > fs.peak) fs.peak = fs.balance;
        fs.floor = floorFor(p, { peak: fs.peak });
        if (fs.balance <= fs.floor || fs.dayPnl <= -dailyLimit) return false;
      }
      if (fs.balance - p.account >= goal && d + 1 >= (p.payoutMinDays || 1)) return true;
    }
    return false;
  }
  function monteCarlo(p, n, seed) {
    var res = { n: n, pass: 0, failDd: 0, failDaily: 0, timeout: 0, payout: 0, avgDays: 0 }, days = 0;
    for (var k = 0; k < n; k++) {
      var r = rng((seed || 1) * 1000003 + k * 7919), st = newState(p);
      while (st.status === 'running') stepDay(p, st, r);
      if (st.status === 'passed') { res.pass++; days += st.passDay; if (fundedPhase(p, st, r)) res.payout++; }
      else if (st.status === 'failed') { if (st.reason.indexOf('Daily') === 0) res.failDaily++; else res.failDd++; }
      else res.timeout++;
    }
    res.passPct = res.pass / n * 100; res.failDdPct = res.failDd / n * 100; res.failDailyPct = res.failDaily / n * 100;
    res.timeoutPct = res.timeout / n * 100; res.payoutPct = res.payout / n * 100; res.avgDays = res.pass ? days / res.pass : null;
    res.failPct = res.failDdPct + res.failDailyPct;
    return res;
  }
  function expectedR(p) { return p.winRate * p.payoffR - (1 - p.winRate); }

  var API = { rng: rng, newState: newState, stepDay: stepDay, monteCarlo: monteCarlo, floorFor: floorFor, consistencyOk: consistencyOk,
              consistencyUsed: consistencyUsed, expectedR: expectedR, fundedPhase: fundedPhase };
  root.T58PropSim = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;

  /* ------------------------------ UI ------------------------------ */
  if (typeof document === 'undefined') return;
  function el(tag, cls, text) { var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function money(n) { var s = Math.abs(Math.round(n)).toLocaleString(); return (n < 0 ? '-$' : '$') + s; }

  API.mount = function (host) {
    var P = { account: 100000, targetPct: 8, dailyLossPct: 5, maxDdPct: 10, ddType: 'trailing', consistencyPct: 30, minDays: 5, winRate: 0.45, payoffR: 1.8,
              tradesPerDay: 3, riskPct: 0.5, maxDays: 60, payoutPct: 4, fundedDays: 30, payoutMinDays: 5 };
    var presets = [], st = newState(P), timer = null, seed = 7, rand = rng(seed * 99 + 1);
    var css = document.createElement('style');
    css.textContent = '.ps{display:grid;grid-template-columns:minmax(260px,320px) 1fr;gap:14px}@media(max-width:900px){.ps{grid-template-columns:1fr}}' +
      '.ps label{display:block;font-size:12px;color:var(--text-muted);margin:9px 0 3px}.ps input[type=number],.ps select{width:100%;padding:9px 10px;border-radius:8px;border:1px solid var(--border-light);background:var(--panel-3);color:var(--text);font-size:14px}' +
      '.ps input[type=range]{width:100%}.ps .big{font-size:20px;font-weight:800}.ps .g{margin:10px 0}.ps .g .h{display:flex;justify-content:space-between;font-size:12.5px;margin-bottom:4px}' +
      '.ps .bar{position:relative;height:14px;border-radius:7px;background:var(--panel-3);overflow:hidden;border:1px solid var(--border)}.ps .bar i{position:absolute;left:0;top:0;bottom:0;border-radius:7px;transition:width .25s}' +
      '.ps .odds{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:8px;margin-top:10px}.ps .odd{background:var(--panel-3);border:1px solid var(--border);border-radius:10px;padding:9px 10px}.ps .odd span{display:block;font-size:10.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--text-muted)}.ps .odd b{font-size:22px}' +
      '.ps .btns{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0}.ps button{padding:9px 13px;border-radius:9px;border:1px solid var(--border-light);background:var(--panel-3);color:var(--text);font-weight:600;font-size:13px;cursor:pointer}.ps button.p{background:var(--teal);color:var(--on-teal,#04130e);border-color:var(--teal)}' +
      '.ps .status{padding:9px 12px;border-radius:9px;font-weight:700;margin:6px 0;border:1px solid var(--border-light)}.ps .status.passed{color:var(--teal);border-color:var(--teal)}.ps .status.failed{color:var(--coral);border-color:var(--coral)}.ps .status.timeout{color:var(--amber);border-color:var(--amber)}' +
      '.ps .row2{display:grid;grid-template-columns:1fr 1fr;gap:8px}.ps .pos{color:var(--teal)}.ps .neg{color:var(--coral)}.ps .muted{color:var(--text-muted);font-size:12px}';
    document.head.appendChild(css);
    host.textContent = '';
    var wrap = el('div', 'ps'), left = el('div'), right = el('div'); wrap.appendChild(left); wrap.appendChild(right); host.appendChild(wrap);

    function num(label, key, step, min, max, note) {
      var w = el('div'); w.appendChild(el('label', null, label)); var i = el('input'); i.type = 'number'; i.step = step; i.min = min; i.max = max; i.value = P[key];
      i.addEventListener('input', function () { var v = parseFloat(i.value); if (isFinite(v)) { P[key] = v; recompute(true); } }); w.appendChild(i); return w;
    }
    var firmSel = el('select'); firmSel.appendChild(Object.assign(el('option', null, 'Custom rules'), { value: '' }));
    left.appendChild(el('label', null, 'Prop-firm rules')); left.appendChild(firmSel);
    firmSel.addEventListener('change', function () {
      var p = presets.filter(function (x) { return x.key === firmSel.value; })[0]; if (!p) return;
      P.account = p.account_size; P.targetPct = p.target_pct; P.dailyLossPct = p.daily_loss_pct; P.maxDdPct = p.max_dd_pct; P.ddType = p.drawdown_type === 'static' ? 'static' : 'trailing';
      P.consistencyPct = p.consistency_pct; P.minDays = p.min_days || 1; if (p.payout_threshold_pct) P.payoutPct = p.payout_threshold_pct; buildInputs(); recompute(true);
    });
    var inputsBox = el('div'); left.appendChild(inputsBox);
    function buildInputs() {
      inputsBox.textContent = '';
      [['Account size ($)', 'account', 1000], ['Profit target (%)', 'targetPct', 0.5], ['Daily loss limit (%)', 'dailyLossPct', 0.5], ['Max drawdown (%)', 'maxDdPct', 0.5], ['Min trading days', 'minDays', 1], ['Consistency rule: best day ≤ (%)', 'consistencyPct', 5]]
        .forEach(function (r) { inputsBox.appendChild(num(r[0], r[1], r[2], 0, 10000000)); });
      var dl = el('label', null, 'Drawdown type'); var ds = el('select'); [['trailing', 'Trailing (floor rises with peak)'], ['static', 'Static (fixed floor)']].forEach(function (o) { var op = el('option', null, o[1]); op.value = o[0]; ds.appendChild(op); }); ds.value = P.ddType;
      ds.addEventListener('change', function () { P.ddType = ds.value; recompute(true); }); inputsBox.appendChild(dl); inputsBox.appendChild(ds);
      inputsBox.appendChild(el('h4', null, 'Strategy edge')).style.margin = '14px 0 0';
      /* Load edge from ANY saved strategy (manual JSON, Python, PineScript, MQL5). Rows with recorded results apply
         instantly; everything else (e.g. the preloaded, never-backtested strategies) is measured on the fly on a
         dataset the person can choose -- the server backtests it with the same engine every other tool uses. */
      inputsBox.appendChild(el('label', null, 'Load edge from a saved strategy'));
      var stratSel = el('select');
      stratSel.appendChild(Object.assign(el('option', null, 'Manual inputs'), { value: '' }));
      var typeNames = { python: 'Python', pinescript: 'PineScript', mql5: 'MQL5', manual: 'JSON / Manual' }, groups = {};
      (API._rows || []).forEach(function (r) {
        var g = groups[r.type]; if (!g) { g = groups[r.type] = el('optgroup'); g.label = typeNames[r.type] || r.type; stratSel.appendChild(g); }
        var has = r.win_rate != null && r.profit_factor != null;
        var o = el('option', null, r.name + (has ? ' (' + r.win_rate + '% win, PF ' + r.profit_factor + ')' : ' (not backtested yet — will measure)'));
        o.value = r.id; g.appendChild(o);
      });
      stratSel.value = API._pick || '';
      inputsBox.appendChild(stratSel);
      var dsSel = el('select'); dsSel.appendChild(Object.assign(el('option', null, 'Measure on: best matching stored dataset'), { value: '' }));
      ((API._ds && API._ds.groups) || []).forEach(function (g) {
        if (!g.files.length) return; var og = el('optgroup'); og.label = g.instrument || 'Data';
        g.files.forEach(function (f) { var o = el('option', null, f.name); o.value = f.full_name; og.appendChild(o); }); dsSel.appendChild(og);
      });
      dsSel.value = API._dsPick || '';
      var dsLab = el('label', null, 'Dataset used to measure the strategy'); inputsBox.appendChild(dsLab); inputsBox.appendChild(dsSel);
      var stat = el('div', 'muted'); stat.style.margin = '6px 0 0'; stat.textContent = API._status || ((API._rows || []).length ? (API._rows.length + ' saved strategies available.') : 'No saved strategies found — build one in the Strategy Library first.'); inputsBox.appendChild(stat);
      function measure(row, refresh) {
        stat.textContent = 'Backtesting ' + row.name + ' to measure its edge… (first run can take up to a minute)';
        stratSel.disabled = true;
        fetch('/api/prop-sim/strategy-edge', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ type: row.type, name: row.name, dataset: API._dsPick || (API._ds && API._ds.suggested && API._ds.suggested[row.id]) || '', refresh: !!refresh }) })
          .then(function (r) { return r.json(); })
          .then(function (d) {
            stratSel.disabled = false;
            if (!d.ok) { API._status = '⚠ ' + (d.error || 'Could not measure this strategy.'); buildInputs(); return; }
            P.winRate = Math.min(0.99, Math.max(0.01, d.win_rate / 100)); P.payoffR = d.payoff_r; P.tradesPerDay = Math.max(1, Math.round(d.trades_per_day));
            API._status = '✔ ' + d.strategy + ': ' + d.trades + ' trades on ' + d.dataset + ' (' + d.period + ') — ' + d.win_rate + '% win, avg win ' + d.payoff_r + 'R, ' + d.trades_per_day + ' trades/day, expectancy ' + d.expectancy_r + 'R' + (d.profit_factor ? ', PF ' + d.profit_factor : '') + (d.cached ? ' (saved result)' : '');
            buildInputs(); recompute(true);
          }).catch(function (e) { stratSel.disabled = false; API._status = '⚠ ' + e; buildInputs(); });
      }
      function pickedRow() { return (API._rows || []).filter(function (x) { return x.id === stratSel.value; })[0]; }
      stratSel.addEventListener('change', function () {
        API._pick = stratSel.value; var r = pickedRow(); if (!r) { API._status = ''; return; }
        if (r.win_rate != null && r.profit_factor != null && r.win_rate > 0) {
          P.winRate = r.win_rate / 100;
          P.payoffR = Math.max(0.2, Math.round(r.profit_factor * (1 - P.winRate) / P.winRate * 100) / 100);
          if (r.trades && r.curve && r.curve.length) { /* trades/day unknown from the recorded row -- keep current */ }
          API._status = 'Using the recorded results for ' + r.name + ' (' + (r.trades != null ? r.trades + ' trades, ' : '') + 'PF ' + r.profit_factor + '). Use “Re-measure” for fresh numbers on a dataset of your choice.';
          buildInputs(); recompute(true);
        } else { measure(r, false); }
      });
      dsSel.addEventListener('change', function () { API._dsPick = dsSel.value; var r = pickedRow(); if (r) measure(r, false); });
      var remeasure = el('button', null, 'Re-measure on this dataset'); remeasure.type = 'button'; remeasure.style.marginTop = '6px';
      remeasure.addEventListener('click', function () { var r = pickedRow(); if (r) measure(r, true); else stat.textContent = 'Pick a saved strategy first.'; });
      inputsBox.appendChild(remeasure);
      inputsBox.appendChild(num('Win rate (0–1)', 'winRate', 0.01, 0.01, 0.99)); inputsBox.appendChild(num('Avg win in R (loss = 1R)', 'payoffR', 0.1, 0.1, 20)); inputsBox.appendChild(num('Trades per day', 'tradesPerDay', 1, 1, 30));
      inputsBox.appendChild(num('Payout milestone: extra profit (%)', 'payoutPct', 0.5, 0.5, 50));
    }
    buildInputs();

    var riskLab = el('div', 'big'); var risk = el('input'); risk.type = 'range'; risk.min = 0.05; risk.max = 3; risk.step = 0.05; risk.value = P.riskPct;
    var rbox = el('div'); rbox.style.marginTop = '14px'; rbox.appendChild(el('label', null, 'Risk per trade (% of account) — drag me')); rbox.appendChild(riskLab); rbox.appendChild(risk); left.appendChild(rbox);
    risk.addEventListener('input', function () { P.riskPct = parseFloat(risk.value); recompute(true); });

    var statusBox = el('div'), gauges = el('div'), odds = el('div', 'odds'), btns = el('div', 'btns'), info = el('div', 'muted');
    right.appendChild(statusBox); right.appendChild(btns); right.appendChild(gauges); right.appendChild(el('h4', null, 'Probabilities (500 simulated attempts at this risk)')).style.margin = '14px 0 0'; right.appendChild(odds); right.appendChild(info);
    function mkBtn(t, cls, fn) { var b = el('button', cls, t); b.type = 'button'; b.addEventListener('click', fn); btns.appendChild(b); return b; }
    var playBtn = mkBtn('▶ Play a new attempt', 'p', function () { if (timer) { clearInterval(timer); timer = null; playBtn.textContent = '▶ Play a new attempt'; return; } if (st.status !== 'running') reset(); playBtn.textContent = '⏸ Pause'; timer = setInterval(function () { stepDay(P, st, rand); drawGauges(); if (st.status !== 'running') { clearInterval(timer); timer = null; playBtn.textContent = '▶ Play a new attempt'; } }, 450); });
    mkBtn('Step 1 day', '', function () { if (st.status !== 'running') reset(); stepDay(P, st, rand); drawGauges(); });
    mkBtn('Restart (new luck)', '', function () { seed += 1; reset(); });

    function gauge(label, valueText, pct, color, metric, sub) {
      var g = el('div', 'g'), h = el('div', 'h'), l = el('span', null, label); if (metric) l.setAttribute('data-metric', metric); h.appendChild(l); h.appendChild(el('b', null, valueText)); g.appendChild(h);
      var bar = el('div', 'bar'), i = el('i'); i.style.width = Math.max(0, Math.min(100, pct)) + '%'; i.style.background = color; bar.appendChild(i); g.appendChild(bar); if (sub) g.appendChild(el('div', 'muted', sub)); return g;
    }
    function drawGauges() {
      var prof = st.balance - P.account, target = P.account * P.targetPct / 100, dLimit = P.account * P.dailyLossPct / 100;
      statusBox.textContent = ''; var s = el('div', 'status ' + st.status, st.status === 'running' ? 'Day ' + st.day + ' — attempt in progress' : (st.status === 'passed' ? '✔ PASSED on day ' + st.passDay : (st.status === 'failed' ? '✖ FAILED — ' + st.reason : '⏱ ' + st.reason))); statusBox.appendChild(s);
      gauges.textContent = '';
      gauges.appendChild(el('div', 'big ' + (prof >= 0 ? 'pos' : 'neg'), money(st.balance))).appendChild(el('span', 'muted', '  balance (' + (prof >= 0 ? '+' : '') + money(prof) + ')'));
      gauges.appendChild(gauge('Profit target progress', money(Math.max(0, prof)) + ' / ' + money(target), prof / target * 100, 'var(--teal)', null, 'Needs ' + money(Math.max(0, target - prof)) + ' more'));
      var dayLoss = Math.max(0, -st.dayPnl); gauges.appendChild(gauge('Daily loss limit used', money(dayLoss) + ' / ' + money(dLimit), dayLoss / dLimit * 100, dayLoss / dLimit > 0.66 ? 'var(--coral)' : 'var(--amber)', 'daily_loss_limit', 'Remaining today: ' + money(Math.max(0, dLimit - dayLoss))));
      var allow = P.account * P.maxDdPct / 100, used = Math.max(0, st.peak - st.balance), floor = st.floor, remain = Math.max(0, st.balance - floor);
      gauges.appendChild(gauge('Trailing drawdown used', money(used) + ' / ' + money(allow), used / allow * 100, used / allow > 0.66 ? 'var(--coral)' : 'var(--amber)', 'trailing_drawdown', 'Current drawdown floor ' + money(floor) + ' · remaining loss allowance ' + money(remain)));
      gauges.appendChild(gauge('Trading days completed', st.day + ' / ' + P.minDays + ' min', st.day / Math.max(1, P.minDays) * 100, 'var(--blue,#6fa8ff)', null, st.day >= P.minDays ? 'Minimum days met' : (P.minDays - st.day) + ' more day(s) needed'));
      if (P.consistencyPct != null && P.consistencyPct > 0) { var cu = consistencyUsed(P, st); gauges.appendChild(gauge('Consistency rule (best day share of profit)', cu.toFixed(0) + '% / ' + P.consistencyPct + '% max', cu / P.consistencyPct * 100, cu > P.consistencyPct ? 'var(--coral)' : 'var(--teal)', 'consistency_rule', cu > P.consistencyPct ? 'One big day is too large a share of your profit — keep trading to dilute it' : 'Within the rule')); }
    }
    function oddCard(label, val, color, metric) { var d = el('div', 'odd'), l = el('span', null, label); if (metric) l.setAttribute('data-metric', metric); d.appendChild(l); var b = el('b', null, val); b.style.color = color; d.appendChild(b); return d; }
    var mcTimer = null;
    function drawOdds() {
      var r = monteCarlo(P, 500, seed); odds.textContent = '';
      odds.appendChild(oddCard('Pass', r.passPct.toFixed(0) + '%', 'var(--teal)', 'eval_pass_probability'));
      odds.appendChild(oddCard('Fail', r.failPct.toFixed(0) + '%', 'var(--coral)'));
      odds.appendChild(oddCard('Payout milestone', r.payoutPct.toFixed(0) + '%', 'var(--amber)', 'first_payout_probability'));
      odds.appendChild(oddCard('Time out', r.timeoutPct.toFixed(0) + '%', 'var(--text-muted)'));
      info.textContent = 'Fails split: ' + r.failDdPct.toFixed(0) + '% drawdown, ' + r.failDailyPct.toFixed(0) + '% daily loss. ' + (r.avgDays ? 'Passing attempts took ' + r.avgDays.toFixed(0) + ' trading days on average. ' : '') +
        'Expectancy ' + expectedR(P).toFixed(2) + 'R per trade (' + money(expectedR(P) * P.account * P.riskPct / 100) + ' at this risk). Payout milestone = +' + P.payoutPct + '% more profit within ' + P.fundedDays + ' funded days.';
    }
    function reset() { st = newState(P); rand = rng(seed * 99 + 1); drawGauges(); }
    function recompute(userInput) {
      riskLab.textContent = P.riskPct.toFixed(2) + '%  =  ' + money(P.account * P.riskPct / 100) + ' per trade';
      reset(); clearTimeout(mcTimer); mcTimer = setTimeout(drawOdds, userInput ? 40 : 0);
    }
    fetch('/api/prop-sim/presets').then(function (r) { return r.json(); }).then(function (d) { presets = d.presets || []; presets.forEach(function (p) { var o = el('option', null, p.label); o.value = p.key; firmSel.appendChild(o); }); }).catch(function () {});
    fetch('/api/library/control').then(function (r) { return r.json(); }).then(function (d) { API._rows = d.rows || []; buildInputs(); }).catch(function () {});
    fetch('/api/prop-sim/datasets').then(function (r) { return r.json(); }).then(function (d) { API._ds = d; buildInputs(); }).catch(function () {});
    recompute(false);
  };
})(typeof window !== 'undefined' ? window : globalThis);
