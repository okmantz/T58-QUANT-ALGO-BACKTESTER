"""
Plain-English definitions for every headline metric in the app. Powers the clickable-metric popups (static/t58-metrics.js):
click Profit Factor, Max Drawdown, T58 Score ... anywhere and get what it means, the formula, what goes into the
calculation, how to read it, and its history from the run log -- the same idea as the "why NOT READY" verdict popup.

`history_for(key)` reads app.reports.run_history and returns the per-run values of that metric (newest last) so the popup can
show a real historical breakdown (latest, best, worst, average, per-strategy rows) rather than a generic blurb.
"""
from __future__ import annotations

import re
from typing import Any, Optional

# history_field -> key in run_history rows (None = no per-run history on record)
METRICS: dict[str, dict[str, Any]] = {
    "profit_factor": {
        "title": "Profit Factor", "aliases": ["profit factor", "pf", "profit factor (pf)", "p.factor"],
        "meaning": "How many dollars the strategy made for every dollar it lost.",
        "formula": "Profit Factor = gross profit ÷ gross loss",
        "inputs": ["Gross profit: sum of all winning trades", "Gross loss: absolute sum of all losing trades"],
        "reading": ["Below 1.0 loses money.", "1.3–1.5 is a workable edge after costs.", "Above 2.0 on a small sample is often luck — check the trade count."],
        "history_field": "profit_factor", "unit": "x", "higher_is_better": True,
    },
    "max_drawdown": {
        "title": "Max Drawdown", "aliases": ["max drawdown", "max dd", "drawdown", "max drawdown %", "max drawdown (%)", "max dd %", "max dd (%)", "dd %", "max dd%", "drawdown %"],
        "meaning": "The deepest peak-to-trough fall in account equity.",
        "formula": "Max Drawdown = max over time of (running peak equity − current equity); in % = that ÷ the peak",
        "inputs": ["Equity curve after every trade", "Running peak of that curve"],
        "reading": ["Prop-firm trailing drawdowns are usually 4–10% — a strategy whose historical drawdown is close to that will bust accounts.", "Compare it to the firm's limit, not to your gut."],
        "history_field": "max_drawdown_pct", "unit": "%", "higher_is_better": False,
    },
    "win_rate": {
        "title": "Win Rate", "aliases": ["win rate", "win %", "win%", "win rate %", "win rate (%)"],
        "meaning": "Share of trades that closed in profit.",
        "formula": "Win Rate = winning trades ÷ total trades",
        "inputs": ["Count of winning trades", "Count of all closed trades"],
        "reading": ["Meaningless alone — a 35% win rate with 3R winners beats a 70% win rate with tiny winners.", "Read it together with average win / average loss."],
        "history_field": "win_rate", "unit": "%", "higher_is_better": True,
    },
    "expectancy": {
        "title": "Expectancy", "aliases": ["expectancy", "avg realized r", "avg r", "average r"],
        "meaning": "What an average trade is worth. In R it is measured in units of the amount you risked.",
        "formula": "Expectancy = win rate × average win − loss rate × average loss  (equivalently net profit ÷ trades)",
        "inputs": ["Win rate", "Average win", "Average loss", "Number of trades"],
        "reading": ["Must be positive after commissions and slippage.", "0.15R+ is meaningful; the T58 score treats 0.6R as a full mark."],
        "history_field": None, "unit": "", "higher_is_better": True,
    },
    "sharpe_ratio": {
        "title": "Sharpe Ratio", "aliases": ["sharpe", "sharpe ratio", "sharpe (ann.)"],
        "meaning": "Return per unit of volatility — how smooth the equity curve is.",
        "formula": "Sharpe = mean(periodic return) ÷ standard deviation(periodic return) × √(periods per year)",
        "inputs": ["Periodic returns of the equity curve", "Annualisation factor"],
        "reading": ["Above 1 is decent, above 2 is excellent, but only trust it with many trades."],
        "history_field": "sharpe_ratio", "unit": "", "higher_is_better": True,
    },
    "net_profit": {
        "title": "Net Profit", "aliases": ["net profit", "total p&l", "total pnl", "net p&l", "net p/l", "net pnl", "net profit ($)", "total p/l"],
        "meaning": "Total money made after all trades, commissions and slippage.",
        "formula": "Net Profit = sum of every closed trade's P&L (after costs)",
        "inputs": ["Every closed trade's P&L", "Commission and slippage per trade"],
        "reading": ["Size-dependent — compare strategies on profit factor / expectancy, not raw dollars."],
        "history_field": "net_profit", "unit": "$", "higher_is_better": True,
    },
    "t58_score": {
        "title": "T58 Score", "aliases": ["t58 score", "t58 scorecard", "score", "t58", "t58 score (0-100)"],
        "meaning": "One 0–100 number that blends every trusted signal: prop-firm odds, payout odds, ruin risk, out-of-sample stability, expectancy and drawdown.",
        "formula": "Score = Σ(weight × component) ÷ Σ(weights present). Monte Carlo family capped at 30 pts; walk-forward/CPCV, parameter stability, expectancy, drawdown and parsimony fill the rest.",
        "inputs": ["Pass probability (lower confidence bound)", "First-payout probability", "Risk of ruin", "Funded survival", "Walk-forward / CPCV stability", "Parameter stability", "Expectancy lower bound", "Drawdown"],
        "reading": ["READY needs every hard gate passed AND a score ≥ 75.", "85+ is Elite. Open a NOT READY tag to see the exact gates that failed."],
        "history_field": None, "unit": "/100", "higher_is_better": True,
    },
    "robustness": {
        "title": "Robustness Score", "aliases": ["robustness", "robustness score"],
        "meaning": "How well the edge survives data it was not tuned on.",
        "formula": "Mean of: best walk-forward/CPCV efficiency, (100 − PBO %), and the last T58 score — whichever exist.",
        "inputs": ["Walk-forward / CPCV efficiency", "Probability of backtest overfitting (PBO)", "T58 score"],
        "reading": ["Below 50: likely curve-fit.", "Blank means no validation has been run yet."],
        "history_field": None, "unit": "/100", "higher_is_better": True,
    },
    "eval_pass_probability": {
        "title": "Evaluation Pass Probability", "aliases": ["pass rate", "pass probability", "eval pass", "eval pass %", "prop pass rate", "probability of passing", "eval %", "eval", "pass %", "p(pass)", "eval-pass %", "eval pass probability", "pass prob"],
        "meaning": "Chance that one purchased evaluation attempt reaches the profit target without breaking a rule.",
        "formula": "Share of Monte Carlo attempts (resampled from your trades) that hit the target before breaching daily loss / drawdown, meeting the minimum days and the consistency rule.",
        "inputs": ["Your trade list (win rate, payoff, frequency)", "The firm's profit target, daily loss limit, drawdown, min days, consistency rule", "Number of simulated attempts"],
        "reading": ["Use the lower confidence bound, not the point estimate.", "Changing risk per trade moves this number faster than anything else."],
        "history_field": "eval_pass_probability", "unit": "%", "higher_is_better": True,
    },
    "first_payout_probability": {
        "title": "Payout Probability", "aliases": ["payout rate", "payout probability", "payout sim rate", "first payout", "first payout %", "payout simulation rate", "payout %", "payout", "p(payout)", "payout prob", "first-payout %", "first payout probability"],
        "meaning": "Chance an attempt both passes the evaluation AND survives to a first payout.",
        "formula": "P(pass) × P(reach payout threshold without breaching | funded)",
        "inputs": ["Evaluation pass probability", "Funded-stage rules (payout threshold, buffer, winning days, consistency)", "Your trade distribution"],
        "reading": ["It is the product of two hurdles, so it is always ≤ pass probability."],
        "history_field": "first_payout_probability", "unit": "%", "higher_is_better": True,
    },
    "oos_pct": {
        "title": "OOS (Out-of-Sample Score)", "aliases": ["oos", "oos %", "oos validation", "out-of-sample", "out of sample", "oos score", "oos efficiency"],
        "meaning": "How well the strategy holds up on data it was NOT tuned on. A strategy that only works on the data it was fitted to is curve-fit; a high OOS says the edge survives on unseen data.",
        "formula": "Best available validation result, as a percent: walk-forward / CPCV efficiency × 100 (out-of-sample performance ÷ in-sample performance), or 100 − PBO (probability of backtest overfitting) when that is what was run.",
        "inputs": ["Walk-forward optimisation, CPCV or walk-forward GA results", "Probability of backtest overfitting (PBO)", "Shows -- until at least one validation has been run"],
        "reading": ["Below 50: likely curve-fit.", "Champion Candidate needs 50 or more.", "Blank (--) means no validation has been run yet -- run Walk-forward or CPCV from the Strategy Lab."],
        "history_field": None, "unit": "%", "higher_is_better": True,
    },
    "promotion_stage": {
        "title": "Stage", "aliases": ["promotion stage"],
        "meaning": "Where the strategy is on the road from idea to something you would trade for real. It only moves forward when it passes that stage's gate.",
        "formula": "Candidate \u2192 Validated \u2192 Champion Candidate \u2192 Forward Testing \u2192 Production Ready. Champion Candidate needs Eval \u2265 60, Payout \u2265 55, OOS \u2265 50 and a verdict that is not NOT READY. Forward Testing needs a READY Champion Check and Eval \u2265 70. Production Ready needs at least 14 days of forward testing.",
        "inputs": ["Eval pass probability", "Payout probability", "OOS score", "Champion Check verdict", "Days of forward testing"],
        "reading": ["Open the Strategy Library to promote a strategy when it qualifies.", "A strategy can never skip a stage."],
        "history_field": None, "unit": "", "higher_is_better": True,
    },
    "risk_of_ruin": {
        "title": "Risk of Ruin", "aliases": ["risk of ruin", "ruin", "ruin %", "risk of ruin %", "ror"],
        "meaning": "Probability the account breaches a loss rule within the horizon (90 days by default).",
        "formula": "Share of simulated accounts that hit the daily-loss or drawdown floor within the ruin horizon",
        "inputs": ["Trade distribution", "Drawdown / daily loss limits", "Risk per trade"],
        "reading": ["Under 10% is comfortable; above 25% means the sizing is too aggressive for the rules."],
        "history_field": "risk_of_ruin_pct", "unit": "%", "higher_is_better": False,
    },
    "avg_win": {
        "title": "Average Win", "aliases": ["avg win", "average win"],
        "meaning": "Mean profit of winning trades.", "formula": "Average Win = gross profit ÷ winning trades",
        "inputs": ["Winning trades"], "reading": ["Compare with average loss: payoff ratio = avg win ÷ |avg loss|."],
        "history_field": None, "unit": "$", "higher_is_better": True,
    },
    "avg_loss": {
        "title": "Average Loss", "aliases": ["avg loss", "average loss"],
        "meaning": "Mean loss of losing trades.", "formula": "Average Loss = gross loss ÷ losing trades",
        "inputs": ["Losing trades"], "reading": ["If it is much larger than your planned risk, stops are slipping or not being honoured."],
        "history_field": None, "unit": "$", "higher_is_better": False,
    },
    "trailing_drawdown": {
        "title": "Trailing Drawdown", "aliases": ["trailing drawdown", "drawdown floor"],
        "meaning": "A loss limit that rises with your best balance and never falls: the floor trails the peak.",
        "formula": "Floor = highest balance reached − max drawdown allowance (stops rising once it reaches the starting balance on most firms)",
        "inputs": ["Peak balance", "Max drawdown allowance"],
        "reading": ["Remaining loss allowance = current balance − floor.", "Winning raises the floor, so banked profit is not extra room."],
        "history_field": None, "unit": "$", "higher_is_better": False,
    },
    "daily_loss_limit": {
        "title": "Daily Loss Limit", "aliases": ["daily loss limit", "daily loss"],
        "meaning": "The most you may lose in one trading day before the account is breached.",
        "formula": "Limit = account size × daily loss %", "inputs": ["Account size", "Daily loss %", "Today's realised P&L"],
        "reading": ["Two or three max-risk losses in a row can use the whole allowance."],
        "history_field": None, "unit": "$", "higher_is_better": False,
    },
    "consistency_rule": {
        "title": "Consistency Rule", "aliases": ["consistency", "consistency rule"],
        "meaning": "No single day may make up more than a set share of your total profit.",
        "formula": "Best day profit ÷ total profit ≤ rule % (e.g. 30%)", "inputs": ["Best single-day profit", "Total profit"],
        "reading": ["It punishes one lucky home-run day; smoother days pass it."],
        "history_field": None, "unit": "%", "higher_is_better": False,
    },
    "net_gex": {
        "title": "Net Gamma Exposure (GEX)", "aliases": ["net gex", "gex", "gamma exposure"],
        "meaning": "Dollar delta that option dealers must trade for a 1% move: positive = they sell rallies / buy dips (calming), negative = they chase moves (amplifying).",
        "formula": "GEX = Σ over strikes of gamma × open interest × 100 × spot² × 1%  (calls +, puts −)",
        "inputs": ["Option gamma", "Open interest per strike", "Spot price"],
        "reading": ["Sign matters more than size.", "Free CBOE data is delayed ~15 min and describes today only."],
        "history_field": None, "unit": "$", "higher_is_better": None,
    },
    "gamma_flip": {
        "title": "Gamma Flip", "aliases": ["gamma flip", "zero gamma"],
        "meaning": "The price where net dealer gamma crosses zero. Above it dealers dampen moves; below it they amplify them.",
        "formula": "Strike where cumulative net GEX (low→high strike) changes sign nearest spot, linearly interpolated",
        "inputs": ["Per-strike net GEX"], "reading": ["A close below the flip in negative gamma favours trend/expansion tactics."],
        "history_field": None, "unit": "", "higher_is_better": None,
    },
    "call_wall": {
        "title": "Call Wall", "aliases": ["call wall"],
        "meaning": "The strike above spot with the largest call gamma — where dealer hedging tends to cap rallies.",
        "formula": "argmax over strikes ≥ spot of call GEX", "inputs": ["Call open interest", "Call gamma"],
        "reading": ["Breaks through it in negative gamma can accelerate."], "history_field": None, "unit": "", "higher_is_better": None,
    },
    "put_wall": {
        "title": "Put Wall", "aliases": ["put wall"],
        "meaning": "The strike below spot with the largest put gamma — a common support where dealer hedging cushions selloffs.",
        "formula": "argmax over strikes ≤ spot of |put GEX|", "inputs": ["Put open interest", "Put gamma"],
        "reading": ["Losing it can unleash selling."], "history_field": None, "unit": "", "higher_is_better": None,
    },
}

_ALIAS: dict[str, str] = {}
for _k, _v in METRICS.items():
    _ALIAS[_k] = _k
    for _a in _v.get("aliases", []):
        _ALIAS[_a.lower()] = _k


def resolve(label_or_key: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", str(label_or_key or "").strip().lower())
    return _ALIAS.get(s) or _ALIAS.get(s.replace("_", " "))


def definition(key: str) -> Optional[dict]:
    k = resolve(key)
    if not k:
        return None
    d = dict(METRICS[k])
    d["key"] = k
    return d


def all_labels() -> dict[str, str]:
    """alias (lower-case) -> metric key, for the client-side auto-binder."""
    return dict(_ALIAS)


def history_for(key: str, limit: int = 200) -> dict:
    k = resolve(key)
    d = METRICS.get(k or "")
    out: dict[str, Any] = {"key": k, "rows": [], "summary": None}
    if not d or not d.get("history_field"):
        return out
    from app.reports import run_history

    field = d["history_field"]
    rows = []
    for r in run_history.load_runs():
        v = r.get(field)
        if isinstance(v, (int, float)) and v == v:
            rows.append({"date": str(r.get("timestamp", ""))[:10], "strategy": r.get("strategy_name"),
                         "instrument": r.get("instrument"), "value": round(float(v), 3)})
    rows.sort(key=lambda x: x["date"])
    rows = rows[-limit:]
    out["rows"] = rows
    if rows:
        vals = [x["value"] for x in rows]
        hib = d.get("higher_is_better", True)
        out["summary"] = {"count": len(vals), "latest": vals[-1], "average": round(sum(vals) / len(vals), 3),
                          "best": max(vals) if hib is not False else min(vals),
                          "worst": min(vals) if hib is not False else max(vals)}
    return out
