"""Prop-firm ACCOUNT LIFECYCLE simulation -- the question a prop trader actually needs answered.

Traditional statistics (win rate, profit factor, Sharpe, net profit) rate a strategy in a vacuum. A prop-firm account is not a
vacuum: it is a rules machine with a target, a trailing drawdown, a daily loss limit, a consistency rule, a minimum number of trading
days, a funded stage that starts from scratch, and payout gates. A strategy with 48% wins, $400 winners and $200 losers has positive
expectancy -- and may still fail a $50K evaluation 70% of the time because of how its losses cluster.

This module replays resampled copies of the strategy's real trade history through ``app.prop.simulator.simulate_account`` (the one rules
engine every other module already uses -- nothing is re-implemented here) and reports the lifecycle as separate, honest stages:

    EVALUATION STAGE
        P(reach target before breaching drawdown)      the order of events, not just "did it ever touch the target"
        P(violate the daily loss limit)                in any session day of the attempt, whatever the firm does about it
        P(satisfy the consistency rule | reached target)
        P(pass)                                        target AND min days AND consistency, no breach
        time to target                                 CALENDAR days (primary) and trading days, with P(pass within N days)
    FUNDED STAGE  (conditional on a funded account existing -- a different probability from passing the evaluation)
        P(survive N days | funded)                     Kaplan-Meier, so accounts still alive when the history ran out are CENSORED,
                                                       not counted as survivors and not dropped
        P(first payout | funded), P(second payout | funded), expected payout per funded account
    LIFECYCLE
        P(pass AND reach a first payout)               the headline objective: "will this configuration complete the firm's
                                                       evaluation AND qualify for a payout?"
        expected net per purchased attempt             fees and the trader's profit split included

WHY "8 days" AND "180 days" CAN DESCRIBE THE SAME STRATEGY
    ``AccountSimResult.days_to_pass`` is a count of TRADING days (days that contained a trade, weekends and idle days skipped), while the
    Full Pipeline's windows (90 / 180-day ruin, replay horizons) are CALENDAR days. A strategy that trades twice a week "passes in 8
    days" on one clock and takes ~30 calendar days on the other; a median taken only over the accounts that DID pass is faster again than
    the unconditional time. ``LifecycleResult.time_to_target`` therefore always reports calendar days first, trading days second, the
    unconditional ``p_pass_within_days`` curve, and states which clock each number is on.

``net_profit`` mode (``objective_mode``) is the alternative objective for LIVE accounts, where there is no evaluation to pass:
see ``net_profit_objective``. Prop-firm testing is the default everywhere; net profit is a selectable alternative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.prop.simulator import PropRules, precompute_day_structure, simulate_account

OBJECTIVE_PROP = "prop_lifecycle"          # default everywhere
OBJECTIVE_NET_PROFIT = "net_profit"        # selectable: live accounts, no evaluation
OBJECTIVE_MODES = (OBJECTIVE_PROP, OBJECTIVE_NET_PROFIT)
DEFAULT_OBJECTIVE_MODE = OBJECTIVE_PROP

OBJECTIVE_MODE_LABELS = {
    OBJECTIVE_PROP: "Prop-firm lifecycle (evaluation pass -> funded survival -> payout)",
    OBJECTIVE_NET_PROFIT: "Net profit (live account, costs included)",
}


def normalize_objective_mode(value) -> str:
    """Map any user/form value to a known objective mode; unknown or empty means the prop-firm default."""
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"prop": OBJECTIVE_PROP, "prop_firm": OBJECTIVE_PROP, "lifecycle": OBJECTIVE_PROP, "prop_lifecycle": OBJECTIVE_PROP,
               "net": OBJECTIVE_NET_PROFIT, "net_profit": OBJECTIVE_NET_PROFIT, "netprofit": OBJECTIVE_NET_PROFIT,
               "live": OBJECTIVE_NET_PROFIT, "profit": OBJECTIVE_NET_PROFIT}
    return aliases.get(key, DEFAULT_OBJECTIVE_MODE)


# --------------------------------------------------------------------------------------------------------- canonical dates
def canonical_trade_dates(trades) -> list:
    """The ONE definition of "when did this trade happen" for every account simulation: the trade's raw ENTRY timestamp.

    ``simulate_account`` buckets timestamps into session days (17:00 CT roll). Handing it ``entry_time.normalize()`` (midnight) -- as
    the Monte Carlo and survival engines did before 2026-11-07 -- throws away the time of day, so an evening-session trade (after the
    roll) is attributed to the PREVIOUS session day by those engines and to the NEXT one by the direct simulation, which moves
    daily-loss and consistency attribution. Every path now uses the raw timestamp so they bucket days identically."""
    return [pd.Timestamp(t.entry_time) for t in trades]


# -------------------------------------------------------------------------------------------------------------- statistics
def kaplan_meier_survival(durations, events, horizons) -> dict:
    """Kaplan-Meier survival S(h) for each horizon. ``events[i]`` is True when the account FAILED at ``durations[i]`` and False when it
    was still alive (censored) when its history ended. Returns {h: (S, n_at_risk_at_h)}; S is None when nobody was observed that long."""
    d = np.asarray(durations, dtype=float)
    e = np.asarray(events, dtype=bool)
    out: dict = {}
    if d.size == 0:
        return {int(h): (None, 0) for h in horizons}
    order = np.argsort(d, kind="stable")
    d, e = d[order], e[order]
    uniq = np.unique(d[e])
    surv = 1.0
    curve = []
    for t in uniq:
        at_risk = int(np.sum(d >= t))
        deaths = int(np.sum((d == t) & e))
        if at_risk > 0:
            surv *= 1.0 - deaths / at_risk
        curve.append((float(t), surv))
    for h in horizons:
        at_risk_h = int(np.sum(d >= h))
        if at_risk_h == 0:
            # Nobody was observed for h days: the survival is unknown beyond the follow-up, not 100%.
            out[int(h)] = (None, 0)
            continue
        s = 1.0
        for t, sv in curve:
            if t <= h:
                s = sv
            else:
                break
        out[int(h)] = (float(s), at_risk_h)
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion, in percent. (0, 100) when n == 0."""
    if n <= 0:
        return 0.0, 100.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, (centre - half) * 100.0), min(100.0, (centre + half) * 100.0)


def _pct(k: int, n: int) -> float:
    return 100.0 * k / n if n > 0 else 0.0


def _quantiles(values, qs=(25, 50, 75, 90)) -> dict:
    if not len(values):
        return {f"p{q}": None for q in qs}
    arr = np.asarray(values, dtype=float)
    return {f"p{q}": float(np.percentile(arr, q)) for q in qs}


# ------------------------------------------------------------------------------------------------------------------ config
@dataclass
class LifecycleConfig:
    n_simulations: int = 2_000
    method: str = "day_block_bootstrap"      # whole session days, like every other engine since the 2026-11-06 audit
    block_size: int = 5
    day_block_days: int = 1
    slippage_stress_pct: float = 1.0
    random_seed: int | None = 42
    pass_windows_days: tuple = (14, 30, 60, 90, 180)       # calendar days: P(pass within N days)
    funded_horizons_days: tuple = (30, 60, 90, 180)         # calendar days: P(survive N days | funded)
    evaluation_fee: float = 0.0
    reset_fee: float | None = None
    profit_split_pct: float = 80.0
    funding_approval_probability: float = 100.0              # firms that review passed evaluations by hand lose some of them

    def resolved_reset_fee(self) -> float:
        return float(self.evaluation_fee if self.reset_fee is None else self.reset_fee)


# ----------------------------------------------------------------------------------------------------------------- results
@dataclass
class EvaluationStage:
    n_simulations: int
    p_reach_target_before_breach: float
    p_pass: float
    p_pass_ci: tuple
    p_daily_loss_violation: float          # any session day of the attempt lost more than the daily limit (whatever the firm then does)
    p_fail_daily_loss: float               # the attempt actually ENDED on the daily-loss rule
    p_fail_max_drawdown: float
    p_fail_time_limit: float
    p_fail_other: float
    p_unresolved: float                    # history ran out before the attempt passed OR failed (censored, not counted as a pass)
    p_consistency_ok_given_target: float | None
    p_blocked_by_rules_after_target: float  # touched the target without breaching, yet could not pass (consistency / minimum days)
    time_to_target: dict                   # see _time_block
    p_pass_within_days: dict               # {N: % of ALL simulated attempts that passed within N calendar days}
    windows_covered: dict                  # {N: bool} False when the source history is shorter than N calendar days

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class FundedStage:
    n_funded: int
    p_funded: float                         # % of all attempts that passed AND cleared the firm's funding approval
    p_survive_given_funded: dict            # {N: % still alive after N calendar days of funded trading (Kaplan-Meier)}
    survival_at_risk: dict                  # {N: number of funded accounts actually observed for >= N days}
    p_fail_funded_daily_loss: float | None  # of funded accounts: share whose life ended on the daily-loss rule
    p_fail_funded_drawdown: float | None
    p_first_payout_given_funded: float | None
    p_second_payout_given_funded: float | None
    p_third_payout_given_funded: float | None
    expected_gross_payout_per_funded: float
    median_payout_amount: float | None
    time_to_first_payout: dict              # calendar days from the PASS to the first payout (funded accounts that got one)

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LifecycleResult:
    evaluation: EvaluationStage
    funded: FundedStage
    p_pass_and_first_payout: float          # THE headline objective, in percent of all purchased attempts
    p_pass_and_first_payout_ci: tuple
    expected_gross_payout_per_attempt: float
    expected_net_per_attempt: float         # trader's split of payouts minus the evaluation fee
    expected_fees_per_first_payout: float | None  # fee x expected attempts needed for one payout (geometric)
    expected_attempts_to_first_payout: float | None
    objective_mode: str
    objective_value: float                  # what an optimizer maximises for ``objective_mode`` (higher is better)
    n_trades: int
    history_calendar_days: int
    notes: list = field(default_factory=list)
    clocks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["evaluation"] = self.evaluation.to_dict()
        d["funded"] = self.funded.to_dict()
        d["headline"] = self.headline()
        return d

    def headline(self) -> str:
        e, f = self.evaluation, self.funded
        t = e.time_to_target.get("calendar", {})
        med = t.get("p50")
        return (f"P(pass)={e.p_pass:.1f}%  P(funded)={f.p_funded:.1f}%  "
                f"P(pass AND payout)={self.p_pass_and_first_payout:.1f}%  "
                + (f"median {med:.0f} calendar days to target" if med is not None else "no pass observed"))


def _time_block(calendar_days, trading_days) -> dict:
    return {"calendar": {**_quantiles(calendar_days), "n": len(calendar_days)},
            "trading": {**_quantiles(trading_days), "n": len(trading_days)},
            "basis": "among attempts that passed; use p_pass_within_days for the unconditional curve"}


# ------------------------------------------------------------------------------------------------------------- the engine
def _day_sums(pnls: np.ndarray, day_idx, n_days: int) -> np.ndarray:
    sums = np.zeros(n_days, dtype=float)
    np.add.at(sums, np.asarray(day_idx, dtype=int), pnls)
    return sums


def run_lifecycle(trades, rules: PropRules, cfg: LifecycleConfig | None = None, objective_mode: str = DEFAULT_OBJECTIVE_MODE) -> LifecycleResult:
    """Resample ``trades`` and replay each path through ``simulate_account``; summarise the lifecycle (module docstring).

    ``trades`` is a list of ``app.backtest.execution.Trade`` (anything with ``pnl`` and ``entry_time``)."""
    cfg = cfg or LifecycleConfig()
    if not trades:
        raise ValueError("Cannot run a lifecycle simulation with zero trades.")
    mode = normalize_objective_mode(objective_mode)

    from app.monte_carlo.day_resample import PathSampler
    from app.monte_carlo.engine import _apply_slippage_stress

    rng = np.random.default_rng(cfg.random_seed)
    approval_rng = np.random.default_rng((cfg.random_seed or 0) + 999_983)
    order = sorted(range(len(trades)), key=lambda i: pd.Timestamp(trades[i].entry_time))
    trades = [trades[i] for i in order]
    base_pnls = np.array([float(t.pnl) for t in trades], dtype=float)
    base_dates = canonical_trade_dates(trades)
    base_ds = precompute_day_structure(base_dates, tz=rules.session_timezone, roll_hour=rules.session_roll_hour)
    sampler = PathSampler(base_dates, rules, cfg.method, cfg.block_size, cfg.day_block_days)
    history_days = int((base_ds.day_dates[-1] - base_ds.day_dates[0]).days) + 1 if base_ds.n_days else 0

    acct = float(rules.account_size)
    target_balance = acct * (1 + rules.evaluation_profit_target_pct / 100.0)
    daily_limit = acct * rules.daily_loss_limit_pct / 100.0 if rules.daily_loss_limit_pct else None
    approval_p = min(max(cfg.funding_approval_probability, 0.0), 100.0) / 100.0
    n = int(cfg.n_simulations)

    reach_before_breach = passed = daily_viol = 0
    f_daily = f_dd = f_time = f_other = unresolved = 0
    cons_ok_at_target = reached_target_clean = blocked_after_target = 0
    cal_to_pass: list = []
    trd_to_pass: list = []
    pass_cal_all: list = []                        # calendar days to pass for every simulation (None = never)
    funded_n = 0
    fund_dur: list = []
    fund_fail: list = []
    fund_fail_daily = fund_fail_dd = 0
    payout_counts: list = []
    payout_amounts_each: list = []
    payout_totals_all: list = []
    first_payout_cal: list = []
    pass_and_payout = 0

    for _ in range(n):
        idx, sim_ds = sampler.draw(rng)
        sim_pnls = _apply_slippage_stress(base_pnls[idx], cfg.slippage_stress_pct)
        ds = sim_ds if sim_ds is not None else base_ds
        res = simulate_account(sim_pnls, base_dates, rules, _day_structure=ds)
        day_dates = ds.day_dates

        # ---- order of events: target vs breach (running realized balance, whole-day granularity)
        running = acct + np.cumsum(sim_pnls)
        hit = np.flatnonzero(running >= target_balance)
        target_day = int(ds.day_index_per_trade[int(hit[0])]) if hit.size else None
        failed_eval = bool(res.failed and not res.passed_evaluation)
        failed_day = res.failure_day_index if failed_eval else None
        reached_clean = target_day is not None and (not failed_eval or (failed_day is not None and target_day < int(failed_day)))
        if res.passed_evaluation:
            reached_clean = True
        if reached_clean:
            reach_before_breach += 1
            reached_target_clean += 1
            # consistency check at the target day on the realized day sums
            sums = _day_sums(sim_pnls, ds.day_index_per_trade, ds.n_days)
            upto = sums[: (target_day if target_day is not None else ds.n_days - 1) + 1]
            total = float(upto.sum())
            cr = rules.consistency_rule_pct
            ok = True if (cr is None or total <= 0) else (float(upto.max()) / total * 100.0 <= cr)
            if ok:
                cons_ok_at_target += 1
            if not res.passed_evaluation and not failed_eval:
                blocked_after_target += 1

        if daily_limit:
            sums_all = _day_sums(sim_pnls, ds.day_index_per_trade, ds.n_days)
            if float(sums_all.min()) < -daily_limit:
                daily_viol += 1

        # ---- evaluation outcome
        if res.passed_evaluation:
            passed += 1
            cal = res.calendar_days_to_pass
            if cal is not None:
                cal_to_pass.append(float(cal))
                pass_cal_all.append(float(cal))
            else:
                pass_cal_all.append(None)
            if res.days_to_pass is not None:
                trd_to_pass.append(float(res.days_to_pass))
        else:
            pass_cal_all.append(None)
            if failed_eval:
                reason = str(res.failure_reason or "")
                if reason == "daily_loss_limit":
                    f_daily += 1
                elif reason.startswith("max_drawdown"):
                    f_dd += 1
                elif reason == "eval_time_limit":
                    f_time += 1
                else:
                    f_other += 1
            else:
                unresolved += 1

        # ---- funded stage
        is_funded = bool(res.passed_evaluation) and (approval_p >= 1.0 or approval_rng.random() < approval_p)
        if is_funded:
            funded_n += 1
            cal_pass = res.calendar_days_to_pass or 1
            start_date = pd.Timestamp(day_dates[0]) + pd.Timedelta(days=int(cal_pass) - 1)
            funded_failed = bool(res.failed and res.passed_evaluation)
            if funded_failed and res.failure_day_index is not None:
                end_date = pd.Timestamp(day_dates[min(int(res.failure_day_index), len(day_dates) - 1)])
            else:
                end_date = pd.Timestamp(day_dates[-1])
            fund_dur.append(float(max((end_date - start_date).days, 0)))
            fund_fail.append(funded_failed)
            if funded_failed:
                reason = str(res.failure_reason or "")
                if reason == "daily_loss_limit":
                    fund_fail_daily += 1
                elif reason.startswith("max_drawdown"):
                    fund_fail_dd += 1
            k = len(res.payouts)
            payout_counts.append(k)
            payout_totals_all.append(float(res.total_payout_amount))
            payout_amounts_each.extend(float(p.amount) for p in res.payouts)
            if k >= 1:
                pass_and_payout += 1
                try:
                    fp_date = pd.Timestamp(res.payouts[0].date)
                    first_payout_cal.append(float(max((fp_date - start_date).days, 0)))
                except Exception:  # noqa: BLE001
                    pass
        else:
            payout_totals_all.append(0.0)

    # ---- assemble
    pass_ci = wilson_interval(passed, n)
    windows_cov = {int(w): history_days >= int(w) for w in cfg.pass_windows_days}
    within = {}
    for w in cfg.pass_windows_days:
        within[int(w)] = _pct(sum(1 for c in pass_cal_all if c is not None and c <= w), n)

    evaluation = EvaluationStage(
        n_simulations=n,
        p_reach_target_before_breach=_pct(reach_before_breach, n),
        p_pass=_pct(passed, n), p_pass_ci=pass_ci,
        p_daily_loss_violation=_pct(daily_viol, n),
        p_fail_daily_loss=_pct(f_daily, n), p_fail_max_drawdown=_pct(f_dd, n),
        p_fail_time_limit=_pct(f_time, n), p_fail_other=_pct(f_other, n),
        p_unresolved=_pct(unresolved, n),
        p_consistency_ok_given_target=(_pct(cons_ok_at_target, reached_target_clean) if reached_target_clean else None),
        p_blocked_by_rules_after_target=_pct(blocked_after_target, n),
        time_to_target=_time_block(cal_to_pass, trd_to_pass),
        p_pass_within_days=within, windows_covered=windows_cov,
    )

    surv = kaplan_meier_survival(fund_dur, fund_fail, cfg.funded_horizons_days)
    arr = np.asarray(payout_counts, dtype=int) if payout_counts else np.zeros(0, dtype=int)

    def _cond(k):
        return _pct(int(np.sum(arr >= k)), funded_n) if funded_n else None

    funded = FundedStage(
        n_funded=funded_n, p_funded=_pct(funded_n, n),
        p_survive_given_funded={h: (None if s is None else 100.0 * s) for h, (s, _) in surv.items()},
        survival_at_risk={h: r for h, (_, r) in surv.items()},
        p_fail_funded_daily_loss=_pct(fund_fail_daily, funded_n) if funded_n else None,
        p_fail_funded_drawdown=_pct(fund_fail_dd, funded_n) if funded_n else None,
        p_first_payout_given_funded=_cond(1), p_second_payout_given_funded=_cond(2), p_third_payout_given_funded=_cond(3),
        expected_gross_payout_per_funded=(float(sum(payout_totals_all)) / funded_n if funded_n else 0.0),
        median_payout_amount=(float(np.median(payout_amounts_each)) if payout_amounts_each else None),
        time_to_first_payout={**_quantiles(first_payout_cal), "n": len(first_payout_cal), "clock": "calendar days from the evaluation pass"},
    )

    life_p = _pct(pass_and_payout, n)
    life_ci = wilson_interval(pass_and_payout, n)
    gross_per_attempt = float(np.sum(payout_totals_all)) / n
    fee = float(cfg.evaluation_fee)
    net_per_attempt = gross_per_attempt * cfg.profit_split_pct / 100.0 - fee
    p_life = pass_and_payout / n
    attempts_needed = (1.0 / p_life) if p_life > 0 else None
    fees_per_payout = (fee + (attempts_needed - 1.0) * cfg.resolved_reset_fee()) if attempts_needed is not None else None

    notes = []
    if history_days and history_days < 60:
        notes.append(f"Only {history_days} calendar days of history: time-to-target beyond that, and every funded-survival horizon "
                     "longer than the history, cannot be observed (reported as unknown, never as 100%).")
    if evaluation.p_unresolved > 20:
        notes.append(f"{evaluation.p_unresolved:.0f}% of simulated attempts neither passed nor failed before the trade history ran out; "
                     "they are NOT counted as passes. A longer history or a faster strategy resolves them.")
    if evaluation.p_blocked_by_rules_after_target > 5:
        notes.append(f"{evaluation.p_blocked_by_rules_after_target:.0f}% of attempts touched the target without breaching but could not "
                     "pass (consistency rule or minimum trading days) -- the strategy's profits are too concentrated or too fast for this firm.")
    if funded_n and funded.p_first_payout_given_funded is not None and funded.p_first_payout_given_funded < 50:
        notes.append(f"Only {funded.p_first_payout_given_funded:.0f}% of funded accounts reach a payout: passing the evaluation is not the "
                     "same as getting paid.")
    if evaluation.p_pass > 0 and life_p < 0.5 * evaluation.p_pass:
        notes.append("Fewer than half of the evaluation passes turn into a payout -- optimise for P(pass AND payout), not P(pass).")

    res_obj = LifecycleResult(
        evaluation=evaluation, funded=funded,
        p_pass_and_first_payout=life_p, p_pass_and_first_payout_ci=life_ci,
        expected_gross_payout_per_attempt=gross_per_attempt, expected_net_per_attempt=net_per_attempt,
        expected_fees_per_first_payout=fees_per_payout, expected_attempts_to_first_payout=attempts_needed,
        objective_mode=mode, objective_value=0.0, n_trades=len(trades), history_calendar_days=history_days, notes=notes,
        clocks={"days_to_pass": "CALENDAR days (first trade day through the day the target was met)",
                "legacy_days_to_pass": "TRADING days (days with at least one trade) -- AccountSimResult.days_to_pass",
                "funded_survival": "CALENDAR days since the evaluation pass, Kaplan-Meier (censored at the end of the history)",
                "pass_within_days": "unconditional: share of ALL simulated attempts"},
    )
    res_obj.objective_value = lifecycle_objective_value(res_obj, trades=trades, mode=mode)
    return res_obj


# ---------------------------------------------------------------------------------------------------------------- objectives
def lifecycle_objective_value(result: LifecycleResult, trades=None, mode: str = DEFAULT_OBJECTIVE_MODE, rules: PropRules | None = None) -> float:
    """The number an optimiser maximises.

    prop_lifecycle: P(pass AND first payout) in percent, plus up to 5 points for speed (shorter calendar time to target) and up to 5 for
        funded survival -- the bonuses only break ties between configurations with the same completion probability.
    net_profit:     net profit per trade after costs (see ``net_profit_objective``)."""
    mode = normalize_objective_mode(mode)
    if mode == OBJECTIVE_NET_PROFIT:
        return net_profit_objective(trades or [])["objective"]
    base = float(result.p_pass_and_first_payout)
    med = result.evaluation.time_to_target.get("calendar", {}).get("p50")
    speed = 0.0 if med is None else 5.0 * (30.0 / max(float(med), 30.0)) ** 0.5
    s90 = result.funded.p_survive_given_funded.get(90) if result.funded.p_survive_given_funded else None
    surv = 0.0 if s90 is None else 5.0 * float(s90) / 100.0
    return base + (speed + surv) * (1.0 if base > 0 else 0.0)


def net_profit_objective(trades, account_size: float | None = None) -> dict:
    """Net-profit objective for LIVE accounts: no evaluation, no target, no payout gate -- just what the equity curve earns after the
    costs already baked into each trade's ``pnl`` (commission, spread, slippage), penalised for drawdown and for a thin sample.

    objective = net_profit_per_trade * sqrt(n_trades) / (1 + max_drawdown_fraction)   (a t-stat-like score shrunk by drawdown)
    Returns the components so a report can show them."""
    if not trades:
        return {"objective": float("-inf"), "net_profit": 0.0, "n_trades": 0, "net_per_trade": 0.0, "profit_factor": 0.0,
                "max_drawdown": 0.0, "max_drawdown_pct": 0.0, "win_rate_pct": 0.0, "sharpe_like": 0.0, "mode": OBJECTIVE_NET_PROFIT}
    pnls = np.array([float(t.pnl) for t in sorted(trades, key=lambda t: pd.Timestamp(t.entry_time))], dtype=float)
    n = len(pnls)
    net = float(pnls.sum())
    wins, losses = pnls[pnls > 0].sum(), -pnls[pnls < 0].sum()
    pf = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
    base = float(account_size) if account_size else max(1.0, float(np.abs(pnls).sum()))
    eq = base + np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate([[base], eq]))[1:]
    dd = float((peak - eq).max())
    dd_frac = dd / max(base, 1.0)
    sd = float(pnls.std(ddof=1)) if n > 1 else 0.0
    sharpe_like = (float(pnls.mean()) / sd * math.sqrt(n)) if sd > 0 else 0.0
    obj = (net / n) * math.sqrt(n) / (1.0 + dd_frac)
    return {"objective": float(obj), "net_profit": net, "n_trades": n, "net_per_trade": net / n, "profit_factor": pf,
            "max_drawdown": dd, "max_drawdown_pct": dd_frac * 100.0, "win_rate_pct": float((pnls > 0).mean() * 100.0),
            "sharpe_like": float(sharpe_like), "mode": OBJECTIVE_NET_PROFIT}


def evaluate_objective(trades, rules: PropRules | None, mode: str = DEFAULT_OBJECTIVE_MODE, cfg: LifecycleConfig | None = None,
                       account_size: float | None = None) -> dict:
    """One call for every optimiser/report: ``mode`` 'prop_lifecycle' (default) or 'net_profit'.

    Returns {"mode", "objective", "label", ...components}. In prop mode ``rules`` is required; if it is missing the call falls back to the
    net-profit components and says so rather than inventing prop numbers."""
    mode = normalize_objective_mode(mode)
    if mode == OBJECTIVE_NET_PROFIT or rules is None:
        out = net_profit_objective(trades, account_size or (rules.account_size if rules else None))
        out["label"] = OBJECTIVE_MODE_LABELS[OBJECTIVE_NET_PROFIT]
        if mode == OBJECTIVE_PROP and rules is None:
            out["fallback"] = "no prop rules supplied -- scored on net profit"
        return out
    lc = run_lifecycle(trades, rules, cfg, objective_mode=mode)
    return {"mode": mode, "objective": lc.objective_value, "label": OBJECTIVE_MODE_LABELS[mode], "lifecycle": lc.to_dict(),
            "headline": lc.headline()}
