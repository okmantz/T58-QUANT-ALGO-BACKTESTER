"""Rulebook v2 account lifecycle: cushion-proportional two-speed sizing and a multi-payout funded phase.

This is the simulator behind gates G1 (design envelope), G4 (day behaviour), G5 (lifecycle) and G7's priors in
``app.validation.rulebook``. It replaces the old fixed-dollar-risk lifecycle for Rulebook purposes:

    Evaluation   risk = 7.5% of the loss limit ($150 on a $2,000 limit) while the cushion (balance - floor) is at least half the limit;
                 x0.5 below half, x0.25 below a quarter; never below one micro contract (~$30).
    Funded       risk = 5% of the cushion, capped at 5% of the loss limit ($100), floor one micro contract.
    Day stop     stop the day after a loss of min(3R, 60% of the soft daily limit); the soft limit itself only locks the day.
    Payouts      taken at first eligibility (default: 5 days with P&L >= $150 and cumulative profit >= $1,000), keeping a buffer above
                 the floor; withdrawing does NOT lower the floor, so the cushion shrinks and the sizing shrinks with it.

Trades are resampled as whole SESSION DAYS (day-block bootstrap, 1-day blocks) and held as R multiples (pnl / planned risk), so the dollar
risk can be re-set per trade by the policy. The floor is checked after every trade; the trailing floor moves on END-OF-DAY balance.

Payout terms are assumptions until the rules sheet (G0) has them sourced: ``PayoutTerms`` holds them in one place and every result
carries them. ``sizing="fixed"`` reproduces the v1 behaviour (a constant dollar risk) so the two policies can be compared on the same
paths. Pure numpy; vectorised across simulated attempts, so 20,000 paths run in a few seconds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

# ---- Rulebook v2 sizing constants (G1) -------------------------------------------------------------------------------
EVAL_RISK_FRAC_OF_LIMIT = 0.075          # evaluation risk = 7.5% of the loss limit
FUNDED_RISK_FRAC_OF_LIMIT = 0.05         # funded risk cap = 5% of the loss limit
FUNDED_RISK_FRAC_OF_CUSHION = 0.05       # funded risk <= 5% of the cushion
DAY_STOP_R = 3.0                         # day stop: lower of 3R ...
DAY_STOP_FRAC_OF_SOFT = 0.60             # ... and 60% of the soft daily limit
MICRO_RISK_DOLLARS = 30.0                # one micro contract at the stop distance (about $30)
BEST_DAY_FRAC_OF_TARGET = 0.33           # G1/G4: best day <= 33% of the profit target


# ---- Rules ---------------------------------------------------------------------------------------------------------------
@dataclass
class PayoutTerms:
    """Funded payout terms. The defaults are the report's ASSUMPTION for Lucid 50k, not a firm-published term (G0 must replace them)."""
    qualifying_days: int = 5
    qualifying_day_min: float = 150.0
    cumulative_min: float = 1_000.0       # profit above the starting balance before a payout is allowed
    buffer_above_floor: float = 500.0     # leave this much above the floor when the firm allows it ($500 to $1,000)
    cap: float | None = None              # max per payout (None = no cap)
    max_payouts: int = 12
    funded_days: int = 252
    split: float = 0.90
    sourced: bool = False                 # True once the rules sheet has each term dated and sourced


@dataclass
class V2Rules:
    account_size: float = 50_000.0
    loss_limit: float = 2_000.0           # trailing max loss ($)
    profit_target: float = 3_000.0
    soft_daily_limit: float = 1_200.0     # locks the day, does not fail the account
    lock_floor_offset: float = 100.0      # the trailing floor stops moving at start + this
    trailing: bool = True
    fee: float = 172.0
    consistency_best_day_pct: float | None = None   # evaluation best-day rule (50 on LucidFlex), None = none
    micro_risk: float = MICRO_RISK_DOLLARS
    max_eval_days: int = 180
    payout: PayoutTerms = field(default_factory=PayoutTerms)

    @classmethod
    def from_prop_rules(cls, rules, *, fee: float = 0.0, split_pct: float | None = None, payout: PayoutTerms | None = None) -> "V2Rules":
        a = float(rules.account_size)
        pt = payout or PayoutTerms()
        if split_pct is not None:
            pt = PayoutTerms(**{**asdict(pt), "split": float(split_pct) / 100.0})
        lock = float(getattr(rules, "trailing_lock_offset_pct", 0.0) or 0.0) / 100.0 * a if getattr(rules, "trailing_lock", False) else 0.0
        return cls(account_size=a, loss_limit=a * float(rules.max_drawdown_pct) / 100.0,
                   profit_target=a * float(rules.evaluation_profit_target_pct) / 100.0,
                   soft_daily_limit=a * float(rules.daily_loss_limit_pct) / 100.0,
                   lock_floor_offset=lock, trailing=str(getattr(rules, "drawdown_type", "trailing")) == "trailing",
                   fee=float(fee), consistency_best_day_pct=getattr(rules, "consistency_rule_pct", None), payout=pt)

    @classmethod
    def lucid_50k(cls) -> "V2Rules":
        return cls()

    def tier(self, account_size: float) -> "V2Rules":
        """The same rule shape scaled to another account size (the demo tiers: 25k / 50k / 100k)."""
        k = float(account_size) / float(self.account_size)
        return V2Rules(account_size=float(account_size), loss_limit=self.loss_limit * k, profit_target=self.profit_target * k,
                       soft_daily_limit=self.soft_daily_limit * k, lock_floor_offset=self.lock_floor_offset * k, trailing=self.trailing,
                       fee=self.fee * k, consistency_best_day_pct=self.consistency_best_day_pct, micro_risk=self.micro_risk,
                       max_eval_days=self.max_eval_days, payout=PayoutTerms(**{**asdict(self.payout),
                       "qualifying_day_min": self.payout.qualifying_day_min * k, "cumulative_min": self.payout.cumulative_min * k,
                       "buffer_above_floor": self.payout.buffer_above_floor * k}))


# ---- Sizing policy (scalar and vector) ----------------------------------------------------------------------------------
def eval_risk(cushion, loss_limit: float, micro: float = MICRO_RISK_DOLLARS):
    """Evaluation risk in dollars: 7.5% of the limit, x0.5 once the cushion is under half the limit, x0.25 under a quarter, floor one micro."""
    c = np.asarray(cushion, dtype=float)
    base = EVAL_RISK_FRAC_OF_LIMIT * float(loss_limit)
    mult = np.where(c >= 0.5 * loss_limit, 1.0, np.where(c >= 0.25 * loss_limit, 0.5, 0.25))
    out = np.maximum(base * mult, micro)
    return float(out) if np.ndim(cushion) == 0 else out


def funded_risk(cushion, loss_limit: float, micro: float = MICRO_RISK_DOLLARS):
    """Funded risk in dollars: 5% of the cushion, capped at 5% of the limit, floor one micro."""
    c = np.asarray(cushion, dtype=float)
    out = np.maximum(np.minimum(FUNDED_RISK_FRAC_OF_CUSHION * c, FUNDED_RISK_FRAC_OF_LIMIT * float(loss_limit)), micro)
    return float(out) if np.ndim(cushion) == 0 else out


def day_stop_dollars(risk_dollars, soft_daily_limit: float):
    """Stop trading for the day after losing this much: the lower of 3R and 60% of the soft daily limit."""
    d = np.minimum(DAY_STOP_R * np.asarray(risk_dollars, dtype=float), DAY_STOP_FRAC_OF_SOFT * float(soft_daily_limit))
    return float(d) if np.ndim(risk_dollars) == 0 else d


def design_envelope(rules: V2Rules) -> dict:
    """G1: what the strategy has to look like, computed from the rules sheet."""
    L = rules.loss_limit
    return {
        "evaluation_risk": round(EVAL_RISK_FRAC_OF_LIMIT * L, 2),
        "evaluation_risk_pct_of_limit": EVAL_RISK_FRAC_OF_LIMIT * 100,
        "funded_risk_cap": round(FUNDED_RISK_FRAC_OF_LIMIT * L, 2),
        "funded_risk_pct_of_cushion": FUNDED_RISK_FRAC_OF_CUSHION * 100,
        "micro_floor": rules.micro_risk,
        "payoff_range": (1.5, 2.5), "take_profit_cap_r": 3.0,
        "trades_per_day": (2, 4),
        "daily_stop": round(min(DAY_STOP_R * EVAL_RISK_FRAC_OF_LIMIT * L, DAY_STOP_FRAC_OF_SOFT * rules.soft_daily_limit), 2),
        "best_day_max": round(BEST_DAY_FRAC_OF_TARGET * rules.profit_target, 2),
        "withdrawal_buffer": rules.payout.buffer_above_floor,
        "profit_target": rules.profit_target, "loss_limit": L, "soft_daily_limit": rules.soft_daily_limit,
    }


# ---- Inputs ---------------------------------------------------------------------------------------------------------------
def day_matrix_from_trades(trades) -> tuple[np.ndarray, dict]:
    """(matrix [n_days, K] of R multiples padded with NaN, info). Trades are grouped by session day; trades without a planned risk are
    dropped and counted (they cannot be re-sized by a risk policy)."""
    from app.data.trading_day import trading_day
    from app.scoring.edge_bounds import trade_r
    import pandas as pd
    by_day: dict = {}
    skipped = 0
    for t in sorted(trades or [], key=lambda x: pd.Timestamp(x.entry_time)):
        r = trade_r(t)
        if r is None or not math.isfinite(r):
            skipped += 1
            continue
        by_day.setdefault(trading_day(pd.Timestamp(t.entry_time)), []).append(float(r))
    if not by_day:
        return np.full((0, 1), np.nan), {"n_days": 0, "n_trades": 0, "skipped_no_risk": skipped}
    k = max(len(v) for v in by_day.values())
    mat = np.full((len(by_day), k), np.nan)
    for i, d in enumerate(sorted(by_day)):
        v = by_day[d]
        mat[i, :len(v)] = v
    n = int(np.isfinite(mat).sum())
    return mat, {"n_days": int(mat.shape[0]), "n_trades": n, "skipped_no_risk": skipped, "max_trades_per_day": int(k),
                 "median_trades_per_day": float(np.median([len(v) for v in by_day.values()])),
                 "mean_r": float(np.nanmean(mat)), "sd_r": float(np.nanstd(mat, ddof=1)) if n > 1 else 0.0}


def binary_day_matrix(edge_r: float, payoff_b: float = 2.0, trades_per_day: int = 3, n_days: int = 2_000, seed: int = 7) -> np.ndarray:
    """Synthetic i.i.d. binary trades (+b R or -1 R) at net edge ``edge_r`` -- the design prior the Rulebook tables were built on."""
    p = (edge_r + 1.0) / (payoff_b + 1.0)
    rng = np.random.default_rng(seed)
    return np.where(rng.random((n_days, trades_per_day)) < p, payoff_b, -1.0)


# ---- Result ---------------------------------------------------------------------------------------------------------------
@dataclass
class V2Result:
    n_paths: int
    sizing: str
    p_pass: float
    p_bust_eval: float
    p_unresolved: float
    p_pass_ci: tuple
    p_paid: float                   # P(first payout) over all purchased attempts
    p_paid_ci: tuple
    p_paid_given_pass: float
    p_bust_funded_before_payout: float
    p_two_plus_given_paid: float
    p_two_plus_ci: tuple
    funded_alive_252: float         # % of funded accounts still alive at the end of the funded window
    mean_payouts_given_paid: float
    gross_paid_per_attempt: float
    net_paid_per_attempt: float     # after the profit split, before the fee
    ev_per_attempt: float           # split * payouts - fee
    ev_per_attempt_haircut: float   # same with payouts cut 50%
    attempts_per_payout: float | None
    fees_per_payout: float | None
    median_days_to_first_payout: float | None
    best_day_over_target_p95: float | None
    p_day_loss_ge_soft: float
    qualifying_day_rate: float
    terms_sourced: bool
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d

    def headline(self) -> str:
        return (f"v2 sizing: P(pass) {self.p_pass:.0f}%  P(paid) {self.p_paid:.0f}%  P(>=2 payouts|paid) {self.p_two_plus_given_paid:.0f}%  "
                f"alive@252 {self.funded_alive_252:.0f}%  EV/attempt ${self.ev_per_attempt:,.0f}")


def wilson_ci(k: float, n: float, z: float = 1.645) -> tuple:
    """One-sided-style 90% Wilson interval (z=1.645 gives the 5th percentile as its lower end), in percent."""
    if n <= 0:
        return (0.0, 100.0)
    ph = k / n
    den = 1.0 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (100.0 * max(0.0, c - h), 100.0 * min(1.0, c + h))


# ---- Simulator --------------------------------------------------------------------------------------------------------------
def _floor_update(floor, max_eod, bal, rules: V2Rules, start: float):
    """End-of-day trailing floor: max EOD balance - loss limit, locked at start + lock offset when the firm locks it."""
    new_max = np.maximum(max_eod, bal)
    if rules.trailing:
        f = new_max - rules.loss_limit
        if rules.lock_floor_offset:
            f = np.minimum(f, start + rules.lock_floor_offset)
        floor = np.maximum(floor, f)
    return floor, new_max


def simulate(day_matrix: np.ndarray, rules: V2Rules | None = None, *, n_paths: int = 20_000, sizing: str = "v2", fixed_risk: float | None = None,
             seed: int | None = 42, payout_haircut: float = 0.5, funded_sizing: str | None = None,
             funded_recompute: str = "payout") -> V2Result:
    """Run ``n_paths`` purchased attempts (evaluation, then a funded account for each pass) over a day-block bootstrap of ``day_matrix``.

    sizing: "v2" (cushion-proportional, two speeds) or "fixed" (``fixed_risk`` dollars in both phases, the v1 behaviour).
    funded_recompute: "payout" (the Rulebook rule: funded risk is set at the start and re-set after every payout) or "trade" (re-set after
    every trade -- tighter, available as a stricter live guardrail). Calibration: at +0.10R, 2:1, 3 trades a day the "payout" setting
    gives P(paid) ~64%, P(>=2 payouts | paid) ~86%, funded alive at 252 days ~64% against the Rulebook table's 66 / 81 / 63."""
    rules = rules or V2Rules()
    pt = rules.payout
    dm = np.asarray(day_matrix, dtype=float)
    notes: list = []
    if dm.ndim != 2 or dm.shape[0] < 20 or not np.isfinite(dm).any():
        raise ValueError("need at least 20 trading days of trades with a planned risk (R) to simulate the lifecycle")
    D, K = dm.shape
    rng = np.random.default_rng(seed)
    P = int(n_paths)
    A, L = rules.account_size, rules.loss_limit
    fixed = float(fixed_risk if fixed_risk is not None else EVAL_RISK_FRAC_OF_LIMIT * L)
    fsz = funded_sizing or sizing

    def risk_for(phase_sizing: str, funded: bool, cushion):
        if phase_sizing == "fixed":
            return np.full_like(cushion, fixed)
        return funded_risk(cushion, L, rules.micro_risk) if funded else eval_risk(cushion, L, rules.micro_risk)

    # ------------------------------------------------ evaluation
    bal = np.full(P, A); floor = np.full(P, A - L); max_eod = np.full(P, A)
    alive = np.ones(P, bool); passed = np.zeros(P, bool); eval_days = np.zeros(P)
    best_day = np.zeros(P); total_profit_days = np.zeros(P)
    day_loss_hits = 0; qual_days = 0; days_seen = 0
    target_bal = A + rules.profit_target
    for d in range(rules.max_eval_days):
        live = alive & ~passed
        if not live.any():
            break
        rows = dm[rng.integers(0, D, P)]
        day_pnl = np.zeros(P); locked = ~live
        eval_days[live] += 1
        for k in range(K):
            r = rows[:, k]
            act = ~locked & np.isfinite(r)
            if not act.any():
                continue
            risk = risk_for(sizing, False, bal - floor)
            pnl = np.where(act, np.nan_to_num(r) * risk, 0.0)
            bal = bal + pnl; day_pnl = day_pnl + pnl
            breach = act & (bal <= floor)
            alive &= ~breach; locked |= breach
            stop = np.minimum(DAY_STOP_R * risk, DAY_STOP_FRAC_OF_SOFT * rules.soft_daily_limit)
            locked |= act & ((-day_pnl >= stop) | (-day_pnl >= rules.soft_daily_limit))
        day_loss_hits += int(((-day_pnl >= rules.soft_daily_limit) & live).sum())
        qual_days += int(((day_pnl >= pt.qualifying_day_min) & live).sum()); days_seen += int(live.sum())
        still = alive & ~passed
        floor, max_eod = _floor_update(floor, max_eod, np.where(still, bal, max_eod), rules, A)
        best_day = np.where(still, np.maximum(best_day, day_pnl), best_day)
        ok_target = still & (bal >= target_bal)
        if rules.consistency_best_day_pct:
            ok_target &= best_day <= (rules.consistency_best_day_pct / 100.0) * np.maximum(bal - A, 1e-9)
        passed |= ok_target
    unresolved = alive & ~passed
    bust_eval = ~alive & ~passed
    n_pass = int(passed.sum())

    # ------------------------------------------------ funded
    idx = np.flatnonzero(passed)
    m = idx.size
    pay_count = np.zeros(m, int); gross = np.zeros(m); first_day = np.full(m, np.nan)
    f_alive = np.ones(m, bool); busted_before_first = np.zeros(m, bool)
    if m:
        fb = np.full(m, A); ff = np.full(m, A - L); fm = np.full(m, A); qual = np.zeros(m, int)
        days_done = np.zeros(m)
        frisk = risk_for(fsz, True, fb - ff)
        for d in range(pt.funded_days):
            live = f_alive
            if not live.any():
                break
            rows = dm[rng.integers(0, D, m)]
            day_pnl = np.zeros(m); locked = ~live
            days_done[live] += 1
            for k in range(K):
                r = rows[:, k]
                act = ~locked & np.isfinite(r)
                if not act.any():
                    continue
                risk = risk_for(fsz, True, fb - ff) if funded_recompute == "trade" else frisk
                pnl = np.where(act, np.nan_to_num(r) * risk, 0.0)
                fb = fb + pnl; day_pnl = day_pnl + pnl
                breach = act & (fb <= ff)
                f_alive &= ~breach; locked |= breach
                stop = np.minimum(DAY_STOP_R * risk, DAY_STOP_FRAC_OF_SOFT * rules.soft_daily_limit)
                locked |= act & ((-day_pnl >= stop) | (-day_pnl >= rules.soft_daily_limit))
            ff, fm = _floor_update(ff, fm, np.where(f_alive, fb, fm), rules, A)
            qual = np.where(f_alive & (day_pnl >= pt.qualifying_day_min), qual + 1, qual)
            elig = f_alive & (qual >= pt.qualifying_days) & ((fb - A) >= pt.cumulative_min) & (pay_count < pt.max_payouts)
            keep = np.maximum(A, ff + pt.buffer_above_floor)
            w = np.where(elig, np.maximum(fb - keep, 0.0), 0.0)
            if pt.cap:
                w = np.minimum(w, pt.cap)
            got = elig & (w > 0)
            fb = fb - w; gross += w; pay_count += got.astype(int); qual = np.where(got, 0, qual)
            if funded_recompute != "trade":
                frisk = np.where(got, risk_for(fsz, True, fb - ff), frisk)
            first_day = np.where(got & np.isnan(first_day), eval_days[idx] + days_done, first_day)
        busted_before_first = ~f_alive & (pay_count == 0)

    paid = pay_count >= 1
    n_paid = int(paid.sum())
    two = int((pay_count >= 2).sum())
    gross_all = float(gross.sum()) / P
    net_all = gross_all * pt.split
    haircut_net = net_all * (1.0 - payout_haircut)
    p_paid = n_paid / P
    day_best = None
    if days_seen:
        day_best = float(np.percentile(best_day[passed | bust_eval | unresolved], 95)) / max(rules.profit_target, 1.0)
    return V2Result(
        n_paths=P, sizing=sizing, p_pass=100.0 * n_pass / P, p_bust_eval=100.0 * float(bust_eval.sum()) / P,
        p_unresolved=100.0 * float(unresolved.sum()) / P, p_pass_ci=wilson_ci(n_pass, P),
        p_paid=100.0 * p_paid, p_paid_ci=wilson_ci(n_paid, P),
        p_paid_given_pass=(100.0 * n_paid / n_pass) if n_pass else 0.0,
        p_bust_funded_before_payout=(100.0 * float(busted_before_first.sum()) / n_pass) if n_pass else 0.0,
        p_two_plus_given_paid=(100.0 * two / n_paid) if n_paid else 0.0, p_two_plus_ci=wilson_ci(two, n_paid),
        funded_alive_252=(100.0 * float(f_alive.sum()) / m) if m else 0.0,
        mean_payouts_given_paid=float(pay_count[paid].mean()) if n_paid else 0.0,
        gross_paid_per_attempt=gross_all, net_paid_per_attempt=net_all,
        ev_per_attempt=net_all - rules.fee, ev_per_attempt_haircut=haircut_net - rules.fee,
        attempts_per_payout=(1.0 / p_paid) if p_paid > 0 else None,
        fees_per_payout=(rules.fee / p_paid) if p_paid > 0 else None,
        median_days_to_first_payout=float(np.nanmedian(first_day)) if n_paid else None,
        best_day_over_target_p95=day_best,
        p_day_loss_ge_soft=(100.0 * day_loss_hits / days_seen) if days_seen else 0.0,
        qualifying_day_rate=(100.0 * qual_days / days_seen) if days_seen else 0.0,
        terms_sourced=bool(pt.sourced),
        notes=notes + ([] if pt.sourced else ["Payout terms are the report's assumption, not a sourced firm term (G0)."]),
    )


def compare_policies(day_matrix: np.ndarray, rules: V2Rules | None = None, **kw) -> dict:
    """v1 fixed sizing vs v2 cushion sizing on the same day matrix -- the Rulebook's headline comparison."""
    rules = rules or V2Rules()
    v1_rules = V2Rules(**{**rules.__dict__, "payout": PayoutTerms(**{**asdict(rules.payout), "buffer_above_floor": 0.0})})
    v1 = simulate(day_matrix, v1_rules, sizing="fixed", **kw)       # v1: constant $ risk, withdraw down to the starting balance
    v2 = simulate(day_matrix, rules, sizing="v2", **kw)
    return {"v1": v1.to_dict(), "v2": v2.to_dict()}


def run_lifecycle_v2(trades, rules, *, fee: float = 0.0, split_pct: float | None = None, n_paths: int = 20_000,
                     seed: int | None = 42, payout: PayoutTerms | None = None) -> dict:
    """Real-trade entry point. ``rules`` is a PropRules (or a V2Rules). Returns {"result": V2Result dict, "design": envelope, "input": info}."""
    mat, info = day_matrix_from_trades(trades)
    r2 = rules if isinstance(rules, V2Rules) else V2Rules.from_prop_rules(rules, fee=fee, split_pct=split_pct, payout=payout)
    if mat.shape[0] < 20:
        return {"ok": False, "reason": f"only {mat.shape[0]} trading days with a planned risk; the lifecycle needs at least 20", "input": info}
    res = simulate(mat, r2, n_paths=n_paths, seed=seed)
    return {"ok": True, "result": res, "design": design_envelope(r2), "input": info}
