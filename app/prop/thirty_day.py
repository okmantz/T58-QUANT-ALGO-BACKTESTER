"""30-DAY PROP EVALUATION OPTIMIZER -- the evaluator.

The objective this module scores is ONE question, asked the way a trader lives it:

    "If I buy a fresh evaluation account on an ordinary weekday and trade this strategy under the firm's real rules,
     what is the probability that I PASS INSIDE 30 CALENDAR DAYS -- and how sure are we of that number?"

It deliberately does NOT rank on Sharpe, profit factor, annual return, a strategy score, or an "eventual" pass probability.
A strategy that passes 99% of the time given unlimited time, but takes ten months, scores ~0 here -- correctly.

Design rules (each one answers a defect found in the 2026-11 audit of the NQ Gap-Aligned Opening Drive report, see
``docs/PROP30_AUDIT.md``):

* The clock starts at the PURCHASE day, not at the first trade. ``PropRules``-level ``calendar_days_to_pass`` counts from the
  account's first trade day, which flatters a sparse strategy (it would "start" on its first signal). Here a window opens on the
  purchase weekday and every elapsed day -- including waiting for the first signal -- counts against the 30.
* Days are CALENDAR days everywhere (``days_to_pass`` in the simulator is a count of days that contained a trade and is never used).
* Every weekday in the data is a purchase date (Monday-Friday; nobody buys an evaluation to start on a Saturday), so no start
  date is cherry-picked, and a window only counts when the data covers its whole horizon (no right-censoring bias in either
  direction: a window cut short by the end of the data is neither a pass nor a fail, it is not a sample).
* The rule engine is ``app.prop.simulator.simulate_account`` (-> ``PropAccount``): floating drawdown, floating daily loss with
  lock-day, EOD/intraday trailing with lock, trading-day minimum, consistency. Nothing is reimplemented here, so a rule change in
  the account state machine reaches this objective automatically. Floating rules need per-trade MAE/MFE; a ledger without
  measured excursions is refused unless the caller explicitly accepts the closed-P&L proxy.
* Overlapping weekday starts are NOT independent trials. The interval is dependence-aware: the LOWER bound is the smaller of a
  moving-block bootstrap (blocks of two horizons) and a Wilson interval on disjoint 30-day windows. The 65% target counts as
  achieved only when that lower bound reaches it, on enough independent windows (``ThirtyDayConfig.min_independent_windows``).
* ``claim_target_achieved`` (here) is a purely statistical check. Whether the data was independent of the search is a separate
  gate, ``app.validation.prop30_validation``. Passing the first without the second is a measurement, not a claim.

Pure NumPy/pandas; no network, no randomness except the seeded bootstrap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from app.data.trading_day import trading_days
from app.prop.excursions import trade_excursions
from app.prop.simulator import PropRules, precompute_day_structure, simulate_account

TARGET_PASS_PROBABILITY = 0.65
DEFAULT_HORIZON_DAYS = 30
DEFAULT_LONG_HORIZON_DAYS = 60
Z95 = 1.959963984540054


# ---------------------------------------------------------------------------------------------------------------- ledger
def _naive_utc(values) -> np.ndarray:
    idx = pd.DatetimeIndex(pd.to_datetime(list(values)))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    return idx.values.astype("datetime64[ns]")


@dataclass
class TradeLedger:
    """Chronological, array-backed trade list: what the evaluator, the exit lab and the portfolio builder all exchange.

    ``pnl``/``mae``/``mfe``/``cost`` are account dollars (``mae <= 0``, ``mfe >= 0``, ``cost >= 0`` round-turn commission + spread +
    slippage already INSIDE ``pnl``). ``risk`` is the initial stop risk in dollars (NaN when unknown). Times are naive UTC.
    ``excursions_measured`` is the share of trades whose MAE/MFE came from the bar path (1.0) rather than the closed-P&L proxy.
    """

    entry_time: np.ndarray
    exit_time: np.ndarray
    pnl: np.ndarray
    mae: np.ndarray
    mfe: np.ndarray
    cost: np.ndarray
    risk: np.ndarray
    side: np.ndarray
    strategy: np.ndarray
    instrument: np.ndarray
    excursions_measured: float = 1.0

    def __post_init__(self):
        n = len(self.pnl)
        for name in ("entry_time", "exit_time", "mae", "mfe", "cost", "risk", "side", "strategy", "instrument"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"TradeLedger field {name!r} has {len(getattr(self, name))} rows, expected {n}")
        if n:
            order = np.argsort(self.entry_time, kind="stable")
            if not np.array_equal(order, np.arange(n)):
                for name in ("entry_time", "exit_time", "pnl", "mae", "mfe", "cost", "risk", "side", "strategy", "instrument"):
                    setattr(self, name, np.asarray(getattr(self, name))[order])

    def __len__(self) -> int:
        return int(len(self.pnl))

    @classmethod
    def empty(cls) -> "TradeLedger":
        z = np.zeros(0)
        return cls(np.zeros(0, "datetime64[ns]"), np.zeros(0, "datetime64[ns]"), z, z, z, z, z, np.zeros(0, int),
                   np.zeros(0, object), np.zeros(0, object))

    @classmethod
    def from_trades(cls, trades: Sequence, risk=None, strategy: str = "", instrument: str = "") -> "TradeLedger":
        """Build from ``app.backtest.execution.Trade`` objects. ``risk`` (a RiskConfig) adds the spread/slippage dollars to ``cost``."""
        if not trades:
            return cls.empty()
        exc = trade_excursions(trades)
        side_cost = float(risk.side_cost_price()) if risk is not None and hasattr(risk, "side_cost_price") else 0.0
        cost, rk = [], []
        for t in trades:
            size = float(getattr(t, "size", 0.0) or 0.0)
            cost.append(abs(float(getattr(t, "commission", 0.0) or 0.0)) + 2.0 * side_cost * size)
            ir = getattr(t, "initial_risk", None)
            rk.append(float(ir) * size if ir is not None and math.isfinite(float(ir)) else float("nan"))
        n = len(trades)
        return cls(
            entry_time=_naive_utc(t.entry_time for t in trades), exit_time=_naive_utc(t.exit_time for t in trades),
            pnl=np.asarray([float(t.pnl) for t in trades]), mae=np.asarray(exc.mae, float), mfe=np.asarray(exc.mfe, float),
            cost=np.asarray(cost, float), risk=np.asarray(rk, float),
            side=np.asarray([int(getattr(t, "direction", 0) or 0) for t in trades]),
            strategy=np.asarray([strategy] * n, dtype=object), instrument=np.asarray([instrument] * n, dtype=object),
            excursions_measured=float(exc.measured_fraction),
        )

    @classmethod
    def merge(cls, ledgers: Iterable["TradeLedger"]) -> "TradeLedger":
        parts = [l for l in ledgers if len(l)]
        if not parts:
            return cls.empty()
        cat = lambda name: np.concatenate([getattr(p, name) for p in parts])  # noqa: E731
        total = sum(len(p) for p in parts)
        measured = sum(p.excursions_measured * len(p) for p in parts) / max(total, 1)
        return cls(cat("entry_time"), cat("exit_time"), cat("pnl"), cat("mae"), cat("mfe"), cat("cost"), cat("risk"), cat("side"),
                   cat("strategy"), cat("instrument"), excursions_measured=float(measured))

    def select(self, mask) -> "TradeLedger":
        m = np.asarray(mask, bool)
        return TradeLedger(self.entry_time[m], self.exit_time[m], self.pnl[m], self.mae[m], self.mfe[m], self.cost[m], self.risk[m],
                           self.side[m], self.strategy[m], self.instrument[m], self.excursions_measured)

    def between(self, start, end) -> "TradeLedger":
        """Trades ENTERED in [start, end) (naive-UTC comparable)."""
        s, e = np.datetime64(pd.Timestamp(start).tz_localize(None) if pd.Timestamp(start).tzinfo else pd.Timestamp(start)), \
            np.datetime64(pd.Timestamp(end).tz_localize(None) if pd.Timestamp(end).tzinfo else pd.Timestamp(end))
        return self.select((self.entry_time >= s) & (self.entry_time < e))

    def with_cost_multiplier(self, mult: float) -> "TradeLedger":
        """The same trades with ``mult`` x their round-turn cost (cost stress). Excursions move with the extra cost so floating rules
        see it too."""
        extra = (float(mult) - 1.0) * self.cost
        pnl = self.pnl - extra
        return TradeLedger(self.entry_time, self.exit_time, pnl, np.minimum(self.mae - extra, pnl), np.maximum(self.mfe - 0.5 * extra, pnl),
                           self.cost * float(mult), self.risk, self.side, self.strategy, self.instrument, self.excursions_measured)

    def scaled(self, factor: float) -> "TradeLedger":
        """Every dollar figure x ``factor`` (position-size scaling: risk fraction up or down, same signals)."""
        f = float(factor)
        return TradeLedger(self.entry_time, self.exit_time, self.pnl * f, self.mae * f, self.mfe * f, self.cost * f, self.risk * f,
                           self.side, self.strategy, self.instrument, self.excursions_measured)

    def r_multiples(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(np.isfinite(self.risk) & (self.risk > 0), self.pnl / self.risk, np.nan)
        return r

    def span(self) -> tuple:
        if not len(self):
            return (None, None)
        return (pd.Timestamp(self.entry_time[0]), pd.Timestamp(self.exit_time.max()))


# ---------------------------------------------------------------------------------------------------------------- intervals
def wilson_interval(p_hat: float, n: float, z: float = Z95) -> tuple:
    """Wilson score interval for a proportion with a (possibly fractional) effective sample size."""
    if n <= 0:
        return (0.0, 1.0)
    d = 1.0 + z * z / n
    c = (p_hat + z * z / (2.0 * n)) / d
    h = z * math.sqrt(max(p_hat * (1.0 - p_hat) / n + z * z / (4.0 * n * n), 0.0)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def moving_block_bootstrap_ci(flags, block_len: int, n_boot: int = 2000, seed: int = 7, alpha: float = 0.05) -> tuple:
    """Circular moving-block bootstrap percentile interval for the mean of an ordered 0/1 series."""
    x = np.asarray(flags, float)
    n = len(x)
    if n == 0:
        return (0.0, 1.0)
    L = int(max(1, min(block_len, n)))
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / L))
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    ext = np.concatenate([x, x[:L]])                                  # circular wrap
    csum = np.concatenate([[0.0], np.cumsum(ext)])
    block_sums = csum[starts + L] - csum[starts]
    means = block_sums.sum(axis=1) / (n_blocks * L)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def dependence_aware_interval(flags, start_ordinals, horizon_days: int, confidence: float = 0.95,
                              n_boot: int = 2000, seed: int = 7) -> dict:
    """95% interval for a pass probability measured on OVERLAPPING purchase windows.

    Consecutive weekday starts share all but a day or two of their horizon, so N starts are nowhere near N independent trials. Two
    estimators that each respect that are computed and the WIDER one is reported:

      * moving-block bootstrap with blocks of two horizons (overlap plus some regime persistence);
      * Wilson interval on DISJOINT windows (one start per ``horizon_days`` bin, averaged over several bin origins).
    """
    x = np.asarray(flags, float)
    o = np.asarray(start_ordinals, int)
    n = len(x)
    alpha = 1.0 - confidence
    z = Z95 if abs(confidence - 0.95) < 1e-9 else float(_norm_ppf(1.0 - alpha / 2.0))
    if n == 0:
        return {"lower": 0.0, "upper": 1.0, "n_windows": 0, "n_independent": 0.0, "block_len": 0, "bootstrap": (0.0, 1.0),
                "wilson_disjoint": (0.0, 1.0), "method": "no windows"}
    p = float(x.mean())
    gaps = np.diff(o)
    spacing = float(np.median(gaps)) if len(gaps) else 1.0
    spacing = max(spacing, 1.0)
    block_len = int(max(1, math.ceil(2.0 * horizon_days / spacing)))
    boot = moving_block_bootstrap_ci(x, block_len, n_boot=n_boot, seed=seed, alpha=alpha)
    ks, ns = [], []
    step = max(1, horizon_days // 6)
    for off in range(0, horizon_days, step):
        bins = (o - (int(o[0]) + off)) // horizon_days
        valid = bins >= 0
        if not valid.any():
            continue
        _, first_idx = np.unique(bins[valid], return_index=True)
        sel = np.flatnonzero(valid)[first_idx]
        ns.append(len(sel))
        ks.append(float(x[sel].mean()) if len(sel) else 0.0)
    n_eff = float(np.mean(ns)) if ns else 0.0
    p_disjoint = float(np.mean(ks)) if ks else p
    w = wilson_interval(p_disjoint, n_eff, z=z)
    lower = float(min(boot[0], w[0], p))
    upper = float(max(boot[1], w[1], p))
    return {"lower": lower, "upper": upper, "n_windows": int(n), "n_independent": n_eff, "block_len": block_len,
            "bootstrap": (float(boot[0]), float(boot[1])), "wilson_disjoint": (float(w[0]), float(w[1])),
            "method": "wider of moving-block bootstrap (2-horizon blocks) and disjoint-window Wilson"}


def _norm_ppf(p: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


# ------------------------------------------------------------------------------------------------------------------ config
@dataclass(frozen=True)
class ThirtyDayConfig:
    horizon_days: int = DEFAULT_HORIZON_DAYS
    long_horizon_days: int = DEFAULT_LONG_HORIZON_DAYS
    start_stride_days: int = 1                 # purchase every N weekdays (1 = every weekday)
    require_full_coverage: bool = True         # a window must fit inside the data for the LONG horizon
    confidence: float = 0.95
    n_boot: int = 2000
    seed: int = 7
    target_probability: float = TARGET_PASS_PROBABILITY
    min_independent_windows: float = 8.0       # disjoint 30-day windows needed before any claim
    cost_multipliers: tuple = (1.0, 1.25, 1.5, 2.0)
    streak_levels: tuple = (3, 4, 5, 6, 8, 10)
    allow_proxy_excursions: bool = False       # explicit opt-in to closed-P&L MAE/MFE (optimistic under floating rules)
    min_regime_windows: int = 20


# ----------------------------------------------------------------------------------------------------------------- results
@dataclass
class ThirtyDayResult:
    horizon_days: int
    long_horizon_days: int
    n_windows: int
    first_purchase: str | None
    last_purchase: str | None
    n_trades: int
    p_pass_30: float
    p_pass_60: float
    p_bust_30: float
    p_open_30: float
    pass_by_day: dict                       # {elapsed calendar day: cumulative share passed by then}, 0..long horizon
    expected_days_to_pass_given_pass: float | None
    median_days_to_pass: float | None
    restricted_mean_days_to_pass: float     # non-passers counted at the long horizon
    p_trailing_dd_breach_30: float
    p_daily_loss_breach_30: float           # a session day whose FLOATING loss reached the daily limit (soft or hard), approx from MAE
    p_daily_loss_account_fail_30: float
    max_intraday_dd_usd: dict               # {mean, p95, max} over windows (floating, from the realized high-water mark)
    max_intraday_dd_pct: dict
    expected_return_per_trading_day_usd: float
    expected_return_per_trading_day_pct: float
    trades_per_30_days: float
    trades_per_trading_day: float
    losing_streak_ge: dict                  # {k: share of windows whose longest losing streak inside the horizon >= k}
    interval: dict                          # dependence-aware 95% interval on p_pass_30 (see dependence_aware_interval)
    meets_target_lower_bound: bool
    target_probability: float
    cost_sensitivity: dict                  # {multiplier: {"p_pass_30": .., "lower": ..}}
    regimes: dict                           # {label: {"n_windows", "p_pass_30", "lower", "upper", "n_independent"}}
    worst_regime: str | None
    excursions_measured: float
    notes: list = field(default_factory=list)
    windows: list = field(default_factory=list)

    @property
    def lower_bound(self) -> float:
        return float(self.interval.get("lower", 0.0))

    def to_dict(self, include_windows: bool = False) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "windows"}
        d["lower_bound"] = self.lower_bound
        if include_windows:
            d["windows"] = list(self.windows)
        return d

    def verdict(self) -> str:
        if self.n_windows == 0:
            return "NO WINDOWS: the data/trades cannot support a 30-day evaluation."
        if self.meets_target_lower_bound:
            return (f"MEASURED: P(pass<= {self.horizon_days}d) = {self.p_pass_30 * 100:.1f}%, 95% lower bound {self.lower_bound * 100:.1f}% "
                    f">= {self.target_probability * 100:.0f}% on this data. (Statistical result only -- see the independence gate before claiming it.)")
        return (f"NOT ESTABLISHED: P(pass<= {self.horizon_days}d) = {self.p_pass_30 * 100:.1f}%, 95% lower bound {self.lower_bound * 100:.1f}% "
                f"< {self.target_probability * 100:.0f}% target.")

    def render(self) -> str:
        lo, hi = self.interval.get("lower", 0.0), self.interval.get("upper", 1.0)
        lines = [
            "T58 30-DAY PROP EVALUATION OPTIMIZER",
            f"Windows (weekday purchases)   {self.n_windows:,}  [{self.first_purchase} .. {self.last_purchase}]   trades {self.n_trades:,}",
            f"P(pass within {self.horizon_days} calendar days)   {self.p_pass_30 * 100:.1f}%   (95% dependence-aware CI {lo * 100:.1f}-{hi * 100:.1f}%, "
            f"~{self.interval.get('n_independent', 0):.1f} independent windows)",
            f"P(pass within {self.long_horizon_days} calendar days)   {self.p_pass_60 * 100:.1f}%",
            f"P(bust within {self.horizon_days}d) {self.p_bust_30 * 100:.1f}%   P(still open) {self.p_open_30 * 100:.1f}%",
            "Expected calendar days to pass (given a pass)  " + (f"{self.expected_days_to_pass_given_pass:.1f}" if self.expected_days_to_pass_given_pass is not None else "n/a")
            + f"   restricted mean (non-passers at {self.long_horizon_days}d) {self.restricted_mean_days_to_pass:.1f}",
            f"P(trailing-drawdown breach, {self.horizon_days}d) {self.p_trailing_dd_breach_30 * 100:.1f}%   P(daily-loss breach) {self.p_daily_loss_breach_30 * 100:.1f}%",
            f"Max floating DD   mean ${self.max_intraday_dd_usd.get('mean', 0):,.0f}  p95 ${self.max_intraday_dd_usd.get('p95', 0):,.0f}  worst ${self.max_intraday_dd_usd.get('max', 0):,.0f}",
            f"Expected return per trading day  ${self.expected_return_per_trading_day_usd:,.1f} ({self.expected_return_per_trading_day_pct:.3f}% of account)   "
            f"trades / 30 days {self.trades_per_30_days:.1f}",
            "Cost sensitivity  " + "  ".join(f"x{m:g}: {v['p_pass_30'] * 100:.1f}% (lb {v['lower'] * 100:.1f}%)" for m, v in self.cost_sensitivity.items()),
            "Longest losing streak inside the window  " + "  ".join(f">={k}: {v * 100:.0f}%" for k, v in self.losing_streak_ge.items()),
            self.verdict(),
        ]
        lines += [f"NOTE: {n}" for n in self.notes]
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------------------------- evaluation
class ExcursionsMissingError(ValueError):
    """The rules are floating-basis but the ledger has no measured MAE/MFE; refusing to produce an optimistic number silently."""


def _floating_rules(rules: PropRules) -> bool:
    return (getattr(rules, "dd_basis", "legacy") in ("floating", "eod") or getattr(rules, "daily_loss_basis", "realized") == "floating"
            or getattr(rules, "drawdown_check_mode", "") == "intrabar")


def _session_ordinals(times: np.ndarray, rules: PropRules) -> np.ndarray:
    days = trading_days(pd.DatetimeIndex(times), tz=rules.session_timezone, roll_hour=rules.session_roll_hour)
    return np.asarray([pd.Timestamp(d).toordinal() for d in days], dtype=np.int64)


def _fail_category(reason: str | None) -> str:
    r = (reason or "").lower()
    if r.startswith("daily"):
        return "daily_loss"
    if r.startswith("max_drawdown") or "drawdown" in r:
        return "trailing_drawdown"
    return "other"


def _window_path_stats(pnl: np.ndarray, mae: np.ndarray, exit_ord: np.ndarray, daily_limit: float) -> tuple:
    """(max floating DD $, daily-loss floating breach?) over a trade path -- sequential approximation from MAE (exact for a one-position
    strategy; conservative-ish for overlapping books)."""
    bal = peak = max_dd = 0.0
    cur_day = None
    day_start = 0.0
    breach = False
    for p, m, d in zip(pnl, mae, exit_ord):
        if d != cur_day:
            cur_day, day_start = d, bal
        low = bal + min(float(m), 0.0)
        if peak - low > max_dd:
            max_dd = peak - low
        if low - day_start <= -daily_limit:
            breach = True
        bal += float(p)
        if bal > peak:
            peak = bal
    return float(max_dd), bool(breach)


def _longest_losing_streak(pnl: np.ndarray) -> int:
    best = cur = 0
    for p in pnl:
        if p < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _build_windows(ledger: TradeLedger, rules: PropRules, cfg: ThirtyDayConfig, data_start, data_end) -> tuple:
    entry_ord = _session_ordinals(ledger.entry_time, rules)
    exit_ord = _session_ordinals(ledger.exit_time, rules)
    d0 = pd.Timestamp(data_start) if data_start is not None else pd.Timestamp(ledger.entry_time[0])
    d1 = pd.Timestamp(data_end) if data_end is not None else pd.Timestamp(ledger.exit_time.max())
    if d0.tzinfo is not None:
        d0 = d0.tz_convert("UTC").tz_localize(None)
    if d1.tzinfo is not None:
        d1 = d1.tz_convert("UTC").tz_localize(None)
    o0 = int(_session_ordinals(np.asarray([np.datetime64(d0)], dtype="datetime64[ns]"), rules)[0])
    o1 = int(_session_ordinals(np.asarray([np.datetime64(d1)], dtype="datetime64[ns]"), rules)[0])
    need = cfg.long_horizon_days if cfg.require_full_coverage else cfg.horizon_days
    last = o1 - need
    starts = []
    for o in range(o0, last + 1):
        if pd.Timestamp.fromordinal(o).weekday() < 5:
            starts.append(o)
    starts = starts[:: max(1, int(cfg.start_stride_days))]
    return entry_ord, exit_ord, starts


def _run_window(ledger, entry_ord, exit_ord, lo, hi, purchase_ord, rules, cfg, daily_limit) -> dict:
    out = {"purchase_ord": purchase_ord, "outcome": "open", "elapsed": None, "category": None, "n_trades": int(hi - lo),
           "max_dd": 0.0, "daily_breach": False, "pnl_30": 0.0, "streak": 0, "weekdays_30": 0}
    H = cfg.horizon_days
    wd = sum(1 for k in range(0, H + 1) if pd.Timestamp.fromordinal(purchase_ord + k).weekday() < 5)
    out["weekdays_30"] = wd
    if hi <= lo:
        return out
    sl = slice(lo, hi)
    pnl, mae, mfe = ledger.pnl[sl], ledger.mae[sl], ledger.mfe[sl]
    ex_times = [pd.Timestamp(t) for t in ledger.exit_time[sl]]
    en_times = [pd.Timestamp(t) for t in ledger.entry_time[sl]]
    ds = precompute_day_structure(ex_times, tz=rules.session_timezone, roll_hour=rules.session_roll_hour)
    res = simulate_account(pnl.tolist(), ex_times, rules, _day_structure=ds, trade_mae=mae.tolist(), trade_mfe=mfe.tolist(),
                           trade_entry_times=en_times, trade_exit_times=ex_times)
    first_ord = int(ds.day_dates[0].toordinal())
    end_ord = None
    if res.passed_evaluation and res.calendar_days_to_pass is not None:
        pass_ord = first_ord + int(res.calendar_days_to_pass) - 1
        out["outcome"], out["elapsed"], end_ord = "passed", pass_ord - purchase_ord, pass_ord
    elif res.failed and res.failure_day_index is not None:
        fail_ord = int(ds.day_dates[int(res.failure_day_index)].toordinal())
        out["outcome"], out["elapsed"], end_ord = "failed", fail_ord - purchase_ord, fail_ord
        out["category"] = _fail_category(res.failure_reason)
    # raw strategy path over the first horizon (for return / streak / floating DD statistics), truncated where the account ended
    horizon_end = purchase_ord + H
    stop_at = min(horizon_end, end_ord) if end_ord is not None else horizon_end
    in_h = exit_ord[sl] <= stop_at
    if in_h.any():
        out["max_dd"], out["daily_breach"] = _window_path_stats(pnl[in_h], mae[in_h], exit_ord[sl][in_h], daily_limit)
    in_raw = exit_ord[sl] <= horizon_end
    out["pnl_30"] = float(pnl[in_raw].sum())
    out["streak"] = _longest_losing_streak(pnl[in_raw])
    out["n_trades_30"] = int(in_raw.sum())
    return out


def _empty_result(cfg: ThirtyDayConfig, notes: list) -> ThirtyDayResult:
    return ThirtyDayResult(
        horizon_days=cfg.horizon_days, long_horizon_days=cfg.long_horizon_days, n_windows=0, first_purchase=None, last_purchase=None,
        n_trades=0, p_pass_30=0.0, p_pass_60=0.0, p_bust_30=0.0, p_open_30=1.0, pass_by_day={}, expected_days_to_pass_given_pass=None,
        median_days_to_pass=None, restricted_mean_days_to_pass=float(cfg.long_horizon_days), p_trailing_dd_breach_30=0.0,
        p_daily_loss_breach_30=0.0, p_daily_loss_account_fail_30=0.0, max_intraday_dd_usd={}, max_intraday_dd_pct={},
        expected_return_per_trading_day_usd=0.0, expected_return_per_trading_day_pct=0.0, trades_per_30_days=0.0,
        trades_per_trading_day=0.0, losing_streak_ge={}, interval={"lower": 0.0, "upper": 1.0, "n_windows": 0, "n_independent": 0.0},
        meets_target_lower_bound=False, target_probability=cfg.target_probability, cost_sensitivity={}, regimes={}, worst_regime=None,
        excursions_measured=0.0, notes=notes)


def evaluate_thirty_day(ledger: TradeLedger, rules: PropRules, *, data_start=None, data_end=None, regime_labels=None,
                        cfg: ThirtyDayConfig | None = None, with_sensitivity: bool = True) -> ThirtyDayResult:
    """Score a trade ledger on P(pass within ``cfg.horizon_days`` calendar days of purchase). See the module docstring.

    ``data_start`` / ``data_end``: the span of PRICE DATA the ledger was produced from. They define which purchase days exist, so a
    quiet stretch with no trades still counts as time in which an account would have waited. Defaults to the trade span, which
    UNDERSTATES waiting time at the edges; pass the data span whenever it is known.
    ``regime_labels``: optional ``{name: pd.Series}`` mapping a date to a label (e.g. volatility tercile). Looked up causally on the
    purchase date (last label at or before it). Calendar-year labels are always added.
    """
    cfg = cfg or ThirtyDayConfig()
    notes: list = []
    if len(ledger) == 0:
        return _empty_result(cfg, ["The ledger has no trades: nothing to evaluate."])
    if _floating_rules(rules) and ledger.excursions_measured < 0.999 and not cfg.allow_proxy_excursions:
        raise ExcursionsMissingError(
            f"{(1 - ledger.excursions_measured) * 100:.0f}% of the trades carry no measured MAE/MFE but the rules use floating drawdown / "
            "daily loss. Re-run the engine so trades carry worst_price/mfe_price, or set ThirtyDayConfig(allow_proxy_excursions=True) "
            "to accept the closed-P&L proxy (optimistic: it hides every intratrade dip).")
    if ledger.excursions_measured < 0.999:
        notes.append("MAE/MFE are the closed-P&L proxy for some trades: intratrade drawdown is understated.")
    rules = replace(rules, max_eval_calendar_days=None)        # the horizon is applied here, in calendar days from PURCHASE
    entry_ord, exit_ord, starts = _build_windows(ledger, rules, cfg, data_start, data_end)
    if not starts:
        return _empty_result(cfg, ["The data span is shorter than the long horizon: no purchase window fits inside it."])
    daily_limit = float(rules.account_size) * float(rules.daily_loss_limit_pct) / 100.0
    H, HL = cfg.horizon_days, cfg.long_horizon_days
    rows = []
    for p_ord in starts:
        lo = int(np.searchsorted(entry_ord, p_ord, side="left"))
        hi = int(np.searchsorted(entry_ord, p_ord + HL, side="right"))
        rows.append(_run_window(ledger, entry_ord, exit_ord, lo, hi, p_ord, rules, cfg, daily_limit))

    elapsed = np.asarray([r["elapsed"] if r["elapsed"] is not None else -1 for r in rows], dtype=float)
    outcome = np.asarray([r["outcome"] for r in rows])
    passed = outcome == "passed"
    p30 = passed & (elapsed <= H)
    p60 = passed & (elapsed <= HL)
    bust30 = (outcome == "failed") & (elapsed <= H)
    n = len(rows)
    flags30 = p30.astype(float)
    ords = np.asarray(starts, int)
    interval = dependence_aware_interval(flags30, ords, H, confidence=cfg.confidence, n_boot=cfg.n_boot, seed=cfg.seed)
    pass_by_day = {int(d): float(np.mean(passed & (elapsed <= d))) for d in range(0, HL + 1)}
    days60 = elapsed[p60]
    cat = np.asarray([r["category"] or "" for r in rows])
    dd = np.asarray([r["max_dd"] for r in rows], float)
    acct = float(rules.account_size)
    pnl30 = np.asarray([r["pnl_30"] for r in rows], float)
    wd = np.asarray([r["weekdays_30"] for r in rows], float)
    nt30 = np.asarray([r.get("n_trades_30", 0) for r in rows], float)
    streak = np.asarray([r["streak"] for r in rows])
    dbreach = np.asarray([r["daily_breach"] for r in rows])
    ret_usd = float(pnl30.sum() / max(wd.sum(), 1.0))
    result = ThirtyDayResult(
        horizon_days=H, long_horizon_days=HL, n_windows=n,
        first_purchase=str(pd.Timestamp.fromordinal(starts[0]).date()), last_purchase=str(pd.Timestamp.fromordinal(starts[-1]).date()),
        n_trades=int(len(ledger)), p_pass_30=float(p30.mean()), p_pass_60=float(p60.mean()), p_bust_30=float(bust30.mean()),
        p_open_30=float(1.0 - p30.mean() - bust30.mean()), pass_by_day=pass_by_day,
        expected_days_to_pass_given_pass=(float(days60.mean()) if days60.size else None),
        median_days_to_pass=(float(np.median(days60)) if days60.size else None),
        restricted_mean_days_to_pass=float(np.where(p60, elapsed, HL).mean()),
        p_trailing_dd_breach_30=float(((cat == "trailing_drawdown") & bust30).mean()),
        p_daily_loss_breach_30=float(dbreach.mean()),
        p_daily_loss_account_fail_30=float(((cat == "daily_loss") & bust30).mean()),
        max_intraday_dd_usd={"mean": float(dd.mean()), "p95": float(np.percentile(dd, 95)), "max": float(dd.max())},
        max_intraday_dd_pct={"mean": float(dd.mean() / acct * 100), "p95": float(np.percentile(dd, 95) / acct * 100), "max": float(dd.max() / acct * 100)},
        expected_return_per_trading_day_usd=ret_usd, expected_return_per_trading_day_pct=float(ret_usd / acct * 100.0),
        trades_per_30_days=float(nt30.mean()), trades_per_trading_day=float(nt30.sum() / max(wd.sum(), 1.0)),
        losing_streak_ge={int(k): float(np.mean(streak >= k)) for k in cfg.streak_levels},
        interval=interval, meets_target_lower_bound=bool(interval["lower"] >= cfg.target_probability
                                                         and interval["n_independent"] >= cfg.min_independent_windows),
        target_probability=cfg.target_probability, cost_sensitivity={}, regimes={}, worst_regime=None,
        excursions_measured=float(ledger.excursions_measured), notes=notes,
        windows=[{"purchase": str(pd.Timestamp.fromordinal(r["purchase_ord"]).date()), "outcome": r["outcome"], "elapsed_days": r["elapsed"],
                  "n_trades": r["n_trades"], "category": r["category"]} for r in rows],
    )
    if interval["n_independent"] < cfg.min_independent_windows:
        notes.append(f"only ~{interval['n_independent']:.1f} independent {H}-day windows (< {cfg.min_independent_windows:g}): "
                     "the target cannot be called achieved on this much data, whatever the point estimate.")
    result.regimes = _regime_table(rows, flags30, ords, regime_labels, cfg)
    scored = {k: v for k, v in result.regimes.items() if v["n_windows"] >= cfg.min_regime_windows}
    result.worst_regime = min(scored, key=lambda k: scored[k]["p_pass_30"]) if scored else None
    if with_sensitivity:
        for m in cfg.cost_multipliers:
            if m == 1.0:
                result.cost_sensitivity[1.0] = {"p_pass_30": result.p_pass_30, "lower": result.lower_bound}
                continue
            sub = evaluate_thirty_day(ledger.with_cost_multiplier(m), rules, data_start=data_start, data_end=data_end,
                                      cfg=replace(cfg, cost_multipliers=(1.0,)), with_sensitivity=False)
            result.cost_sensitivity[float(m)] = {"p_pass_30": sub.p_pass_30, "lower": sub.lower_bound}
    return result


def _regime_table(rows, flags30, ords, regime_labels, cfg) -> dict:
    labels: dict = {}
    years = np.asarray([pd.Timestamp.fromordinal(int(o)).year for o in ords])
    for y in sorted(set(years.tolist())):
        labels[f"year:{y}"] = years == y
    for name, series in (regime_labels or {}).items():
        s = pd.Series(series).copy()
        s.index = pd.DatetimeIndex(pd.to_datetime(s.index)).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        d = pd.DatetimeIndex([pd.Timestamp.fromordinal(int(o)) for o in ords])
        look = s.reindex(s.index.union(d)).ffill().reindex(d)
        vals = look.to_numpy(dtype=object)
        for v in sorted({str(x) for x in vals if x is not None and str(x) != "nan"}):
            labels[f"{name}:{v}"] = np.asarray([str(x) == v for x in vals])
    table = {}
    for lab, mask in labels.items():
        k = int(mask.sum())
        if k == 0:
            continue
        iv = dependence_aware_interval(flags30[mask], ords[mask], cfg.horizon_days, confidence=cfg.confidence, n_boot=max(200, cfg.n_boot // 4),
                                       seed=cfg.seed)
        # a regime subset is not contiguous: bound the independent count by what its calendar span can hold
        span = max(int(ords[mask].max() - ords[mask].min()), 1)
        iv_n = min(iv["n_independent"], max(1.0, span / cfg.horizon_days + 1.0), k)
        if iv_n < iv["n_independent"]:
            w = wilson_interval(float(flags30[mask].mean()), iv_n)
            iv["lower"], iv["upper"] = min(iv["lower"], w[0]), max(iv["upper"], w[1])
        table[lab] = {"n_windows": k, "p_pass_30": float(flags30[mask].mean()), "lower": iv["lower"], "upper": iv["upper"], "n_independent": float(iv_n)}
    return table


def claim_target_achieved(result: ThirtyDayResult) -> bool:
    """Statistical check only: the 95% dependence-aware LOWER bound reaches the target on enough independent windows AND at 1.5x costs
    the strategy still has a positive-edge profile (point estimate above half the target). Independence of the data is checked
    separately by ``app.validation.prop30_validation``."""
    if not result.meets_target_lower_bound:
        return False
    stressed = result.cost_sensitivity.get(1.5)
    if stressed is not None and stressed["p_pass_30"] < 0.5 * result.target_probability:
        return False
    return True


def rank_key(result: ThirtyDayResult) -> tuple:
    """Sort key for candidates, best first: the CI LOWER bound on P(pass<=30d) (what we can defend), then the point estimate, then
    speed, then fewer breaches. Deliberately blind to Sharpe, profit factor, annual return and eventual pass probability."""
    return (-result.lower_bound, -result.p_pass_30, result.restricted_mean_days_to_pass, result.p_bust_30)
