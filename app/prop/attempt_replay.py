"""
ATTEMPT REPLAY (2026-10-07 accuracy overhaul) -- "if I bought a fresh account
on day X and traded this strategy, what happens?", answered by running the
REAL bar engine from day X, not by re-adding a pre-recorded trade list.

Why the old rolling evaluation was not enough: it re-fed one fixed sequence of
trade P&Ls to simulate_account from different start days. But a trade's size
depends on the equity of the account it is traded in, and on the floor it is
sitting above (fit_stop / skip decisions, adaptive risk, the daily-loss lock),
so the sequence a fresh account would have produced is a DIFFERENT sequence.
Here every start date gets its own run of `run_execution` in
account_model="prop", attempt_mode="single", stop_on_pass=True: one account,
the firm's rules, ending at pass, bust or the evaluation time limit.

Statistics reported (all per start date, i.e. per purchased account):
  pass_rate, bust_rate (account breached before passing), open_rate (ran out of
  data/time with neither), bust_before_pass among terminated attempts,
  expected_attempts_to_pass (= 1 / pass_rate: the geometric number of purchased
  accounts per pass), expected_cost_per_pass when a fee is supplied, median
  days to pass, a Wilson interval, and an honesty flag when the number of
  attempts is below `min_attempts`.

Signals are computed ONCE on the full history by the caller (indicators warm
up on history that precedes the start date, exactly like live trading with a
chart that has history) and sliced with the data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import run_execution

MIN_ATTEMPTS_DEFAULT = 30


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


@dataclass
class AttemptOutcome:
    start_bar: int
    start_time: str
    outcome: str                      # "passed" | "failed" | "open"
    failure_reason: str | None
    days_to_pass: int | None
    n_trades: int
    end_balance: float
    bust_before_pass: bool

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class AttemptReplayResult:
    n_attempts: int
    n_passed: int
    n_failed: int
    n_open: int
    pass_rate: float
    bust_rate: float
    open_rate: float
    pass_ci: tuple
    bust_before_pass_rate: float      # busts / (busts + passes): of attempts that RESOLVED, how many died
    expected_attempts_to_pass: float | None
    expected_cost_per_pass: float | None
    median_days_to_pass: float | None
    failure_breakdown: dict
    sample_ok: bool
    min_attempts: int
    notes: list = field(default_factory=list)
    attempts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["attempts"] = [a.to_dict() for a in self.attempts]
        return d

    def render(self) -> str:
        lo, hi = self.pass_ci
        lines = [
            "T58 ATTEMPT REPLAY -- fresh account at every start date, real engine",
            f"Attempts            {self.n_attempts:,}  (passed {self.n_passed:,} / busted {self.n_failed:,} / unresolved {self.n_open:,})",
            f"Pass rate           {self.pass_rate * 100:.1f}%   (95% CI {lo * 100:.1f}-{hi * 100:.1f}%)",
            f"Bust rate           {self.bust_rate * 100:.1f}%",
            f"Bust before pass    {self.bust_before_pass_rate * 100:.1f}%  (of attempts that resolved)",
            "Expected accounts per pass  " + (f"{self.expected_attempts_to_pass:.2f}" if self.expected_attempts_to_pass else "n/a (no passes)"),
        ]
        if self.expected_cost_per_pass is not None:
            lines.append(f"Expected fee cost per pass  ${self.expected_cost_per_pass:,.0f}")
        if self.median_days_to_pass is not None:
            lines.append(f"Median days to pass {self.median_days_to_pass:.0f}")
        lines += [f"NOTE: {n}" for n in self.notes]
        return "\n".join(lines)


def _slice_kw(value, start: int, stop: int, n: int):
    if isinstance(value, (pd.Series, pd.DataFrame)) and len(value) == n:
        return value.iloc[start:stop]
    return value


def _replay_one_start(s: int, df, signals, risk_p, rules, stop_loss_pips,
                      take_profit_pips, horizon_days: int, stop_on_pass: bool,
                      exec_kwargs: dict) -> AttemptOutcome:
    """One fresh-account attempt from start bar `s` -- the serial loop's
    per-start body, verbatim (v9.13), so the serial and parallel paths
    share one implementation and produce identical outcomes."""
    import warnings

    n = len(df)
    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    end_t = ts[s] + pd.Timedelta(days=horizon_days)
    e = int(np.searchsorted(ts.values, np.datetime64(end_t)))
    e = min(max(e, s + 10), n)
    sub = df.iloc[s:e].reset_index(drop=True)
    sig = signals.iloc[s:e].reset_index(drop=True)
    kw = {k: (v.iloc[s:e].reset_index(drop=True) if isinstance(v, pd.Series) and len(v) == n else v)
          for k, v in exec_kwargs.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trades, eq = run_execution(
            sub, sig, risk_p, stop_loss_pips, take_profit_pips,
            attempt_mode="single", stop_on_pass=stop_on_pass, **kw,
        )
    atts = eq.attrs.get("prop_attempts") or []
    a = atts[0] if atts else {"outcome": "open", "failure_reason": None, "days_to_pass": None,
                              "n_trades": 0, "end_balance": float(rules.account_size)}
    outcome = a["outcome"]
    if outcome == "passed_funded":
        outcome = "passed"
    if outcome == "failed" and a.get("passed_evaluation"):
        outcome = "passed"   # passed the evaluation, later busted funded: still a pass
    return AttemptOutcome(
        start_bar=s, start_time=str(ts[s]), outcome=outcome,
        failure_reason=a.get("failure_reason") if outcome == "failed" else None,
        days_to_pass=a.get("days_to_pass"), n_trades=int(a.get("n_trades") or 0),
        end_balance=float(a.get("end_balance") or 0.0),
        bust_before_pass=(outcome == "failed"),
    )


# -- v9.13: cross-process attempt replay ---------------------------------
# Each start date is an independent fresh-account engine run; only the
# outcomes list order (start order) matters downstream, so workers
# return AttemptOutcomes that the parent slots back by start index --
# bit-identical to the serial loop. Spawn matches every other pool in
# this app; any pool failure finishes the remaining starts serially.
_REPLAY_WORKER: dict = {}


def _replay_worker_init(df, signals, risk_p, rules, stop_loss_pips,
                        take_profit_pips, horizon_days, stop_on_pass, exec_kwargs) -> None:
    global _REPLAY_WORKER
    _REPLAY_WORKER = {
        "df": df, "signals": signals, "risk_p": risk_p, "rules": rules,
        "stop_loss_pips": stop_loss_pips, "take_profit_pips": take_profit_pips,
        "horizon_days": horizon_days, "stop_on_pass": stop_on_pass,
        "exec_kwargs": exec_kwargs,
    }


def _replay_worker_task(s: int) -> AttemptOutcome:
    w = _REPLAY_WORKER
    return _replay_one_start(
        s, w["df"], w["signals"], w["risk_p"], w["rules"],
        w["stop_loss_pips"], w["take_profit_pips"], w["horizon_days"],
        w["stop_on_pass"], w["exec_kwargs"],
    )


def run_attempt_replay(
    df: pd.DataFrame,
    signals: pd.Series,
    risk,
    rules,
    stop_loss_pips,
    take_profit_pips,
    *,
    horizon_days: int | None = None,
    stride_bars: int | None = None,
    n_starts: int = 200,
    min_attempts: int = MIN_ATTEMPTS_DEFAULT,
    eval_fee: float | None = None,
    stop_on_pass: bool = True,
    keep_attempts: bool = True,
    max_workers: int | None = None,
    progress_cb=None,
    **exec_kwargs,
) -> AttemptReplayResult:
    """Run `n_starts` evenly spread fresh-account attempts.

    horizon_days: calendar days of data given to each attempt (default: the
    rules' max_eval_calendar_days, else 120). stride_bars overrides n_starts.
    Remaining keyword args (stop_loss_distance, take_profit_distance,
    trailing_stop_distance, breakeven_trigger_r, partial_exit_config,
    adaptive_risk, intrabar_df) are forwarded; per-bar Series are sliced.

    max_workers (v9.13): when > 1, the independent start-date attempts
    run across worker processes and are collected back in start order --
    bit-identical outcomes, only faster. None (the default) keeps every
    existing caller on the exact serial path. progress_cb, when given,
    is invoked as (done, total) as attempts complete; purely
    observational.
    """
    from dataclasses import replace
    n = len(df)
    if n < 50:
        raise ValueError("Not enough bars for an attempt replay.")
    if horizon_days is None:
        horizon_days = int(getattr(rules, "max_eval_calendar_days", None) or 120)
    risk_p = replace(risk, account_model="prop", prop_account_rules=rules)

    ts = pd.DatetimeIndex(df["timestamp"])
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    last_possible = ts[-1] - pd.Timedelta(days=max(1, horizon_days // 4))
    max_start = int(np.searchsorted(ts.values, np.datetime64(last_possible)))
    max_start = max(1, min(max_start, n - 2))
    if stride_bars:
        starts = list(range(0, max_start, int(stride_bars)))
    else:
        starts = sorted(set(int(x) for x in np.linspace(0, max_start - 1, num=min(n_starts, max_start))))

    outcomes: list = [None] * len(starts)

    def _one(s: int) -> AttemptOutcome:
        return _replay_one_start(
            s, df, signals, risk_p, rules, stop_loss_pips, take_profit_pips,
            horizon_days, stop_on_pass, exec_kwargs,
        )

    def _fire(done_count: int) -> None:
        if progress_cb is not None:
            try:
                progress_cb(done_count, len(starts))
            except Exception:  # noqa: BLE001 -- progress must never sink the run
                pass

    done_up_to = 0
    if max_workers is not None and int(max_workers) > 1 and len(starts) >= 4:
        import multiprocessing
        from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor
        from concurrent.futures import wait as _futures_wait

        pool = None
        try:
            pool = ProcessPoolExecutor(
                max_workers=int(max_workers),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_replay_worker_init,
                initargs=(df, signals, risk_p, rules, stop_loss_pips,
                          take_profit_pips, horizon_days, stop_on_pass, exec_kwargs),
            )
            futures = {pool.submit(_replay_worker_task, s): i for i, s in enumerate(starts)}
            pending = set(futures)
            completed = 0
            while pending:
                finished, pending = _futures_wait(pending, return_when=FIRST_COMPLETED)
                for fut in finished:
                    outcomes[futures[fut]] = fut.result()
                    completed += 1
                _fire(completed)
            done_up_to = len(starts)
        except Exception:  # noqa: BLE001 -- fall back to serial for the rest
            done_up_to = 0
        finally:
            if pool is not None:
                try:
                    pool.shutdown(wait=False, cancel_futures=True)
                except Exception:  # noqa: BLE001
                    pass
    for i in range(done_up_to, len(starts)):
        outcomes[i] = _one(starts[i])
        _fire(i + 1)

    N = len(outcomes)
    passed = sum(o.outcome == "passed" for o in outcomes)
    failed = sum(o.outcome == "failed" for o in outcomes)
    opened = N - passed - failed
    resolved = passed + failed
    pass_rate = passed / N if N else 0.0
    days = sorted(o.days_to_pass for o in outcomes if o.outcome == "passed" and o.days_to_pass)
    notes = []
    sample_ok = N >= min_attempts
    if not sample_ok:
        notes.append(f"only {N} attempts (< {min_attempts}): the pass rate is not statistically meaningful.")
    if opened / N > 0.3 if N else False:
        notes.append(f"{opened / N * 100:.0f}% of attempts neither passed nor busted inside {horizon_days} days; "
                     "the target may be unreachable at this risk level or the strategy trades too rarely.")
    from collections import Counter
    return AttemptReplayResult(
        n_attempts=N, n_passed=passed, n_failed=failed, n_open=opened,
        pass_rate=pass_rate, bust_rate=(failed / N if N else 0.0), open_rate=(opened / N if N else 0.0),
        pass_ci=_wilson(passed, N),
        bust_before_pass_rate=(failed / resolved if resolved else 0.0),
        expected_attempts_to_pass=(1.0 / pass_rate if pass_rate > 0 else None),
        expected_cost_per_pass=((eval_fee / pass_rate) if (eval_fee and pass_rate > 0) else None),
        median_days_to_pass=(float(days[len(days) // 2]) if days else None),
        failure_breakdown=dict(Counter(o.failure_reason for o in outcomes if o.outcome == "failed")),
        sample_ok=sample_ok, min_attempts=min_attempts, notes=notes,
        attempts=outcomes if keep_attempts else [],
    )
