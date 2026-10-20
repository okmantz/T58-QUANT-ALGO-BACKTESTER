"""
Backtest engine.

Orchestrates: Dataset + Strategy + Risk/Execution Configuration
           -> Trade List + Equity Curve + Statistics
"""
from __future__ import annotations

import warnings as _warnings_module
from dataclasses import dataclass, field

import pandas as pd

from app.backtest.adaptive_risk import AdaptiveRiskConfig
from app.backtest.execution import Trade, run_execution
from app.backtest.risk import RiskConfig, position_sizing_deviation_message
from app.backtest.statistics import BacktestStatistics, compute_statistics
from app.data.timeframe_resample import prepare_timeframe_aligned_data
from app.strategy.base import Strategy, StrategyResult, apply_days_of_week_exclusion, apply_regime_exclusion


@dataclass
class BacktestResult:
    strategy_name: str
    trades: list[Trade]
    equity_curve: pd.DataFrame
    statistics: BacktestStatistics
    initial_balance: float
    warnings: list[str] = field(default_factory=list)
    # 2026-10 audit: True when the behavioural lookahead check was requested but CRASHED. A crash used to be
    # swallowed (``except Exception: result = None``), which read as "no leak found". It is now recorded here, emitted as
    # a RuntimeWarning in ``warnings``, and consumers (full_pipeline) demote the verdict on it.
    lookahead_check_failed: bool = False
    # ^ Execution-integrity warnings from app.backtest.execution.run_execution
    # (fallback stops, forced daily-limit closes, pip-size/instrument
    # mismatches, gap-through stop fills). run_execution raises these as
    # ordinary Python RuntimeWarnings; nothing upstream was ever catching
    # them, so on the desktop app (which doesn't surface stderr anywhere)
    # they were silently invisible no matter how serious. Capturing them
    # here, once, means every caller of run_backtest gets them for free.


_LOOKAHEAD_CACHE: dict = {}


def _structure_only(obj):
    """Config with every number blanked: two genomes that differ only in
    parameter values share a structure."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return 0
    if isinstance(obj, dict):
        return {k: _structure_only(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_structure_only(v) for v in obj]
    return str(type(obj))


def _lookahead_cache_key(strategy, df):
    import hashlib
    import json
    try:
        cfg = getattr(strategy, "config", None)
        if getattr(strategy, "source_type", None) == "manual" and isinstance(cfg, dict):
            body = json.dumps(_structure_only(cfg), sort_keys=True, default=str)
        else:
            code = getattr(strategy, "code", None) or getattr(strategy, "file_path", None) or repr(strategy)
            body = str(code)
        h = hashlib.sha1(body.encode("utf-8", "ignore")).hexdigest()
        return (type(strategy).__name__, h, len(df), str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1]))
    except Exception:
        return None


# -- 2026-10-15: persistent lookahead cache ---------------------------------------------------------------
# The in-memory cache above dies with the process, and for Python strategies its key hashes the file PATH, so an
# edited file under the same name kept a stale verdict. The disk cache is keyed on what actually determines the
# answer: the strategy's CONTENT hash, a strong fingerprint of the data it ran on, a fingerprint of the code that
# runs the check, and the check's parameters. Only checks that really ran (checked=True) are stored; a crash or a
# skip is never cached. Anything that goes wrong reading/writing the cache just means "recompute".
_LOOKAHEAD_DISK_SCHEMA = 1
_code_fp_memo: list = []


def _lookahead_disk_enabled() -> bool:
    import os
    return os.environ.get("T58_LOOKAHEAD_DISK_CACHE", "1").strip().lower() not in ("0", "false", "no", "off")


def _lookahead_cache_dir():
    import os
    from pathlib import Path
    override = os.environ.get("T58_CACHE_DIR")
    if override:
        return Path(override) / "lookahead"
    from app.data.storage import user_data_dir
    return Path(user_data_dir()) / "cache" / "lookahead"


def _lookahead_code_fingerprint() -> str:
    """Hash of the source that decides a lookahead verdict (strategy adapters + the checker + resampling)."""
    if _code_fp_memo:
        return _code_fp_memo[0]
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    h = hashlib.blake2b(digest_size=16)
    files = sorted((root / "strategy").glob("*.py")) + [root / "data" / "timeframe_resample.py"]
    for f in files:
        try:
            h.update(f.name.encode())
            h.update(f.read_bytes())
        except OSError:
            h.update(b"?")
    _code_fp_memo.append(h.hexdigest())
    return _code_fp_memo[0]


def _strategy_content_hash(strategy):
    import hashlib
    import json
    from pathlib import Path
    cfg = getattr(strategy, "config", None)
    if getattr(strategy, "source_type", None) == "manual" and isinstance(cfg, dict):
        body = json.dumps(_structure_only(cfg), sort_keys=True, default=str).encode("utf-8", "ignore")
    else:
        fp = getattr(strategy, "file_path", None)
        if fp is not None and Path(fp).is_file():
            body = Path(fp).read_bytes()          # CONTENT, not the path
        else:
            code = getattr(strategy, "code", None)
            if code is None:
                return None
            body = str(code).encode("utf-8", "ignore")
    h = hashlib.blake2b(digest_size=16)
    h.update(type(strategy).__name__.encode())
    h.update(b"\0")
    h.update(body)
    return h.hexdigest()


def _data_fingerprint(df) -> str:
    import hashlib
    import numpy as np
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{len(df)}|{','.join(map(str, df.columns))}".encode())
    if len(df):
        h.update(str(df["timestamp"].iloc[0]).encode())
        h.update(str(df["timestamp"].iloc[-1]).encode())
    for col in ("close", "high", "low"):
        if col in df.columns:
            h.update(np.ascontiguousarray(df[col].to_numpy(dtype="float64", copy=False)).tobytes())
    return h.hexdigest()


def _lookahead_disk_path(strategy, df):
    import hashlib
    sh = _strategy_content_hash(strategy)
    if sh is None:
        return None
    from app.strategy import lookahead_check as _lc
    params = repr((getattr(_lc, "DEFAULT_FRACTIONS", None), getattr(_lc, "MAX_SIGNAL_CHECKPOINTS", None),
                   getattr(_lc, "MIN_BARS_FOR_CHECK", None), _LOOKAHEAD_DISK_SCHEMA))
    key = hashlib.blake2b(
        f"{sh}|{_data_fingerprint(df)}|{_lookahead_code_fingerprint()}|{params}".encode(), digest_size=20
    ).hexdigest()
    return _lookahead_cache_dir() / f"{key}.json"


def _lookahead_disk_load(strategy, df):
    if not _lookahead_disk_enabled():
        return None
    try:
        import json
        from app.strategy.lookahead_check import LookaheadCheckResult
        path = _lookahead_disk_path(strategy, df)
        if path is None or not path.is_file():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != _LOOKAHEAD_DISK_SCHEMA:
            return None
        res = LookaheadCheckResult(**data["result"])
        return res if res.checked else None
    except Exception:  # noqa: BLE001 -- a bad cache entry is just a miss
        return None


def _lookahead_disk_store(strategy, df, result) -> None:
    if not _lookahead_disk_enabled() or result is None or not getattr(result, "checked", False):
        return
    try:
        import dataclasses
        import json
        import os
        path = _lookahead_disk_path(strategy, df)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps({"schema": _LOOKAHEAD_DISK_SCHEMA, "result": dataclasses.asdict(result)}),
                       encoding="utf-8")
        os.replace(tmp, path)                      # atomic: concurrent workers never see a half-written file
    except Exception:  # noqa: BLE001 -- caching must never break a backtest
        pass


def _run_resting_orders(df, strat_result, strategy, risk):
    """Resting-order / zone strategies: fills from app.backtest.resting_orders
    (same sizing and costs as the bar engine), then a per-bar equity curve."""
    import numpy as np

    from app.backtest.resting_orders import simulate_resting_orders

    cfg = getattr(strategy, "config", None)
    if isinstance(cfg, dict) and cfg.get("zone_entry"):
        from app.strategy.zones import check_zone_causality
        problems = check_zone_causality(cfg["zone_entry"], df, n_checks=4)
        if problems:
            _warnings_module.warn("LOOKAHEAD BIAS DETECTED in zone strategy: " + "; ".join(problems[:3]), RuntimeWarning)
    trades = simulate_resting_orders(df, strat_result.entry_orders, risk)
    ts = pd.to_datetime(df["timestamp"]).reset_index(drop=True)
    eq = np.full(len(df), float(risk.initial_balance))
    if trades:
        ex = pd.to_datetime([t.exit_time for t in trades])
        pos = np.searchsorted(ts.values, ex.values, side="left")
        steps = np.zeros(len(df) + 1)
        for p_, t in zip(pos, trades):
            steps[min(p_, len(df))] += t.pnl
        eq = float(risk.initial_balance) + np.cumsum(steps)[: len(df)]
    curve = pd.DataFrame({"timestamp": ts, "equity": eq})
    curve.attrs["sizing_halt"] = {"halted": False, "skipped": 0, "skip_ratio": 0.0, "last_trade_exit": trades[-1].exit_time if trades else None,
                                  "risk_value": risk.risk_value, "contract_size": risk.contract_size, "sizing_mode": risk.sizing_mode, "skip_reasons": {}}
    return trades, curve


def _trade_pnl_moments(trades) -> dict:
    """Mean / sample std / n of per-trade P&L, used by the holdout gate to put an
    upper confidence bound on expectancy (a thin holdout that merely lost money
    is noise, not proof of no edge)."""
    vals = [float(getattr(t, "pnl", 0.0) or 0.0) for t in trades or []]
    n = len(vals)
    if n == 0:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(vals) / n
    var = (sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": var ** 0.5}


def run_holdout_comparison(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    holdout_frac: float = 0.2,
    adaptive_risk=None,
    continuous_account: bool = True,
    min_holdout_trades: int = 30,
) -> dict:
    """
    continuous_account (default True): ONE run over the whole sample on one
    account, trades then assigned to in-sample / holdout by entry time. The
    holdout therefore starts from the account state the development period
    left behind and cannot be wiped out by a fresh account that sizes to
    zero (the old 'compares nothing' failure). The strategy sees only past
    data at each bar, so the holdout stays out-of-sample. False restores the
    legacy two independent runs. `holdout_ok` is False below
    `min_holdout_trades` trades -- a comparison on fewer is not evidence.

    Chronological in-sample / out-of-sample split, run ONCE.

    Splits df at a single time point (default: the last 20% of bars becomes
    the holdout), runs the identical strategy + risk config independently on
    each half, and returns both statistics sets side by side. Each half is
    passed to the strategy as its own standalone DataFrame -- higher-timeframe
    context the strategy derives internally (e.g. resampling to 1H/4H) is
    therefore also correctly confined to that half, with no leakage of
    future (holdout-period) price action into the in-sample run or vice
    versa.

    This is deliberately NOT a walk-forward optimizer and does not re-tune
    any parameters between segments -- it answers a narrower, falsification-
    style question: "does this exact strategy, as written, keep working on
    data it has never touched?" A strategy whose edge is real should degrade
    gracefully, not evaporate or invert, on the holdout segment.

    FIX (holdout-comparison-ignored-adaptive-risk): `adaptive_risk` used to
    be silently dropped here even when the CALLER's main backtest of this
    same strategy+risk used it -- despite this function's own promise to
    run "the identical strategy + risk config" on each half. A strategy
    that an adaptive-risk throttle had suppressed down to almost no trading
    in the main run could then show up here trading completely normally
    (no throttle applied), producing wildly inconsistent, confusing
    trade counts/statistics for what's supposed to be the same
    configuration. None (the default) is byte-identical to before this
    parameter existed."""
    n = len(df)
    split_idx = int(n * (1 - holdout_frac))
    split_idx = max(1, min(split_idx, n - 1)) if n > 1 else n

    if continuous_account and n > 1:
        from app.backtest.statistics import compute_statistics
        full = run_backtest(df, strategy, risk, adaptive_risk=adaptive_risk)
        split_ts = df["timestamp"].iloc[split_idx]
        ins_tr = [t for t in full.trades if t.entry_time < split_ts]
        hold_tr = [t for t in full.trades if t.entry_time >= split_ts]
        eq = full.equity_curve
        def _stats(trs, curve, start_bal):
            if not trs:
                return None
            c = curve.copy()
            return compute_statistics(trs, c, initial_balance=start_bal).to_dict()
        eq_ts = pd.to_datetime(eq["timestamp"]) if "timestamp" in eq.columns else None
        if eq_ts is not None:
            ins_curve = eq[eq_ts < pd.Timestamp(split_ts)]
            hold_curve = eq[eq_ts >= pd.Timestamp(split_ts)]
        else:
            ins_curve, hold_curve = eq.iloc[:split_idx], eq.iloc[split_idx:]
        hold_start = float(ins_curve["equity"].iloc[-1]) if len(ins_curve) else float(risk.initial_balance)
        return {
            "holdout_frac": holdout_frac,
            "mode": "continuous_account",
            "in_sample_period": (str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[split_idx - 1])),
            "holdout_period": (str(split_ts), str(df["timestamp"].iloc[-1])),
            "in_sample_bars": split_idx,
            "holdout_bars": n - split_idx,
            "in_sample_trades": len(ins_tr),
            "holdout_trades": len(hold_tr),
            "min_holdout_trades": min_holdout_trades,
            "holdout_ok": len(hold_tr) >= min_holdout_trades,
            "in_sample_statistics": _stats(ins_tr, ins_curve, float(risk.initial_balance)),
            "holdout_statistics": _stats(hold_tr, hold_curve, hold_start),
            "holdout_trade_pnl": _trade_pnl_moments(hold_tr),
        }

    in_sample_df = df.iloc[:split_idx].reset_index(drop=True)
    holdout_df = df.iloc[split_idx:].reset_index(drop=True)

    in_sample_result = run_backtest(in_sample_df, strategy, risk, adaptive_risk=adaptive_risk) if len(in_sample_df) else None
    holdout_result = run_backtest(holdout_df, strategy, risk, adaptive_risk=adaptive_risk) if len(holdout_df) else None

    return {
        "holdout_frac": holdout_frac,
        "in_sample_period": (
            (str(in_sample_df["timestamp"].iloc[0]), str(in_sample_df["timestamp"].iloc[-1]))
            if len(in_sample_df) else (None, None)
        ),
        "holdout_period": (
            (str(holdout_df["timestamp"].iloc[0]), str(holdout_df["timestamp"].iloc[-1]))
            if len(holdout_df) else (None, None)
        ),
        "in_sample_bars": len(in_sample_df),
        "holdout_bars": len(holdout_df),
        "in_sample_statistics": in_sample_result.statistics.to_dict() if in_sample_result else None,
        "holdout_statistics": holdout_result.statistics.to_dict() if holdout_result else None,
        "holdout_trade_pnl": _trade_pnl_moments(holdout_result.trades) if holdout_result else None,
    }


def run_backtest(
    df: pd.DataFrame,
    strategy: Strategy,
    risk: RiskConfig,
    adaptive_risk: AdaptiveRiskConfig | None = None,
    intrabar_df: pd.DataFrame | None = None,
    lookahead_check: str = "auto",
    compute_stats: bool = True,
) -> BacktestResult:
    """
    compute_stats: False skips compute_statistics and returns
        ``statistics=None`` (trades, equity curve and warnings are
        unchanged). Only for inner search loops that read nothing but the
        trade list / equity-curve attrs (the walk-forward GA scores every
        candidate that way); every caller that reads ``.statistics`` keeps
        the default True.
    intrabar_df: finer-timeframe (e.g. 1-minute) frame for fill-order replay
        (used only when risk.intrabar_replay is True).
    lookahead_check: "auto" (default) runs the behavioural lookahead check once
        per strategy STRUCTURE (numeric parameters ignored) and data window and
        reuses the verdict; "always" re-runs it every call (old behaviour);
        "skip" omits it (GA / Stage-1 / grid inner loops).

    df: standardized OHLCV DataFrame (see app.data.importer)
    strategy: any Strategy subclass instance (manual/python/pinescript/mql5)
    risk: RiskConfig describing sizing, costs, and execution assumptions
    adaptive_risk: optional declarative money-management overlay (see
        app.backtest.adaptive_risk) -- None/omitted runs exactly as before
        this parameter existed.
    """
    # FIX (MTF-STRATEGY-001): resample `df` to whatever timeframe(s)
    # `strategy` itself declares it needs (see app.data.timeframe_resample
    # for exactly how each source type declares this) BEFORE the strategy
    # ever sees the data -- this is the one chokepoint every tool in the
    # app funnels through (Run & Report, Full Pipeline, Quick Optimize,
    # Search Lab, Evolution Lab, Forge, Speed Run, CPCV, WFO/WFGA, Multi-
    # Objective, Ensemble, Portfolio, ...), so fixing it here fixes it
    # everywhere with no other caller needing to change. A strategy that
    # declares nothing gets `df` back completely unchanged -- byte-
    # identical to every run before this existed.
    from app.strategy.base import apply_strategy_trade_cap
    risk = apply_strategy_trade_cap(risk, strategy)   # MAX_TRADES_PER_DAY declared by the strategy (one-shot-per-day systems)
    # 2026-10 audit: behaviour rules the strategy declares (BEHAVIOR_* constants) -> engine-level overlays (see app.backtest.behavior_rules)
    from app.backtest.behavior_rules import apply_strategy_behavior
    risk, adaptive_risk = apply_strategy_behavior(risk, adaptive_risk, strategy)
    _raw_native_df = df
    from app.data.timeframe_resample import apply_timeframe_override
    apply_timeframe_override(strategy, risk)     # RiskConfig.exec_timeframe -> this strategy object
    df, timeframe_warnings = prepare_timeframe_aligned_data(df, strategy)

    # v9.5: intrabar replay auto-derivation. When the strategy runs on a
    # resampled (coarser) frame but the caller handed us finer native bars
    # and risk.intrabar_replay is on, the native frame IS the intrabar
    # frame: stop/target order inside each strategy bar resolves on the
    # real finer path instead of the coarse bar's high/low (stop-first).
    # No finer data -> nothing changes (execution ignores intrabar_df
    # unless it is strictly finer than the strategy bars).
    if (
        intrabar_df is None
        and getattr(risk, "intrabar_replay", False)
        and len(df) < len(_raw_native_df)
    ):
        intrabar_df = _raw_native_df

    strat_result: StrategyResult = strategy.generate(df)

    # 2026-10-19: macro (higher-timeframe) bias filter -- any strategy type, causal (closed HTF bars only).
    if getattr(risk, "macro_timeframe", None) and getattr(risk, "macro_mode", "off") not in ("off", "", None):
        from app.strategy.macro_filter import apply_macro_filter
        try:
            strat_result, _macro_note = apply_macro_filter(
                strat_result, df, risk.macro_timeframe, risk.macro_mode, int(getattr(risk, "macro_length", 50) or 50),
                bool(getattr(risk, "macro_strict", False)), bool(getattr(risk, "macro_exit_on_flip", False)))
            timeframe_warnings = list(timeframe_warnings) + [_macro_note]
        except ValueError as _macro_exc:
            timeframe_warnings = list(timeframe_warnings) + [f"Macro filter not applied: {_macro_exc}"]

    if getattr(risk, "regime_filter", "off") not in ("off", "", None):
        from app.strategy.macro_filter import apply_regime_filter
        try:
            strat_result, _reg_note = apply_regime_filter(strat_result, df, risk.regime_filter)
            timeframe_warnings = list(timeframe_warnings) + [_reg_note]
        except ValueError as _reg_exc:
            timeframe_warnings = list(timeframe_warnings) + [f"Regime filter not applied: {_reg_exc}"]

    # P2-4 (warmup bars): a strategy may declare WARMUP_BARS = N so its
    # first N signals are forced flat (0). Indicators are seeded on NaN
    # (or, worse, fillna'd to a garbage constant like RSI.fillna(50)) and
    # must not trade before their longest lookback has real data behind
    # it -- e.g. an EMA(200) on daily bars trades on 200 days of garbage
    # without this. 0 (default) = unchanged.
    warmup_bars = int(getattr(strategy, "WARMUP_BARS", 0) or 0)  # per-strategy; 0 = unchanged
    if warmup_bars > 0:
        arr = strat_result.signals.to_numpy(copy=True); arr[:warmup_bars] = 0
        strat_result.signals = pd.Series(arr, index=strat_result.signals.index)

    # P0-5 (lookahead gate): run the behavioral lookahead check on every
    # backtest, not just Full Pipeline's gated paths. A crash of the check must not
    # break the backtest itself, but it must not read as a PASS either: it is recorded
    # (BacktestResult.lookahead_check_failed) and surfaced as a warning.
    _lookahead_result = None
    _lookahead_error = None
    if lookahead_check != "skip":
        try:
            from app.strategy.lookahead_check import check_for_lookahead
            _key = _lookahead_cache_key(strategy, df) if lookahead_check == "auto" else None
            if _key is not None and _key in _LOOKAHEAD_CACHE:
                _lookahead_result = _LOOKAHEAD_CACHE[_key]
            else:
                _lookahead_result = _lookahead_disk_load(strategy, df) if _key is not None else None
                if _lookahead_result is None:
                    _lookahead_result = check_for_lookahead(strategy, df)
                    if _key is not None:
                        _lookahead_disk_store(strategy, df, _lookahead_result)
                if _key is not None:
                    if len(_LOOKAHEAD_CACHE) >= 512:
                        _LOOKAHEAD_CACHE.pop(next(iter(_LOOKAHEAD_CACHE)))
                    _LOOKAHEAD_CACHE[_key] = _lookahead_result
        except Exception as _exc:  # noqa: BLE001
            _lookahead_result = None
            _lookahead_error = f"{type(_exc).__name__}: {_exc}"

    # UPGRADE (day-of-week trading restriction): forces the strategy's own
    # signal flat (0) on any weekday it has declared excluded (Manual's
    # config["filters"]["days_of_week"]["exclude"], Python's
    # EXCLUDE_DAYS_OF_WEEK, or PineScript/MQL5's `// T58_EXCLUDE_DAYS=`
    # directive -- see app.strategy.base.resolve_excluded_days_of_week for
    # the full convention). A strategy that declares nothing here gets its
    # signals back completely unchanged. Applied here, in the one
    # chokepoint every tool in the app funnels through, so it works
    # identically regardless of which tool ran the backtest.
    strat_result.signals = apply_days_of_week_exclusion(df, strat_result.signals, strategy)

    # UPGRADE (regime-conditional-trading primitive): same chokepoint,
    # same "declares nothing -> completely unchanged" convention as the
    # day-of-week exclusion just above -- see app.strategy.base.
    # apply_regime_exclusion/resolve_excluded_regimes for what a strategy
    # can declare and how app.validation.regime_matrix's own regime
    # detection is reused to enforce it.
    strat_result.signals = apply_regime_exclusion(df, strat_result.signals, strategy)

    condition_warnings: list[str] = []
    if getattr(strategy, "source_type", None) == "manual" and isinstance(getattr(strategy, "config", None), dict):
        # Catches a class of bug the GA-gene-bounds fix in
        # app.optimize.parameter_space now prevents going forward, but a
        # hand-typed, imported, or already-saved config can still contain:
        # a comparison against a bounded oscillator (RSI, Stochastic, ...)
        # whose threshold sits outside that oscillator's possible range,
        # silently disabling that branch of the strategy's logic. See
        # app.strategy.manual.validate_bounded_conditions for the full
        # explanation.
        from app.strategy.manual import validate_bounded_conditions
        condition_warnings = validate_bounded_conditions(strategy.config)

    with _warnings_module.catch_warnings(record=True) as caught:
        _warnings_module.simplefilter("always", RuntimeWarning)
        # P0-5: emit the lookahead verdict as a RuntimeWarning INSIDE the
        # catch_warnings block so engine.py's existing funnel carries it
        # into BacktestResult.warnings (the report's snippet calls
        # warnings.warn directly; this placement is what makes that
        # funnel actually catch it).
        if _lookahead_result is not None and _lookahead_result.bug_detected:
            _warnings_module.warn(
                "LOOKAHEAD BIAS DETECTED: " + _lookahead_result.summary(),
                RuntimeWarning,
            )
        if _lookahead_error is not None:
            _warnings_module.warn(
                "LOOKAHEAD CHECK CRASHED: " + _lookahead_error + " -- the strategy was NOT cleared of using future data; "
                "treat any pass as unverified.",
                RuntimeWarning,
            )
        if getattr(strat_result, "entry_orders", None) is not None:
            trades, equity_curve = _run_resting_orders(df, strat_result, strategy, risk)
        else:
          trades, equity_curve = run_execution(
            df=df,
            signals=strat_result.signals,
            risk=risk,
            stop_loss_pips=strat_result.stop_loss_pips,
            take_profit_pips=strat_result.take_profit_pips,
            stop_loss_distance=strat_result.stop_loss_distance,
            take_profit_distance=strat_result.take_profit_distance,
            trailing_stop_distance=strat_result.trailing_stop_distance,
            breakeven_trigger_r=strat_result.breakeven_trigger_r,
            partial_exit_config=strat_result.partial_exit,
            adaptive_risk=adaptive_risk,
            intrabar_df=intrabar_df,
        )
    execution_warnings = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]

    if not compute_stats:
        return BacktestResult(
            strategy_name=strat_result.name,
            trades=trades,
            equity_curve=equity_curve,
            statistics=None,
            initial_balance=risk.initial_balance,
            warnings=timeframe_warnings + condition_warnings + execution_warnings,
            lookahead_check_failed=_lookahead_error is not None,
        )

    stats = compute_statistics(trades, equity_curve, initial_balance=risk.initial_balance)

    # UPGRADE (buried-position-sizing-deviation): computed from `stats`,
    # which only exists AFTER the catch_warnings block above closes --
    # unlike condition_warnings/timeframe_warnings, this can't be raised
    # as a RuntimeWarning inside run_execution itself, so it's appended
    # here instead. See app.backtest.risk.position_sizing_deviation_
    # message for what triggers it; None (the common case) adds nothing.
    sizing_warning = position_sizing_deviation_message(stats.to_dict())

    return BacktestResult(
        strategy_name=strat_result.name,
        trades=trades,
        equity_curve=equity_curve,
        statistics=stats,
        initial_balance=risk.initial_balance,
        warnings=timeframe_warnings + condition_warnings + execution_warnings + ([sizing_warning] if sizing_warning else []),
        lookahead_check_failed=_lookahead_error is not None,
    )
