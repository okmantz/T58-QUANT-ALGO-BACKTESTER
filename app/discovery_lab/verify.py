"""Real-engine proof for Discovery Lab finalists.

The Lab's own simulator is fast but simplified. Before a candidate is called a winner it is (1) written out EXACTLY as it will be saved to the
Strategy Library, (2) loaded by the real backtester and run on the full history, and (3) put through the same behavioural lookahead test Full
Pipeline uses (truncate the data, re-run, diff the signals). A strategy that peeks at the future, or that the real engine does not reproduce as
profitable, is rejected here -- not discovered later as a failed pipeline run.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .codegen import render


def _profit_factor(pnls: list) -> float:
    pos = sum(p for p in pnls if p > 0)
    neg = -sum(p for p in pnls if p < 0)
    return pos / neg if neg > 0 else (9.99 if pos > 0 else 0.0)


def _pipeline_ready(strat, d: pd.DataFrame, cut, pipeline_risk, prop_rules, symbol: str) -> dict:
    """Would Full Pipeline's baseline preflight accept this strategy? Mirrors its sizing resolution exactly (shared function)."""
    from dataclasses import replace

    from app.backtest.engine import run_backtest
    from app.validation.preflight import resolve_baseline_sizing

    try:
        ts = pd.to_datetime(d["timestamp"])
        dev = d[(ts < cut).to_numpy()] if cut is not None else d
        if len(dev) < 500:
            dev = d
        risk = pipeline_risk
        if prop_rules is not None:
            try:
                from app.backtest.risk import with_prop_safety_defaults
                risk = with_prop_safety_defaults(risk, prop_rules)
            except Exception:  # noqa: BLE001
                pass
        risk = replace(risk, sizing_mode="rr_planned")
        _s, _r, bt, pf, notes = resolve_baseline_sizing(dev, strat, risk, prop_rules, run_backtest_fn=run_backtest, symbol=symbol)
        if pf is None:
            return {"ok": False, "reason": "Full Pipeline's preflight could not size a single trade of this strategy"}
        if pf.blocked:
            msgs = "; ".join(i.message.split(".")[0] for i in pf.by_severity("block"))[:220]
            return {"ok": False, "reason": f"Full Pipeline would stop at preflight: {msgs}"}
        adj = notes.get("sizing_adjustment")
        return {"ok": True, "remedy": (adj or {}).get("text") if isinstance(adj, dict) else None}
    except Exception as exc:  # noqa: BLE001 -- inability to prove readiness is not readiness
        return {"ok": False, "reason": f"could not confirm Full Pipeline readiness ({type(exc).__name__}: {exc})"}


def make_verifier(raw_df: pd.DataFrame, frames: dict, risk, instrument: str, sealed_frac: float = 0.2, log=lambda m: None,
                  pipeline_risk=None, prop_rules=None):
    """Returns ``verify(candidate, minutes) -> dict``.

    ``raw_df``: the data as the user loaded it (what Full Pipeline will be handed). ``frames``: {minutes: resampled frame} used by the Lab.
    The lookahead test runs on a recent slice of ``raw_df`` (a few hundred thousand rows -- the signal logic is the same everywhere, and the
    test re-runs the strategy many times); the performance proof runs on the Lab's full-history frame at the strategy's bar size."""
    from app.backtest.engine import run_backtest
    from app.strategy.python import PythonStrategy

    tmp = Path(tempfile.mkdtemp(prefix="t58_lab_verify_"))
    n_raw = len(raw_df)
    tail = raw_df.iloc[max(0, n_raw - 360_000):].reset_index(drop=True)

    def verify(c: dict, minutes: int) -> dict:
        name = f"lab_proof_{abs(hash(c.get('id', id(c)))) % 10**8}"
        path = tmp / f"{name}.py"
        path.write_text(render(c, instrument, minutes, name), encoding="utf-8")
        strat = PythonStrategy(path)
        out: dict = {"ok": False}
        # Order (2026-10-30, speed): the CHEAP rejections come first. A would-be winner is a single backtest on the Lab's frame; most fail
        # there (no reproduced profit) or at Full Pipeline's sizing gate, and only survivors pay for the (re-running) lookahead probes.
        d = frames.get(int(minutes))
        if d is None:
            out["reason"] = "no full-history frame for this bar size"
            return out
        # (1) the real engine on the full-history frame at this bar size
        bt = run_backtest(d, strat, risk, lookahead_check="skip")
        trades = bt.trades
        out["trades"] = len(trades)
        if len(trades) < 25:
            out["reason"] = f"the real engine only produced {len(trades)} trades"
            return out
        pnl = [float(t.pnl) for t in trades]
        st = bt.statistics
        out["net_profit"] = round(float(sum(pnl)), 2)
        out["profit_factor"] = round(_profit_factor(pnl), 2)
        out["max_drawdown_pct"] = round(float(getattr(st, "max_drawdown_pct", 0.0) or 0.0), 2)
        ts = pd.to_datetime(d["timestamp"])
        cut = ts.iloc[int(len(d) * (1 - sealed_frac))]
        late = [float(t.pnl) for t in trades if pd.Timestamp(t.entry_time).tz_localize(None) >= cut.tz_localize(None)] if hasattr(cut, "tz_localize") else []
        out["sealed_trades"] = len(late)
        out["sealed_net_profit"] = round(float(sum(late)), 2)
        if out["net_profit"] <= 0 or out["profit_factor"] < 1.05:
            out["reason"] = (f"the real engine does not reproduce a profit (net ${out['net_profit']:,.0f}, profit factor {out['profit_factor']}) "
                             "once its own fills, sizing and costs are applied")
            return out
        if late and out["sealed_net_profit"] <= 0:
            out["reason"] = "the real engine loses money on the sealed (most recent) slice"
            return out
        # (2) Full Pipeline readiness (cheap, and the most common reason a winner was unusable)
        if pipeline_risk is not None:
            ready = _pipeline_ready(strat, d, cut, pipeline_risk, prop_rules, instrument)
            out["pipeline_ready"] = ready["ok"]
            if not ready["ok"]:
                out["reason"] = ready["reason"]
                return out
            if ready.get("remedy"):
                out["pipeline_remedy"] = ready["remedy"]
        # (3) lookahead, exactly as the pipeline does it: run_backtest resamples to the strategy's TIMEFRAME and then runs the behavioural
        #     lookahead test on that frame, reporting a verdict as a "LOOKAHEAD BIAS DETECTED" warning. The recent slice AND an early slice of
        #     the history are probed (a leak that only shows in older data -- a different session layout, a gap, a contract roll -- used to pass
        #     here and fail later in Full Pipeline); a bounded early slice keeps this cheap instead of re-running the whole history.
        probe = run_backtest(tail, strat, risk, lookahead_check="always")
        bad = [w for w in probe.warnings if "LOOKAHEAD BIAS DETECTED" in str(w)]
        crashed = [w for w in probe.warnings if "LOOKAHEAD CHECK CRASHED" in str(w)]
        out["lookahead_checked"] = not crashed
        out["lookahead_bug"] = bool(bad)
        if bad:
            out["reason"] = "failed the lookahead test (its signal changes when future bars are removed)"
            return out
        if crashed:
            out["reason"] = "the lookahead test could not run: " + str(crashed[0])[:160]
            return out
        early = d.iloc[: max(2_000, min(len(d) // 3, 60_000))].reset_index(drop=True)
        if len(early) > 2_000 and len(d) > len(tail) // max(1, int(minutes)) + 500:
            early_probe = run_backtest(early, strat, risk, lookahead_check="always")
            if [w for w in early_probe.warnings if "LOOKAHEAD BIAS DETECTED" in str(w)]:
                out["lookahead_bug"] = True
                out["reason"] = "failed the lookahead test on the early history (its signal changes when future bars are removed)"
                return out
        out["ok"] = True
        return out

    def cleanup():
        shutil.rmtree(tmp, ignore_errors=True)

    verify.cleanup = cleanup          # type: ignore[attr-defined]
    return verify
