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


def make_verifier(raw_df: pd.DataFrame, frames: dict, risk, instrument: str, sealed_frac: float = 0.2, log=lambda m: None):
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
        # (1) lookahead, exactly as the pipeline does it: run_backtest resamples to the strategy's TIMEFRAME and then runs the behavioural
        #     lookahead test on that frame, reporting a verdict as a "LOOKAHEAD BIAS DETECTED" warning.
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
        # (2) the real engine on the full-history frame at this bar size
        d = frames.get(int(minutes))
        if d is None:
            out["reason"] = "no full-history frame for this bar size"
            return out
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
        out["ok"] = True
        return out

    def cleanup():
        shutil.rmtree(tmp, ignore_errors=True)

    verify.cleanup = cleanup          # type: ignore[attr-defined]
    return verify
