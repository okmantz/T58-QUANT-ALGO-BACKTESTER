"""Paper-mode forward harness: the SAME engine code as the backtest, fed incrementally (report section 6, "parity").

``PaperSession`` keeps the accumulated bar history. ``step(new_bars)`` appends the bars, re-runs the frozen strategy through
app.backtest.engine.run_backtest and returns only the trades that CLOSED after the last call, recording them in the fill ledger
at the engine's own fill prices. Because backtest and paper share one code path, parity is exact by construction; ``parity_report``
verifies it anyway against a full-history backtest of the same data (and, once a broker fills file exists, against real fills).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from app.forward_test import fill_ledger
from app.forward_test.frozen import freeze, verify
from app.forward_test.sprt_monitor import SprtMonitor


@dataclass
class PaperSession:
    strategy_config: dict
    risk: object
    symbol: str
    monitor: SprtMonitor
    history: pd.DataFrame = field(default_factory=pd.DataFrame)
    last_exit: pd.Timestamp | None = None
    frozen: dict = field(default_factory=dict)
    ledger_path: object = None

    def __post_init__(self):
        if not self.frozen:
            self.frozen = freeze(self.strategy_config, self.risk)

    def step(self, new_bars: pd.DataFrame) -> dict:
        ok, problems = verify(self.frozen, self.strategy_config, self.risk)
        if not ok:
            return {"error": "FROZEN-CODE VIOLATION: " + "; ".join(problems), "new_trades": 0, **self.monitor.status()}
        from app.backtest.engine import run_backtest
        from app.optimize.two_stage import trade_r_and_days
        from app.strategy.manual import ManualStrategy
        self.history = (pd.concat([self.history, new_bars], ignore_index=True)
                        .drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True))
        bt = run_backtest(self.history, ManualStrategy(self.strategy_config), self.risk, lookahead_check="skip")
        fresh = [t for t in bt.trades if self.last_exit is None or pd.Timestamp(t.exit_time) > self.last_exit]
        rs, days = trade_r_and_days(fresh)
        for t in fresh:
            fill_ledger.record_fill(self.symbol, t.direction, t.entry_price, t.entry_price, t.size, t.commission, kind="entry", path=self.ledger_path)
            fill_ledger.record_fill(self.symbol, -t.direction, t.exit_price, t.exit_price, t.size, 0.0, kind="exit", path=self.ledger_path)
        for r, d in zip(rs, days):
            self.monitor.add(r, pd.Timestamp(d).normalize() if d is not None else None)
        if fresh:
            self.last_exit = max(pd.Timestamp(t.exit_time) for t in fresh)
        return {"new_trades": len(fresh), **self.monitor.status()}


def parity_report(backtest_trades: list, paper_trades: list, price_tol: float = 1e-9) -> dict:
    """Match trades by (entry_time, direction) and compare entry/exit prices. 100% match is expected for same-code paper mode."""
    key = lambda t: (pd.Timestamp(t.entry_time), int(t.direction))  # noqa: E731
    a, b = {key(t): t for t in backtest_trades}, {key(t): t for t in paper_trades}
    common = set(a) & set(b)
    bad = [k for k in common if abs(a[k].entry_price - b[k].entry_price) > price_tol or abs(a[k].exit_price - b[k].exit_price) > price_tol]
    n = max(len(a), len(b), 1)
    return {"backtest_trades": len(a), "paper_trades": len(b), "matched": len(common) - len(bad), "price_mismatches": len(bad),
            "only_in_backtest": len(set(a) - set(b)), "only_in_paper": len(set(b) - set(a)), "parity_pct": 100.0 * (len(common) - len(bad)) / n}
