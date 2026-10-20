"""Sequential forward-test monitor (report 2026-10-20, sections 4 and 6): SPRT + kill switch + ladder state.

H0 = 0R per trade, H1 = the shrunk lower-bound edge the backtest earned (NOT its in-sample mean), 5% false-accept, 20%
false-reject, futility truncation. State is persisted so a forward test can run for months across restarts. The monitor consumes
closed-trade R values and returns one of CONTINUE / ACCEPT_H1 / ACCEPT_H0 / STOP_FUTILE plus the ladder state to hand to
app.orchestration.verdict_ladder.ladder_stage(forward_state=...). The kill switch is independent of the SPRT: it fires on
rule-style risk breaches (daily loss, consecutive losses, drawdown in R) so a bad streak halts trading even while the SPRT is
still inconclusive.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.validation.edge_stats import DEFAULT_TRADE_SD_R, sprt_boundaries, sprt_decision


@dataclass
class KillSwitch:
    max_daily_loss_r: float = 3.0
    max_consecutive_losses: int = 12
    max_drawdown_r: float = 15.0

    def check(self, r_values: list, day_of_each: list | None = None) -> list:
        reasons = []
        streak = worst_streak = 0
        eq = peak = dd = 0.0
        for x in r_values:
            streak = streak + 1 if x < 0 else 0
            worst_streak = max(worst_streak, streak)
            eq += x
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        if worst_streak >= self.max_consecutive_losses:
            reasons.append(f"{worst_streak} consecutive losses (limit {self.max_consecutive_losses})")
        if dd >= self.max_drawdown_r:
            reasons.append(f"drawdown {dd:.1f}R (limit {self.max_drawdown_r:.1f}R)")
        if day_of_each:
            by: dict = {}
            for x, d in zip(r_values, day_of_each):
                by[d] = by.get(d, 0.0) + x
            worst = min(by.values()) if by else 0.0
            if -worst >= self.max_daily_loss_r:
                reasons.append(f"single-day loss {-worst:.1f}R (limit {self.max_daily_loss_r:.1f}R)")
        return reasons


@dataclass
class SprtMonitor:
    h1_r: float
    sd_r: float = DEFAULT_TRADE_SD_R
    alpha: float = 0.05
    beta: float = 0.20
    max_trades: int | None = None
    kill: KillSwitch = field(default_factory=KillSwitch)
    r_values: list = field(default_factory=list)
    days: list = field(default_factory=list)
    frozen_hash: str | None = None

    def __post_init__(self):
        b = sprt_boundaries(self.h1_r, self.sd_r, 0.0, self.alpha, self.beta)
        if self.max_trades is None:
            self.max_trades = int(round(3 * b.expected_trades_h1))  # truncate at ~3x the expected stopping point under H1

    def add(self, r: float, day=None) -> dict:
        self.r_values.append(float(r))
        self.days.append(day)
        return self.status()

    def status(self) -> dict:
        d = sprt_decision(self.r_values, self.h1_r, self.sd_r, 0.0, self.alpha, self.beta, self.max_trades)
        halted = self.kill.check(self.r_values, self.days if all(x is not None for x in self.days) else None)
        d["kill_switch"] = halted
        d["halted"] = bool(halted)
        d["ladder_state"] = {"ACCEPT_H1": "EVALUATE"}.get(d["decision"], "FORWARD-TEST CANDIDATE" if d["decision"] == "CONTINUE" else "REJECT")
        if d["halted"]:
            d["ladder_state"] = "REJECT"
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({**self.__dict__, "kill": self.kill.__dict__}, default=str), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SprtMonitor":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        kill = KillSwitch(**d.pop("kill"))
        return cls(kill=kill, **d)
