"""Rescue search: give a weak strategy a fair chance by changing its STRUCTURE, not just its numbers.

A parameter GA can only move the numbers a strategy already has. Many weak strategies fail for structural reasons: they trade a side that
has no edge, hold too long, give back profit without a trail, or trade hours when the idea does not work. This module tries those
structural changes one stage at a time (greedy, so the number of tries stays small and countable):

    direction -> stop/target -> exit style (opposite-signal, time stop, trailing, break-even) -> session window

Honesty rules, because a rescue search is a big multiple-testing machine:
  * every variant is scored on the TRAIN part of the data only (``edge_lb_r``: lower bound of mean R per trade);
  * the winner is then judged on the held-out last part, which no variant ever saw, and the result says so;
  * every variant tried is added to the trial ledger, so the Deflated Sharpe check later corrects for all of them.
A rescue that wins on train and not on the holdout is reported as "no rescue found", not as a win.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Iterable

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy

MIN_TRADES = 30


def _cond(left: dict, op: str, right: dict) -> dict:
    return {"left": left, "operator": op, "right": right}


SESSION_WINDOWS = (("Overnight", "00:00", "07:59"), ("Morning", "08:00", "12:59"), ("Afternoon", "13:00", "16:59"), ("Evening", "17:00", "23:59"))
STOPS = (1.0, 1.5, 2.0, 3.0)
TARGETS = (1.5, 2.0, 3.0, 4.0)


def _rm(cfg: dict) -> dict:
    return cfg.setdefault("risk_management", {})


def direction_variants(cfg: dict) -> list[tuple[str, dict]]:
    out = []
    ec = cfg.get("entry_conditions", {}) or {}
    has_long, has_short = bool(ec.get("long")), bool(ec.get("short"))
    if has_long and has_short:
        for side, label in (("short", "long only"), ("long", "short only")):
            v = copy.deepcopy(cfg)
            v["entry_conditions"][side] = []
            v["entry_conditions"][side + "_connectors"] = []
            out.append((label, v))
    return out


def stop_target_variants(cfg: dict) -> list[tuple[str, dict]]:
    out = []
    for s in STOPS:
        for t in TARGETS:
            if t < 0.75 * s:
                continue
            v = copy.deepcopy(cfg)
            rm = _rm(v)
            rm.update({"stop_type": "atr", "stop_value": s, "stop_atr_period": rm.get("stop_atr_period", 14),
                       "target_type": "atr", "target_value": t, "target_atr_period": rm.get("target_atr_period", 14)})
            out.append((f"stop {s:g} ATR, target {t:g} ATR", v))
    return out


def exit_style_variants(cfg: dict) -> list[tuple[str, dict]]:
    out = []
    rm0 = _rm(copy.deepcopy(cfg))
    v = copy.deepcopy(cfg)
    _rm(v)["opposite_signal_exit"] = not bool(rm0.get("opposite_signal_exit", True))
    out.append(("opposite-signal exit " + ("on" if _rm(v)["opposite_signal_exit"] else "off"), v))
    for bars in (12, 48):
        v = copy.deepcopy(cfg)
        _rm(v)["max_bars_in_trade"] = bars
        out.append((f"time stop {bars} bars", v))
    v = copy.deepcopy(cfg)
    _rm(v)["trailing_stop"] = {"enabled": True, "value": 2.0, "atr_period": 14}
    out.append(("trailing stop 2 ATR", v))
    v = copy.deepcopy(cfg)
    _rm(v)["break_even"] = {"enabled": True, "trigger_r": 1.0}
    out.append(("break-even at 1R", v))
    return out


def session_variants(cfg: dict) -> list[tuple[str, dict]]:
    out = []
    for name, a, b in SESSION_WINDOWS:
        v = copy.deepcopy(cfg)
        ec = v.setdefault("entry_conditions", {})
        gate = _cond({"type": "time_of_day", "session_start": a, "session_end": b}, "is true", {"type": "value", "value": 1})
        for side in ("long", "short"):
            if ec.get(side):
                ec[side] = list(ec[side]) + [copy.deepcopy(gate)]
                ec[side + "_connectors"] = list(ec.get(side + "_connectors") or ["AND"] * (len(ec[side]) - 2)) + ["AND"]
        out.append((f"only {name.lower()} ({a}-{b})", v))
    return out


def _distinct(make: Callable[[dict], list[tuple[str, dict]]]) -> Callable[[dict], list[tuple[str, dict]]]:
    """Drop variants identical to the strategy they start from, so no try is wasted or counted twice."""
    def wrapped(cfg: dict) -> list[tuple[str, dict]]:
        return [(label, v) for label, v in make(cfg) if v != cfg]
    wrapped.__name__ = make.__name__
    return wrapped


STAGES: tuple[tuple[str, Callable[[dict], list[tuple[str, dict]]]], ...] = (
    ("Which side to trade", _distinct(direction_variants)),
    ("Stop and target", _distinct(stop_target_variants)),
    ("How trades end", _distinct(exit_style_variants)),
    ("Trading hours", _distinct(session_variants)),
)


def score_config(df, cfg: dict, risk: RiskConfig) -> tuple[float, int]:
    """(edge_lb_r, trade count) of one configuration on ``df``; (-inf, n) if it cannot be measured."""
    from app.discovery_engine.objective import edge_lb_r
    try:
        res = run_backtest(df, ManualStrategy(copy.deepcopy(cfg)), risk, lookahead_check="skip", compute_stats=False)
    except Exception:  # noqa: BLE001 -- a variant that cannot run simply loses
        return float("-inf"), 0
    n = len(res.trades)
    if n < MIN_TRADES:
        return float("-inf"), n
    return float(edge_lb_r(res.trades, min_trades=MIN_TRADES)), n


def _r(x: float) -> str:
    return "too few trades" if x is None or x != x or x in (float("inf"), float("-inf")) else f"{x:+.2f}R"


@dataclass
class RescueResult:
    base_cfg: dict
    best_cfg: dict
    base_train: float
    best_train: float
    base_holdout: float
    best_holdout: float
    trials: int
    steps: list = field(default_factory=list)      # [{"stage","label","train","kept"}]
    changes: list = field(default_factory=list)    # labels of the changes that were kept, in order

    @property
    def rescued(self) -> bool:
        """True only when the winner beats the original on data no variant was scored on, and its lower bound is above zero."""
        return bool(self.changes) and self.best_holdout > self.base_holdout and self.best_holdout > 0.0

    def headline(self) -> str:
        if not self.changes:
            return f"No structural change beat the original on the {self.trials} variants tried. The idea itself may be the problem."
        what = "; ".join(self.changes)
        if self.rescued:
            return (f"Rescued on unseen data: {what}. Holdout edge {_r(self.base_holdout)} to {_r(self.best_holdout)} "
                    f"(lower bound per trade, after costs). {self.trials} variants were tried and counted as trials.")
        return (f"Looked better on the training data ({what}) but did not hold up on the holdout "
                f"({_r(self.base_holdout)} to {_r(self.best_holdout)}). Treat it as a coincidence, not a rescue. {self.trials} variants tried.")


def rescue(df, cfg: dict, risk: RiskConfig, *, holdout_frac: float = 0.3, min_gain: float = 0.01,
           stages: Iterable = STAGES, progress: Callable[[str], None] | None = None,
           ledger_label: str | None = "rescue") -> RescueResult:
    """Greedy structural search. ``df`` is split in time: the first ``1 - holdout_frac`` selects, the rest only judges."""
    say = progress or (lambda _m: None)
    cut = int(len(df) * (1.0 - holdout_frac))
    train, hold = df.iloc[:cut], df.iloc[cut:]
    best = copy.deepcopy(cfg)
    base_train, _ = score_config(train, best, risk)
    cur = base_train
    trials, steps, changes = 1, [], []
    for stage, make in stages:
        say(f"{stage}: trying variants...")
        stage_best, stage_label, stage_cfg = cur, None, None
        for label, v in make(best):
            sc, _n = score_config(train, v, risk)
            trials += 1
            steps.append({"stage": stage, "label": label, "train": None if sc == float("-inf") else round(sc, 4), "kept": False})
            if sc > stage_best + min_gain or (stage_best == float("-inf") and sc > float("-inf")):
                stage_best, stage_label, stage_cfg = sc, label, v
        if stage_cfg is not None:
            best, cur = stage_cfg, stage_best
            changes.append(stage_label)
            for s in steps:
                if s["stage"] == stage and s["label"] == stage_label:
                    s["kept"] = True
            say(f"{stage}: kept '{stage_label}' ({cur:+.3f}R on training data)")
    base_h, _ = score_config(hold, cfg, risk)
    best_h, _ = score_config(hold, best, risk) if changes else (base_h, 0)
    if ledger_label:
        try:
            from app.validation import trial_ledger
            trial_ledger.record(ledger_label, trials)
        except trial_ledger.TrialLedgerError:
            raise    # 2026-10-31 audit: trials that could not be recorded must stop the run
        except Exception:  # noqa: BLE001 -- bookkeeping must never break a search
            pass
    return RescueResult(copy.deepcopy(cfg), best, base_train, cur, base_h, best_h, trials, steps, changes)
