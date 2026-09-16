"""
Bankroll / EV survival -- turning a resampling distribution into the
actual question a trader with a finite amount of money has: "if I put $X
in, what's my probability of a good return?"

Three pieces this app already has, each independently validated, answer
different halves of that question on their own but not the whole thing:

  - app.monte_carlo.engine.run_monte_carlo(reset_on_breach=True) shows
    how many mechanical rebuys a chain typically needs, but assumes an
    infinite bankroll -- it never asks whether the trader could actually
    AFFORD to keep rebuying.
  - app.prop.survival_engine.simulate_reset_chain already prices resets
    with real fees and a real profit split, but draws a FRESH independent
    resample for every attempt in the chain -- a bad stretch of trades in
    the real history never stays clustered together across attempts the
    way it would in one continuous, plausible future.
  - Neither ties bankroll depletion into the chain itself: a life is
    modeled as running for a fixed number of attempts, not as stopping
    the instant the trader can no longer afford the next one.

This module closes that gap: for each of n_simulations simulated
"lives", it draws ONE resampled trade ordering (block bootstrap by
default -- see BankrollConfig.method), runs the mechanical reset-on-
breach chain ONCE against that single ordering (so a real bad stretch
stays a real bad stretch across every attempt drawn from it), and then
walks that life's attempt-by-attempt outcome applying real economics --
subtracting each attempt's fee, crediting each payout's trader's-cut --
stopping the life the moment its bankroll can no longer cover the next
attempt (ruin) or (optionally) the moment it reaches a first payout.
Every dollar amount here still comes from app.prop.simulator.
simulate_account(); this module only sequences its per-attempt outputs
against a finite starting bankroll.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.backtest.execution import Trade
from app.monte_carlo.engine import _apply_slippage_stress, _resample_pnls
from app.prop.simulator import PropRules, precompute_day_structure, simulate_account
from app.prop.survival_engine import ResetEconomics


@dataclass
class _ResampleCfg:
    """Minimal stand-in so this module doesn't need to import
    MonteCarloConfig just to satisfy _resample_pnls' type hint -- it only
    ever reads .method and .block_size off whatever's passed in. Mirrors
    the identical small helper in app.prop.survival_engine, kept local
    here rather than imported so this module has no dependency on that
    module's private names."""
    method: str
    block_size: int


@dataclass
class BankrollConfig:
    starting_bankroll: float = 1_000.0
    reset_economics: ResetEconomics = field(default_factory=ResetEconomics)
    n_simulations: int = 5_000
    # Block bootstrap is the recommended default here specifically (unlike
    # MonteCarloConfig's plain-Monte-Carlo default of i.i.d. "bootstrap"):
    # a bankroll question is exactly the case where destroying real
    # streakiness/regime-clustering matters most -- a bad week smeared
    # randomly across the simulated chain understates how often a trader
    # actually runs out of money during one continuous rough patch.
    method: str = "block_bootstrap"          # "shuffle" | "bootstrap" | "block_bootstrap"
    block_size: int = 5
    slippage_stress_pct: float = 0.0
    random_seed: int | None = 42
    # If True, a life's chain stops the instant it reaches its first
    # payout (the trader "cashed out" of this analysis rather than being
    # modeled as continuing to risk that bankroll on further attempts).
    # If False (default), the chain keeps going -- still bounded by
    # reset_economics.max_attempts and by running out of bankroll --
    # which is the right default for "what's my realistic lifetime return
    # on this bankroll," and stop_after_first_payout=True is the right
    # setting for "what's my probability of getting paid AT ALL before I
    # go broke," a narrower and usually higher-probability question.
    stop_after_first_payout: bool = False

    def __post_init__(self):
        self.starting_bankroll = max(float(self.starting_bankroll), 0.0)
        self.n_simulations = max(int(self.n_simulations), 1)


@dataclass
class BankrollSurvivalResult:
    n_simulations: int
    starting_bankroll: float

    probability_reach_first_payout: float           # % of lives that got at least one payout before ruin/attempt-cap
    probability_ruin_before_first_payout: float      # % of lives that ran out of bankroll before ever getting paid
    probability_exhausted_attempts_without_payout: float  # hit reset_economics.max_attempts (or the trade data ran out) still solvent, but never got paid

    expected_net_profit: float
    median_net_profit: float
    probability_net_positive: float

    expected_attempts_used: float
    median_attempts_used: float
    expected_fees_paid: float
    expected_gross_payouts: float

    median_bankroll_low_point: float                 # median of each life's lowest bankroll point reached
    worst_bankroll_low_point: float                  # most negative-going low point across all lives (>=0 if nobody would have gone negative)

    net_profit_distribution: list = field(default_factory=list)
    attempts_used_distribution: list = field(default_factory=list)

    max_attempts_configured: int = 0
    profit_split_pct: float = 0.0
    methodology_note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _methodology_note(cfg: BankrollConfig, n_trades: int) -> str:
    method_descriptions = {
        "bootstrap": (
            f"i.i.d. bootstrap ({n_trades} historical trades treated as independent -- real "
            "streakiness in the strategy's actual results is not preserved, so bankroll-ruin "
            "risk here is likely UNDERSTATED)"
        ),
        "shuffle": (
            f"shuffling the order of the same {n_trades} historical trades (no repeats)"
        ),
        "block_bootstrap": (
            f"block bootstrap (block size {cfg.block_size}, preserving short local runs of the "
            f"{n_trades} historical trades so a real bad stretch stays clustered together "
            "across the chain's attempts, the way it would in one continuous plausible future)"
        ),
    }
    method_text = method_descriptions.get(cfg.method, f"resampling (method: {cfg.method})")
    stop_text = (
        "each life's chain stopped at its first payout (this measures 'probability of getting "
        "paid at all before going broke', not lifetime return)"
        if cfg.stop_after_first_payout else
        "each life's chain kept going after a payout, up to the configured max attempts or "
        "until the bankroll or the trade data ran out (this measures realistic lifetime return "
        "on the starting bankroll, not just 'did I ever get paid')"
    )
    return (
        f"Simulated {cfg.n_simulations:,} independent 'lives', each one ONE resampled trade "
        f"ordering ({method_text}) run through the mechanical reset-on-breach chain "
        "(app.prop.simulator.simulate_account) exactly once, then walked attempt-by-attempt "
        f"against a starting bankroll of {cfg.starting_bankroll:,.0f} paying "
        f"{cfg.reset_economics.evaluation_fee:,.0f} for the first attempt and "
        f"{cfg.reset_economics.resolved_reset_fee():,.0f} for every reset after it, crediting "
        f"{cfg.reset_economics.profit_split_pct:.0f}% of every gross payout, up to "
        f"{cfg.reset_economics.max_attempts} attempt(s) per life. {stop_text}. This is "
        "resampling-order risk on the historical edge, not an independent out-of-sample test."
    )


def simulate_bankroll_survival(
    trades: list[Trade],
    rules: PropRules,
    cfg: BankrollConfig | None = None,
) -> BankrollSurvivalResult:
    """
    The actual "if I put $X in, do I have a high probability of a good
    return" answer -- a gambler's-ruin-style calculation over the same
    chained reset-on-breach mechanics app.monte_carlo.engine exposes with
    an infinite-bankroll assumption, now bounded by what the trader can
    actually afford to keep paying for resets.
    """
    cfg = cfg or BankrollConfig()
    if not trades:
        raise ValueError("Cannot simulate bankroll survival with zero trades.")

    rng = np.random.default_rng(cfg.random_seed)
    base_pnls = np.array([t.pnl for t in trades], dtype=float)
    base_dates = [pd.Timestamp(t.entry_time).normalize() for t in trades]
    day_structure = precompute_day_structure(base_dates)
    resample_cfg = _ResampleCfg(cfg.method, cfg.block_size)
    econ = cfg.reset_economics

    reached_payout_flags: list[bool] = []
    ruined_before_payout_flags: list[bool] = []
    exhausted_without_payout_flags: list[bool] = []
    net_profits: list[float] = []
    attempts_used_list: list[int] = []
    fees_list: list[float] = []
    gross_list: list[float] = []
    low_points: list[float] = []

    for _ in range(cfg.n_simulations):
        sim_pnls = _resample_pnls(rng, base_pnls, resample_cfg)
        sim_pnls = _apply_slippage_stress(sim_pnls, cfg.slippage_stress_pct)
        result = simulate_account(
            sim_pnls, base_dates, rules, _day_structure=day_structure, reset_on_breach=True,
        )

        bankroll = cfg.starting_bankroll
        low_point = bankroll
        reached_payout = False
        ruined = False
        fees_paid = 0.0
        gross_payouts = 0.0
        net_payouts = 0.0
        attempts_used = 0

        for attempt in result.attempts:
            fee = econ.evaluation_fee if attempts_used == 0 else econ.resolved_reset_fee()
            if fee > bankroll:
                ruined = True
                break
            bankroll -= fee
            fees_paid += fee
            attempts_used += 1
            low_point = min(low_point, bankroll)

            if attempt.payout_amount > 0:
                gross_payouts += attempt.payout_amount
                net = attempt.payout_amount * (econ.profit_split_pct / 100.0)
                net_payouts += net
                bankroll += net

            if attempt.reached_first_payout and not reached_payout:
                reached_payout = True
                if cfg.stop_after_first_payout:
                    break

            if attempts_used >= econ.max_attempts:
                break

        net_profit = net_payouts - fees_paid
        reached_payout_flags.append(reached_payout)
        ruined_before_payout_flags.append(ruined and not reached_payout)
        exhausted_without_payout_flags.append((not ruined) and (not reached_payout))
        net_profits.append(net_profit)
        attempts_used_list.append(attempts_used)
        fees_list.append(fees_paid)
        gross_list.append(gross_payouts)
        low_points.append(low_point)

    def _pct(flags: list[bool]) -> float:
        return float(np.mean(flags) * 100.0) if flags else 0.0

    return BankrollSurvivalResult(
        n_simulations=cfg.n_simulations,
        starting_bankroll=cfg.starting_bankroll,
        probability_reach_first_payout=_pct(reached_payout_flags),
        probability_ruin_before_first_payout=_pct(ruined_before_payout_flags),
        probability_exhausted_attempts_without_payout=_pct(exhausted_without_payout_flags),
        expected_net_profit=float(np.mean(net_profits)) if net_profits else 0.0,
        median_net_profit=float(np.median(net_profits)) if net_profits else 0.0,
        probability_net_positive=_pct([p > 0 for p in net_profits]),
        expected_attempts_used=float(np.mean(attempts_used_list)) if attempts_used_list else 0.0,
        median_attempts_used=float(np.median(attempts_used_list)) if attempts_used_list else 0.0,
        expected_fees_paid=float(np.mean(fees_list)) if fees_list else 0.0,
        expected_gross_payouts=float(np.mean(gross_list)) if gross_list else 0.0,
        median_bankroll_low_point=float(np.median(low_points)) if low_points else 0.0,
        worst_bankroll_low_point=float(np.min(low_points)) if low_points else 0.0,
        net_profit_distribution=net_profits,
        attempts_used_distribution=attempts_used_list,
        max_attempts_configured=econ.max_attempts,
        profit_split_pct=econ.profit_split_pct,
        methodology_note=_methodology_note(cfg, len(trades)),
    )
