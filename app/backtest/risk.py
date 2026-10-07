"""
Risk & execution configuration and position sizing.

Position sizing is expressed in generic "units" rather than broker-specific
lots: pnl = units * price_move. This keeps the engine instrument-agnostic
(FX, indices, crypto, etc.) while still allowing pip-based stop/target
distances via `pip_size`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Position-sizing modes (ACCURACY OVERHAUL 2026-10-07, plan section "Stop and
# size"). A real prop trader chooses DOLLARS first, then places a stop that
# fits them. The engine used to do the opposite only: the strategy's stop was
# fixed, size was floored from it, and a stop wider than the budget produced
# 0 contracts and a silently skipped signal (99% of signals in the audited
# ES 1h run). These modes make the choice explicit:
#
#   "skip"            Honest default. Size = whole contracts whose WORST-CASE
#                     loss at the strategy's own stop (stop + exit spread/
#                     slippage + commission) fits the budget. Zero contracts
#                     means the signal is skipped and counted -- there is NO
#                     1-contract rescue that exceeds the budget.
#   "fit_stop"        Reproduces "pick the dollars, then place the stop":
#                     when the strategy's stop does not fit the budget with
#                     one contract, take 1 contract and CAP the stop to the
#                     distance that does fit (target optionally scaled with
#                     it). max_stop_dollars adds an explicit per-contract
#                     ceiling even when the strategy's stop would fit.
#   "fixed_contracts" Always trade `fixed_contracts` contracts at the
#                     strategy's stop. Risk may exceed the budget -- tagged
#                     on the trade, never hidden.
#   "micro_fallback"  Full-size contract when it fits; otherwise the micro
#                     equivalent (ES->MES etc.) when THAT fits; otherwise
#                     skip. Never oversizes.
# ---------------------------------------------------------------------------
SIZING_MODES = ("skip", "fit_stop", "fixed_contracts", "micro_fallback")
FIT_STOP_TARGET_MODES = ("scale", "keep", "fixed_r")

# Reasons a SizingDecision can have units == 0 (the entry is skipped).
SKIP_NO_BUDGET = "no_budget"
SKIP_NO_STOP = "no_stop_distance"
SKIP_STOP_TOO_WIDE = "stop_too_wide_for_budget"
SKIP_COSTS_EXCEED_BUDGET = "costs_exceed_budget"
SKIP_STOP_CAP_TOO_TIGHT = "stop_cap_too_tight"
SKIP_NO_MICRO = "no_micro_defined"
SKIP_TOO_WIDE_FOR_MICRO = "stop_too_wide_even_for_micro"
SKIP_POSITION_CAP = "position_cap_below_one_contract"


@dataclass(frozen=True)
class SizingDecision:
    """Everything the engine needs to open (or skip) one position, computed
    in ONE place (RiskConfig.size_for_stop) so the bar engine, the vectorized
    fast path and the preflight all size identically."""
    units: float = 0.0                      # sizing units (0 = skip the entry)
    contracts: float = 0.0                  # whole contracts (0 when the instrument has no contract_size)
    stop_distance: float = 0.0              # price distance the position is protected with (possibly capped)
    original_stop_distance: float = 0.0     # the strategy's own stop distance
    stop_capped: bool = False
    stop_scale: float = 1.0                 # stop_distance / original_stop_distance
    budget: float = 0.0                     # dollar risk budget this trade was sized against
    risk_at_stop: float = 0.0               # worst-case $ loss at the stop (no gap): stop + exit costs + commission
    skip_reason: str | None = None
    contract_size: float | None = None      # units per contract actually used (micro-aware)
    commission_per_contract: float = 0.0    # per-contract commission actually used (micro-aware)
    used_micro: bool = False
    above_budget: bool = False              # True only for fixed_contracts when the stop risk exceeds the budget


@dataclass
class RiskConfig:
    initial_balance: float = 10_000.0
    risk_mode: str = "percent"          # "percent" | "fixed"
    risk_value: float = 1.0             # % of equity, or fixed $ amount, per trade
    max_trades_per_day: int = 10
    commission_per_trade: float = 0.0   # flat $ per round-turn trade
    # B2-2 (fill honesty): market entries AND signal-driven exits fill at
    # open[i + entry_fill_lag_bars], not the signal bar's close -- 1 = next bar's open (0 = the
    # signal bar's own open; the old same-bar-close fill is gone by design).
    entry_fill_lag_bars: int = 1
    # B2-4: $ commission per CONTRACT per round-turn, charged at settle on top of
    # commission_per_trade (total = commission_per_trade + commission_per_contract * contracts).
    commission_per_contract: float = 0.0
    # Part A port #4 (time-invalidation stop): None (default) = off; when set, a position older
    # than this many hours is closed if |close - entry| is still within time_stop_atr_band * ATR.
    time_stop_hours: float | None = None
    # Stagnation band for the time stop, in multiples of the 14-bar ATR (astra-quant-agent used 0.15).
    time_stop_atr_band: float = 0.15
    # Part C fix 2 (opt-in): when True, the dead-lock rescue fires at ANY equity level, not just
    # equity < initial_balance; rescued trades are tagged sized_above_risk_target=True.
    allow_single_contract_minimum: bool = False
    slippage_pips: float = 0.0
    spread_pips: float = 0.0
    pip_size: float = 0.0001            # price move that equals "1 pip" (e.g. 0.0001 for EURUSD)
    max_position_size: float | None = None  # cap on units, None = unlimited
    # UPGRADE (Sep 2026 -- futures lot-size realism): every unit above is
    # continuous/fractional by default (correct for FX/CFD-style
    # instruments, which really can be sized to an arbitrary fraction of a
    # lot). A real futures contract cannot -- ES/NQ/GC trade in whole
    # contracts only. Before this field existed, position_size() below
    # would happily size a trade to e.g. 37.5 "units", silently reporting
    # backtest P&L more precise than any account could actually place live
    # -- for a small account risking a modest % against a big-point-value
    # contract (ES = $50/point, NQ = $20/point, GC = $100/point in this
    # engine's "1 unit = $1 per pip_size move" terms), the gap between
    # continuous and whole-contract sizing is often the difference between
    # 0 and 1 contracts, not a rounding rumor. None (default) reproduces
    # every backtest before this field existed, byte for byte, since it's
    # the correct setting for FX/CFD-style instruments. Set this to the
    # number of "units" that make up ONE real contract for a futures
    # instrument (e.g. 50.0 for ES, 20.0 for NQ, 100.0 for GC, given
    # pip_size=1.0 for all three -- see app.backtest.risk.suggest_pip_size)
    # to have position_size() floor to the nearest whole contract instead.
    contract_size: float | None = None
    daily_loss_limit_pct: float | None = None  # % of initial_balance; once a day's REALIZED
    # pnl breaches -this, no new entries are taken for the rest of that calendar day.
    # None = disabled (no circuit breaker; this was the only behavior before this field
    # existed). This is the correct, supported way to give a strategy a daily-loss cutoff --
    # a strategy's own generate_signals() cannot implement this itself (see app/strategy/
    # python.py) because it never sees realized trade outcomes, only price data.
    prop_daily_loss_is_breach: bool = False
    # P1-2: daily-loss semantics switch. False (default) = byte-identical
    # to the historical behavior: a daily-loss-limit forced close only
    # blocks NEW entries for the rest of that day, and trading resumes
    # the next morning. True = the forced close is treated the way most
    # real prop firms treat it -- the account is BLOWN
    # (app.backtest.execution sets account_blown, so no new trades open
    # for the rest of the run, exactly like hitting
    # max_account_drawdown_pct), matching the post-hoc prop simulator's
    # hard-fail reading of a daily-loss breach.
    reentry_cooldown_bars: int = 1
    # FIX (EXEC-002): after an open position is stopped out (or hits its
    # take-profit) intrabar, app.backtest.execution used to let a brand
    # new position open in the SAME direction on that exact same bar's
    # close, with no cooldown at all, as long as the strategy's signal
    # hadn't gone flat -- i.e. a whipsaw bar that clips a tight stop and
    # then closes back at a level the strategy still signals on produced
    # TWO trades and re-exposed the same risk within one bar.
    # P2-1: the default is now 1, not 0 -- a whipsaw bar that clips the
    # stop intrabar can no longer re-open on that same bar's close; the
    # earliest reentry is the next bar. This is the minimal honest
    # cooldown (same-bar reentry inflated trade counts up to 2x). Set
    # explicitly to 0 to opt back into the original, unconditional
    # same-bar reentry -- it is still available as a deliberate choice,
    # just no longer the default.
    # Setting this to N blocks any NEW entry for N bars (measured from
    # the bar the previous position closed on, inclusive) regardless of
    # what the strategy's signal says -- see app.backtest.execution's
    # module docstring for the exact accounting.

    # --- account-survivability hard caps ---------------------------------
    # A real prop/broker account has negative-balance protection and a hard
    # firm-level loss floor: no single trade can ever cost more than the
    # money actually in the account, and once the firm's max-drawdown floor
    # is breached the account is terminated -- it cannot keep incurring
    # further losses. Before these fields existed, a pip_size/instrument
    # mismatch or an honest gap-through fill could size a trade so far off
    # that a single loss exceeded the ENTIRE account (real observed cases:
    # -$50,000 and -$2,534,176 single-run net losses on nominal $50k
    # accounts), which cannot happen in live trading and silently poisoned
    # every downstream risk-of-ruin / eval-pass-probability number. These
    # two caps make that structurally impossible instead of just warning
    # about it after the fact.
    max_loss_per_trade_pct: float | None = None
    # Hard ceiling on any single trade's realized loss, as a % of
    # initial_balance. None (default) falls back to 3x the trade's own
    # intended risk_amount -- generous enough to still show a real gap-
    # through loss as materially worse than a normal stop-out, but bounded.
    max_account_drawdown_pct: float | None = None
    # % of initial_balance the account may lose (from its starting balance)
    # before the engine marks it BLOWN and stops opening new trades for the
    # rest of the run -- mirrors a prop firm terminating a failed account
    # rather than letting the simulation keep "trading" a dead account into
    # deeper and deeper negative equity. None = disabled (no such floor is
    # enforced beyond the per-trade cap above). Typically set from the
    # active PropRules.max_drawdown_pct so the raw backtest and the prop
    # simulation agree on where the account actually dies.
    reset_on_breach: bool = True
    # UPGRADE (2026-09-29, "trade until the data ends"): the raw backtest
    # engine now NEVER stops trading before the last bar of the dataset.
    # Breaching the loss floor is logged as a "breach" event, a fresh
    # account is started at initial_balance, and trading continues -- see
    # run_execution. This used to be opt-in (default False) and its
    # permanent-halt behavior is what made a strategy take a few trades at
    # the very start of a multi-year dataset and then never trade again.
    # `reset_on_breach` is kept for backward compatibility with every saved
    # config / UI form that passes it, but the raw engine no longer honors
    # False as "halt" -- use `halt_on_breach` below if you truly want the
    # legacy permanent-halt behavior (it exists for tests and for anyone
    # who wants to see exactly where the FIRST account would have died).
    # The post-hoc scoring layers (simulate_account / Monte Carlo) still
    # read reset_on_breach themselves for their own pass/fail accounting.
    news_blackout_csv: str | None = None
    # P2-6: path to a CSV with one YYYY-MM-DD date per line (blank lines
    # and #-comments ignored). When set, run_execution blocks ALL new
    # entries on those dates -- the backtest-path equivalent of the
    # news-blackout gate the live engine already enforces. None
    # (default) = disabled, byte-identical to every run before this
    # field existed.
    block_weekend_hold: bool = False
    # P2-6: when True, any open position is force-closed at the close of
    # the week's last bar (Friday for normal market data) and no new
    # entries are taken until Monday -- the backtest-path equivalent of
    # the live engine's weekend-hold ban. False (default) = byte-
    # identical to every run before this field existed.
    max_contracts: int | None = None
    # P2-6: cap on whole contracts per position, enforced in
    # position_size() after whole-contract flooring (so it is expressed
    # in the same whole-contract units a real account can place; with no
    # contract_size set, one "contract" is one sizing unit). None
    # (default) = off, byte-identical to every run before this field
    # existed.
    halt_on_breach: bool = False
    # UPGRADE (2026-09-27): the profit-target twin of reset_on_breach
    # above. Before this existed, execution.py had literally no concept
    # of a profit target -- only a breach (loss floor) could ever end an
    # "account" mid-run. In practice that meant a strategy that reached
    # its eval/funded profit target just kept trading on the SAME ever-
    # growing equity number forever, which is not how a real prop
    # account behaves (a real account either gets its profit paid out,
    # funded-stage, or is mechanically replaced by a fresh eval purchase
    # once passed) -- and, worse, gave no way to see how the strategy
    # performs across MANY such cycles the way reset_on_breach already
    # does for the loss side. profit_target_pct/reset_on_target close
    # that gap the same way: None/False = fully disabled, byte-identical
    # to every backtest before this field existed.
    profit_target_pct: float | None = None
    # % of initial_balance in REALIZED profit (never floating/open P&L --
    # see run_execution's payout check) that triggers a payout event once
    # reset_on_target is True. Each payout withdraws the profit above the
    # current baseline (equity is brought back down to that baseline,
    # exactly like a funded account being paid out or a passed eval being
    # "cashed in") and the SAME dollar profit_target_pct-of-initial_balance
    # gain is required again for the next one -- so the run keeps trading
    # and can rack up any number of payouts, rather than stalling flat the
    # instant the target is first hit. Never forces the open position
    # closed (a payout is not an account termination) and is completely
    # independent of reset_on_breach -- a run can have both, either, or
    # neither enabled.
    reset_on_target: bool = True
    # UPGRADE (2026-09-29): now ON by default. Whenever a profit target is
    # known (profit_target_pct set directly, or filled in from the prop
    # rules' evaluation_profit_target_pct by with_prop_safety_defaults),
    # reaching it is logged as a "payout" event, the profit above baseline
    # is banked, and trading continues on a fresh baseline -- the engine
    # never idles once a target is hit. With no target configured this is
    # a no-op (there is nothing to reach).
    # FIX (2026-09-18): every optimization tab already offers a
    # "reset-on-breach" checkbox ("score on the basis that a blown account
    # gets a fresh eval and keeps going, not a dead end") and threads it
    # into the POST-HOC scoring layer (app.prop.simulator.simulate_account,
    # app.monte_carlo.engine.MonteCarloConfig) -- but none of those callers
    # ever set THIS field, so the RAW bar-by-bar backtest that produces the
    # trade list those layers score (app.backtest.execution.run_execution)
    # never knew about it. Its account-blown circuit breaker (see
    # max_account_drawdown_pct above) permanently stopped opening new
    # trades for the rest of the run -- often years of remaining data --
    # the instant the first account blew, regardless of this flag. The
    # post-hoc layers could only ever rebuy/resample within the handful of
    # trades that occurred before that permanent halt; they could never
    # cause the strategy's own signal to actually keep trading against the
    # rest of the dataset, which is what "gets a fresh eval and keeps
    # going" was supposed to mean. This is why a strategy would take a
    # handful of trades right at the start of a multi-year dataset, blow
    # the configured drawdown floor, and then simply never trade again for
    # the remaining years, no matter how much data was fed in.
    #
    # Setting this True makes the RAW backtest itself mechanically "buy a
    # new account" the instant the current one is blown: any position still
    # open at that instant is forced closed first (an account that just got
    # terminated can't keep holding a position), a reset event is recorded
    # (see run_execution's returned equity_df.attrs["account_reset_events"]),
    # and equity resets to initial_balance so entries resume on the very
    # next eligible signal -- exactly the "if I get stopped out, I'll buy a
    # new account and keep going" mental model a prop-firm evaluator (who
    # doesn't care about blowing accounts, only about making money before
    # the max drawdown is hit) actually trades under. False (the default)
    # is byte-for-byte the original permanent-halt behavior -- this is
    # purely additive and changes nothing for any existing caller/saved
    # config that doesn't explicitly set it.

    # ------------------------------------------------------------------
    # ACCURACY OVERHAUL (2026-10-07): sizing modes -- see SIZING_MODES.
    # ------------------------------------------------------------------
    sizing_mode: str = "skip"
    # "skip" | "fit_stop" | "fixed_contracts" | "micro_fallback". The old
    # dead-lock "1-contract rescue" is gone: it opened one contract ABOVE
    # the risk budget on every trade it fired for. The only remaining way to
    # exceed the budget is an explicit opt-in (allow_single_contract_minimum,
    # or sizing_mode="fixed_contracts"), and such trades are tagged.
    max_stop_dollars: float | None = None
    # fit_stop only: hard ceiling on the per-contract dollar risk of the
    # stop (e.g. 450 = never a stop worth more than $450 per contract),
    # applied even when the strategy's own stop would have fit the budget.
    fixed_contracts: int = 1
    # fixed_contracts mode: contracts per trade.
    fit_stop_min_fraction: float = 0.2
    # fit_stop only: refuse to shrink the strategy's stop below this
    # fraction of its own width -- a stop squeezed to a sliver is a
    # different strategy, so the entry is skipped (stop_cap_too_tight)
    # instead of silently trading something the strategy never defined.
    fit_stop_target: str = "scale"
    # What happens to the take-profit distance when fit_stop capped the
    # stop: "scale" (shrink it by the same ratio, keeping the strategy's
    # reward:risk), "keep" (leave the strategy's own target), "fixed_r"
    # (target = fit_stop_target_r x the capped stop, e.g. 1.0 = 1:1).
    fit_stop_target_r: float = 1.0
    micro_contract_size: float | None = None
    # micro_fallback only: sizing units per ONE micro contract (MES = 5).
    # Filled by app.data.instrument_specs.apply_instrument_spec from the
    # instrument's micro equivalent. None = no micro available.
    micro_commission_per_contract: float | None = None
    # micro_fallback only: $ round-turn commission per micro contract.
    intrabar_replay: bool = False
    # When True AND the caller supplies a finer-timeframe frame (see
    # run_execution's intrabar_df), every open trade is walked bar-by-bar
    # through the finer data to resolve which of stop/target traded first
    # inside a strategy bar. Off by default: byte-identical to before.
    account_model: str = "legacy"
    # "legacy" keeps the old realized-equity static-floor breach check and
    # the equity teleport-on-reset. "prop" routes every breach / daily-loss
    # decision through app.prop.account.PropAccount (trailing high-water
    # mark, floating-equity breach, firm session day, lock) and ENDS each
    # attempt instead of teleporting equity.
    prop_account_rules: object | None = None
    # PropRules instance consumed by account_model="prop". Typed `object`
    # to keep this module free of an app.prop import cycle.

    # ------------------------------------------------------------------
    # Cost helpers (costs belong INSIDE the sizing budget).
    # ------------------------------------------------------------------
    def side_cost_price(self) -> float:
        """Price-unit cost of ONE fill side: spread + slippage. The engine
        charges this on the entry fill AND on the exit fill."""
        return (self.spread_pips + self.slippage_pips) * self.pip_size

    def commission_for(self, contracts: float, commission_per_contract: float | None = None) -> float:
        per = self.commission_per_contract if commission_per_contract is None else commission_per_contract
        return self.commission_per_trade + per * contracts

    def worst_case_loss(
        self, units: float, stop_distance: float, contract_size: float | None = None,
        commission_per_contract: float | None = None,
    ) -> float:
        """Worst-case $ loss of `units` stopped out at exactly `stop_distance`
        (price units) with no gap: the stop distance itself, the EXIT side's
        spread+slippage (the entry side's cost is already inside the filled
        entry price the stop is anchored to), plus commission. This is the
        number the budget must cover -- the audit's "100% of trades
        overshoot" was this cost being ignored by the sizing budget."""
        cs = self.contract_size if contract_size is None else contract_size
        contracts = (units / cs) if cs else 0.0
        return units * (stop_distance + self.side_cost_price()) + self.commission_for(contracts, commission_per_contract)

    def round_trip_cost_dollars(self, units: float, contract_size: float | None = None) -> float:
        """Round-trip $ cost (entry + exit spread/slippage + commission) of
        `units`, independent of any stop. For preflight/diagnostics."""
        cs = self.contract_size if contract_size is None else contract_size
        contracts = (units / cs) if cs else 0.0
        return 2.0 * self.side_cost_price() * units + self.commission_for(contracts)

    def risk_amount(self, current_equity: float) -> float:
        # Floor equity at 0 for sizing purposes: a negative-equity account
        # is already blown (see max_account_drawdown_pct / account_blown
        # handling in app.backtest.execution) and must never be sized as if
        # it had negative money to risk, which previously flipped the sign
        # of "%-of-equity" sizing and could make losses compound instead of
        # halting.
        equity_for_sizing = max(current_equity, 0.0)
        if self.risk_mode == "fixed":
            return max(self.risk_value, 0.0)
        return max(equity_for_sizing * (self.risk_value / 100.0), 0.0)

    def size_for_stop(self, current_equity: float, stop_distance: float) -> "SizingDecision":
        """The ONE sizing routine (see SIZING_MODES). `stop_distance` is the
        strategy's own stop in PRICE units. Returns a SizingDecision whose
        `units` is 0 when the entry must be skipped, with `skip_reason`
        saying why -- never a silent zero.

        Budget accounting: the worst-case loss at the stop INCLUDES the
        exit-side spread/slippage and commission (see worst_case_loss), so a
        trade that stops out for exactly its sized risk never reads as an
        "overshoot"."""
        mode = self.sizing_mode if self.sizing_mode in SIZING_MODES else "skip"
        budget = self.risk_amount(current_equity)
        D = float(stop_distance) if stop_distance is not None and math.isfinite(stop_distance) else 0.0
        if budget <= 0:
            return SizingDecision(budget=budget, original_stop_distance=D, skip_reason=SKIP_NO_BUDGET)
        if D <= 0:
            return SizingDecision(budget=budget, original_stop_distance=D, skip_reason=SKIP_NO_STOP)

        c = self.side_cost_price()
        fixed = self.commission_per_trade
        avail = budget - fixed

        # -- fractional-unit instruments (FX/CFD/crypto): no whole-contract
        # constraint, so every mode reduces to "size so the worst case fits".
        if not self.contract_size:
            if avail <= 0:
                return SizingDecision(budget=budget, original_stop_distance=D, stop_distance=D,
                                      skip_reason=SKIP_COSTS_EXCEED_BUDGET)
            units = avail / (D + c)
            if self.max_position_size is not None:
                units = min(units, self.max_position_size)
            if self.max_contracts is not None:
                units = min(units, float(self.max_contracts))
            units = max(units, 0.0)
            return SizingDecision(
                units=units, contracts=0.0, stop_distance=D, original_stop_distance=D, budget=budget,
                risk_at_stop=self.worst_case_loss(units, D), contract_size=None,
                commission_per_contract=self.commission_per_contract,
                skip_reason=None if units > 0 else SKIP_NO_BUDGET,
            )

        cs = float(self.contract_size)
        cpc = float(self.commission_per_contract)

        def _cap_contracts(n: int, contract_units: float) -> int:
            if self.max_position_size is not None:
                n = min(n, int(math.floor(self.max_position_size / contract_units + 1e-9)))
            if self.max_contracts is not None:
                n = min(n, int(self.max_contracts))
            return max(n, 0)

        def _fit(dist: float, contract_units: float, per_contract_commission: float) -> int:
            denom = contract_units * (dist + c) + per_contract_commission
            if avail <= 0 or denom <= 0:
                return 0
            return int(math.floor(avail / denom + 1e-9))

        def _decision(n: int, dist: float, contract_units: float, per_contract_commission: float,
                      capped: bool = False, micro: bool = False, above: bool = False) -> "SizingDecision":
            units = n * contract_units
            return SizingDecision(
                units=units, contracts=float(n), stop_distance=dist, original_stop_distance=D,
                stop_capped=capped, stop_scale=(dist / D) if D else 1.0, budget=budget,
                risk_at_stop=self.worst_case_loss(units, dist, contract_units, per_contract_commission),
                contract_size=contract_units, commission_per_contract=per_contract_commission,
                used_micro=micro, above_budget=above,
            )

        def _skip(reason: str, dist: float = D) -> "SizingDecision":
            return SizingDecision(budget=budget, original_stop_distance=D, stop_distance=dist,
                                  skip_reason=reason, contract_size=cs, commission_per_contract=cpc)

        if mode == "fixed_contracts":
            n = _cap_contracts(max(int(self.fixed_contracts), 1), cs)
            if n <= 0:
                return _skip(SKIP_POSITION_CAP)
            d = _decision(n, D, cs, cpc)
            if d.risk_at_stop > budget * 1.001:
                d = _decision(n, D, cs, cpc, above=True)
            return d

        if mode == "fit_stop":
            D_eff = D
            if self.max_stop_dollars is not None and self.max_stop_dollars > 0:
                D_eff = min(D_eff, self.max_stop_dollars / cs)
            n = _cap_contracts(_fit(D_eff, cs, cpc), cs)
            if n >= 1:
                return _decision(n, D_eff, cs, cpc, capped=D_eff < D * (1 - 1e-12))
            if _cap_contracts(1, cs) < 1:
                return _skip(SKIP_POSITION_CAP)
            # One contract at the (possibly ceilinged) stop still busts the
            # budget: keep 1 contract and shrink the STOP to the distance
            # whose worst case lands exactly on the budget.
            d_fit = (avail - cpc) / cs - c
            if d_fit <= 0:
                return _skip(SKIP_COSTS_EXCEED_BUDGET)
            D_cap = min(D_eff, d_fit)
            if D_cap < self.fit_stop_min_fraction * D:
                return _skip(SKIP_STOP_CAP_TOO_TIGHT, dist=D_cap)
            return _decision(1, D_cap, cs, cpc, capped=True)

        if mode == "micro_fallback":
            n = _cap_contracts(_fit(D, cs, cpc), cs)
            if n >= 1:
                return _decision(n, D, cs, cpc)
            if not self.micro_contract_size:
                return _skip(SKIP_NO_MICRO)
            m_cs = float(self.micro_contract_size)
            m_cpc = float(self.micro_commission_per_contract if self.micro_commission_per_contract is not None else cpc)
            # a micro's contract cap is expressed in micro contracts
            n_m = _fit(D, m_cs, m_cpc)
            if self.max_position_size is not None:
                n_m = min(n_m, int(math.floor(self.max_position_size / m_cs + 1e-9)))
            if self.max_contracts is not None:
                n_m = min(n_m, int(self.max_contracts) * max(int(round(cs / m_cs)), 1))
            if n_m >= 1:
                return _decision(n_m, D, m_cs, m_cpc, micro=True)
            return _skip(SKIP_TOO_WIDE_FOR_MICRO)

        # mode == "skip"
        n = _cap_contracts(_fit(D, cs, cpc), cs)
        if n >= 1:
            return _decision(n, D, cs, cpc)
        if avail <= 0 or (cs * c + cpc) >= avail:
            return _skip(SKIP_COSTS_EXCEED_BUDGET)
        return _skip(SKIP_STOP_TOO_WIDE)

    def position_size(self, current_equity: float, stop_loss_pips: float) -> float:
        """Units such that a full stop-out loses (at most, after whole-
        contract rounding -- see contract_size's own docstring) the risk
        budget, COSTS INCLUDED. Thin wrapper over size_for_stop kept for the
        callers (fast path, UI, sweeps) that only want a size; callers that
        need the capped stop / skip reason / micro swap should call
        size_for_stop directly."""
        if not stop_loss_pips or stop_loss_pips <= 0:
            stop_loss_pips = 10.0  # sane fallback so sizing never divides by zero
        return self.size_for_stop(current_equity, stop_loss_pips * self.pip_size).units

    def sizing_floored_to_zero_contracts(self, current_equity: float, stop_loss_pips: float) -> bool:
        """True when this trade's intended risk was real and positive
        (there WAS money to risk and a real stop distance to size against)
        but whole-contract rounding (or the stop being wider than the
        budget allows) brought it down to exactly 0 -- i.e. THIS specific
        reason for a skipped entry, as opposed to a genuinely degenerate
        stop distance (NaN/zero/negative) or no money at all. See
        run_execution's zero-size-floor warning: without distinguishing
        this case, a run where every single entry gets silently skipped
        reports "0 trades" with no indication of why, which is
        indistinguishable from a strategy that simply never signals."""
        if not self.contract_size or not stop_loss_pips or stop_loss_pips <= 0:
            return False
        d = self.size_for_stop(current_equity, stop_loss_pips * self.pip_size)
        return d.units <= 0 and d.skip_reason in (
            SKIP_STOP_TOO_WIDE, SKIP_COSTS_EXCEED_BUDGET, SKIP_STOP_CAP_TOO_TIGHT,
            SKIP_TOO_WIDE_FOR_MICRO, SKIP_NO_MICRO,
        )

    def max_trade_loss(self, equity_at_entry: float) -> float:
        """Hard dollar ceiling on how much a single trade may realistically
        lose, regardless of how it was sized or how far price gapped past
        its stop. Real prop firms/brokers cap the damage one trade can do
        (negative-balance protection, firm-level daily/overall loss
        floors) -- a simulated trade should never be able to blow past
        that on its own.

        TAIL-RISK CAVEAT (P2-7): the 3x-intended-risk fallback below
        prevents impossible -$2.5M single-trade prints, but a real
        limit-move gap can cost 10x+ intended risk -- so the clamp
        understates genuine tail risk by construction. Widen it with
        max_loss_per_trade_pct (a % of initial_balance) when the
        instrument/timeframe is gap-prone, rather than trusting the
        clamped number as the worst case."""
        if self.max_loss_per_trade_pct is not None:
            return max(self.initial_balance * (self.max_loss_per_trade_pct / 100.0), 0.0)
        return self.risk_amount(equity_at_entry) * 3.0

    def account_blown_floor(self) -> float | None:
        """Equity level at/below which the account is BLOWN and must stop
        opening new trades. None if no such floor is configured."""
        if self.max_account_drawdown_pct is None:
            return None
        return self.initial_balance * (1.0 - self.max_account_drawdown_pct / 100.0)


def with_prop_safety_defaults(risk: "RiskConfig", prop_rules) -> "RiskConfig":
    """Returns a copy of `risk` with max_account_drawdown_pct, AND
    daily_loss_limit_pct filled in from `prop_rules` whenever the caller
    hasn't already set an explicit value of their own for that field, AND
    initial_balance forced to match `prop_rules.account_size`. This is
    what makes the account-blown circuit breaker AND the daily-loss
    circuit breaker (see app.backtest.execution) apply automatically
    during the RAW BACKTEST itself, against the SAME dollar account the
    post-hoc prop simulator (app.prop.simulator.simulate_account) checks
    the finished trade sequence against -- not just later, and not
    against a different balance. Never overrides a value the caller
    explicitly configured on either PERCENTAGE field (max_account_
    drawdown_pct / daily_loss_limit_pct); initial_balance is the one
    exception -- see the RISK-001 fix note below for why.

    FIX (2026-09-12): daily_loss_limit_pct used to be left out of this
    function entirely -- only max_account_drawdown_pct was wired through.
    That meant a strategy could freely keep opening new trades on a day
    that had already blown through the prop firm's daily loss limit
    during its OWN raw backtest (the stage every search/optimization loop
    scores fitness from), with the violation only surfacing later at the
    Monte Carlo / CPCV / prop-simulation stage -- wasting compute
    exploring candidates whose raw backtest was never a realistic
    account to begin with, and understating how early a real account
    would have been forced to stop trading that day. This is exactly the
    'hard, declarative prop-firm-realistic constraint layer at strategy-
    discovery time, not just at simulation time' this app was missing:
    every caller of this function (Speed Run, Full Pipeline, and now
    Evolution Lab -- see app.evolution.engine.EvolutionRunner.__init__)
    gets the fix automatically, with no other code path needing to
    change.

    FIX (RISK-001): `daily_loss_limit_pct` and `max_account_drawdown_pct`
    are PERCENTAGES, and they used to get applied against two different
    dollar bases depending on which layer checked them -- the raw
    backtest's intrabar circuit breaker used `risk.initial_balance`,
    while the post-hoc prop-firm verdict (simulate_account) used
    `prop_rules.account_size`. Both values default to different numbers
    (10,000 vs 100,000) and, worse, both were independently user-editable
    (two separate "Initial balance ($)" / "Account size ($)" fields in
    both the desktop and web UI, with no sync between them), so a
    strategy could cleanly survive the raw-backtest circuit breaker
    (checked against the wrong, too-generous dollar floor) and then be
    silently re-scored against a completely different floor at the final
    verdict, or vice versa. Since PropRules IS the definition of the
    account actually being evaluated, `prop_rules.account_size` is now
    treated as authoritative and always wins here -- this function is
    the one policy chokepoint already shared by every pipeline that
    matters for a research verdict, so fixing it here fixes the
    divergence everywhere this function is already called. Callers that
    construct a RiskConfig/PropRules pair directly (rather than through
    a pipeline that calls this function) should call
    account_size_mismatch_message() themselves first if they want to
    warn the person BEFORE the values get silently reconciled -- see its
    docstring."""
    from dataclasses import replace
    updates: dict = {}
    if risk.max_account_drawdown_pct is None:
        max_dd = getattr(prop_rules, "max_drawdown_pct", None)
        if max_dd is not None:
            updates["max_account_drawdown_pct"] = max_dd
    if risk.daily_loss_limit_pct is None:
        daily_loss = getattr(prop_rules, "daily_loss_limit_pct", None)
        if daily_loss is not None:
            updates["daily_loss_limit_pct"] = daily_loss
    if risk.profit_target_pct is None and risk.reset_on_target:
        # Same number the prop-firm verdict scores against: reaching it is
        # a logged payout/"target reached" event and trading continues,
        # rather than the run going quiet once the account has "won".
        eval_target = getattr(prop_rules, "evaluation_profit_target_pct", None)
        if eval_target:
            updates["profit_target_pct"] = float(eval_target)
    account_size = getattr(prop_rules, "account_size", None)
    if account_size is not None and risk.initial_balance != account_size:
        updates["initial_balance"] = account_size
    if not updates:
        return risk
    return replace(risk, **updates)


def account_size_mismatch_message(initial_balance: float, account_size: float) -> str | None:
    """RISK-001: None if `initial_balance` (RiskConfig, drives position
    sizing and the raw backtest's own intrabar circuit breakers) and
    `account_size` (PropRules, what the final prop-firm verdict is
    actually computed against) already agree -- otherwise an actionable
    message explaining the mismatch and that with_prop_safety_defaults()
    will make `account_size` win. Callers that build a RiskConfig and a
    PropRules directly (web routes, the desktop UI, any script) should
    call this BEFORE calling with_prop_safety_defaults so the person
    sees why their numbers just changed, the same way
    instrument_scale_mismatch_message() is surfaced before its own
    silent-but-safe correction."""
    if initial_balance == account_size:
        return None
    return (
        f"Account-size mismatch: Risk config's 'Initial balance' (${initial_balance:,.2f}) does not "
        f"match the prop rules' 'Account size' (${account_size:,.2f}). These must be the same dollar "
        "account -- position sizing, the raw backtest's daily-loss/max-drawdown circuit breakers, AND "
        "the final prop-firm pass/fail verdict all need to agree on how big the account actually is, "
        "or a strategy can silently survive one stage's check and fail (or pass) the other's, on the "
        "exact same trades, purely because they used different dollar floors for the same percentage "
        f"rule. Using the prop rules' account size (${account_size:,.2f}) for this run -- update the "
        "'Initial balance' field to match if that wasn't intended."
    )


_ACCOUNT_SIZE_MISMATCH_MARKER = "Account-size mismatch:"


def has_account_size_mismatch(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` is account_size_mismatch_message's
    warning -- same shared-detection pattern as has_instrument_scale_mismatch
    and has_impossible_condition, so every caller escalates this the same way."""
    return any(_ACCOUNT_SIZE_MISMATCH_MARKER in w for w in warnings)


def suggest_pip_size(df) -> float:
    """Suggests a starting pip_size from a loaded OHLCV DataFrame's actual
    price scale, purely by magnitude of the median close price. This is a
    starting point for the person to confirm, not an authoritative
    per-instrument lookup (it can't distinguish gold from a $2,000 index,
    for instance) -- it exists because leaving pip_size at its FX default
    (0.0001) against a non-FX-scaled instrument (stocks, indices, crypto,
    JPY pairs) is the single most common cause of a strategy's fixed-pips
    stop translating into a nonsensical position size (see the
    pip_scale_mismatch warning in app.backtest.execution).

    Rough bands, all "1 pip = smallest meaningful price increment" for
    that price level:
      >= 500          -> 1.0    (large-index / high-priced-crypto scale)
      >= 20            -> 0.01   (typical stock-in-dollars or JPY-pair scale)
      >= 5             -> 0.01   (lower-priced stocks; still cent-scale)
      < 5              -> 0.0001 (FX-major scale, e.g. EURUSD ~1.10)
    """
    if df is None or "close" not in getattr(df, "columns", []) or len(df) == 0:
        return 0.0001
    median_price = float(df["close"].abs().median())
    if not median_price or median_price != median_price:  # NaN guard
        return 0.0001
    if median_price >= 500:
        return 1.0
    if median_price >= 5:
        return 0.01
    return 0.0001


# The two exact substrings app.backtest.execution's run_backtest emits
# when a fixed-pips stop is an implausible fraction of either the
# instrument's raw price (pip_scale_mismatch) or its own recent ATR
# (atr_scale_mismatch) -- see that module for the full warning text.
# Both are checked (not just the price-ratio one) because a fixed-pips
# stop can pass the price-ratio check -- look like a perfectly ordinary
# fraction of price -- while still being tiny next to the instrument's
# own actual volatility; a high-priced but volatile instrument such as
# an equity index is the case that price-ratio alone misses.
_INSTRUMENT_MISMATCH_MARKERS = (
    "doesn't match the instrument actually being tested",
    "under 15% of this instrument's own recent ATR",
)


def has_instrument_scale_mismatch(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` (e.g. a BacktestResult.warnings
    list) is app.backtest.execution's pip_scale_mismatch or
    atr_scale_mismatch warning -- the single most common cause of a
    strategy's numbers being unreliable (see suggest_pip_size above).
    Shared by app.orchestration.full_pipeline (which skips its GA search
    on this) and app.orchestration.quick_optimize (which surfaces it
    prominently instead) so both tools agree on exactly what counts."""
    return any(marker in w for w in warnings for marker in _INSTRUMENT_MISMATCH_MARKERS)


def has_impossible_condition(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` is app.strategy.manual's
    validate_bounded_conditions warning -- a condition comparing a
    bounded oscillator (RSI, Stochastic, MFI, ...) to a threshold outside
    its possible range, which can never be satisfied and permanently
    disables that branch of the strategy's logic. Shared by Quick
    Optimize and Full Pipeline so both escalate it the same prominent
    way they already escalate has_instrument_scale_mismatch."""
    from app.strategy.manual import IMPOSSIBLE_CONDITION_MARKER
    return any(IMPOSSIBLE_CONDITION_MARKER in w for w in warnings)


# UPGRADE (buried-position-sizing-deviation): the % of trades computed
# by app.backtest.statistics.compute_risk_reconciliation, below which a
# report's muted informational note stays the only place this shows up.
# At/above this, it's escalated into an actual warning on
# BacktestResult.warnings (see app.backtest.engine.run_backtest) -- the
# same mechanism has_instrument_scale_mismatch/has_impossible_condition
# above already use, so it shows up in the report's real "Execution
# warnings" banner and gets a chance to be scored into a verdict/result,
# not just a line a report has to be scrolled to.
POSITION_SIZING_DEVIATION_WARNING_THRESHOLD_PCT = 20.0
_POSITION_SIZING_DEVIATION_MARKER = "Position-sizing deviation:"


def has_position_sizing_deviation(warnings: "list[str]") -> bool:
    """True if any warning in `warnings` is position_sizing_deviation_
    message's own warning -- lets a caller escalate it the same
    prominent way Full Pipeline/Quick Optimize already escalate
    has_instrument_scale_mismatch, without re-deriving the stats."""
    return any(_POSITION_SIZING_DEVIATION_MARKER in w for w in warnings)


def position_sizing_deviation_message(
    stats: dict, threshold_pct: float = POSITION_SIZING_DEVIATION_WARNING_THRESHOLD_PCT,
) -> str | None:
    """Surfaces app.backtest.statistics.compute_risk_reconciliation's
    pct_trades_position_capped / pct_trades_risk_overshoot as an actual
    warning once either one is material, instead of leaving them as
    numbers a report table computes but nothing ever flags: a strategy
    that looks fine on INTENDED risk (RiskConfig.risk_value) can behave
    very differently at the real, rounded size actually taken -- common
    on instruments with coarse per-contract/point granularity (e.g.
    MNQ's $2/point) combined with whole-contract position-size rounding.

    Checked per-direction against `threshold_pct` rather than summed --
    "10% capped + 10% overshoot" describes two much smaller, opposite
    problems, not one material combined one (see compute_risk_
    reconciliation's docstring for what each direction actually means).
    Returns None when neither direction is material, e.g. every trade
    with no intended_risk_dollars recorded at all (no message to give)."""
    pct_capped = stats.get("pct_trades_position_capped", 0.0) or 0.0
    pct_overshoot = stats.get("pct_trades_risk_overshoot", 0.0) or 0.0
    if pct_capped < threshold_pct and pct_overshoot < threshold_pct:
        return None
    if pct_capped >= pct_overshoot:
        # ADAPTIVE-RISK-ATTRIBUTION: compute_risk_reconciliation now hands
        # back direct trade-level evidence of whether the adaptive-risk
        # throttle was the actual driver, instead of this message having
        # to hedge across three unverified possibilities. Only named
        # explicitly once the evidence supports it (throttle active on at
        # least `threshold_pct` of entries AND materially shrinking size);
        # otherwise the cause is left to the remaining, still-unverified
        # possibilities (a position cap or whole-contract rounding), with
        # adaptive risk dropped from the list since the trades themselves
        # show it wasn't the (main) cause here.
        pct_adaptive_active = stats.get("pct_trades_adaptive_throttle_active", 0.0) or 0.0
        avg_adaptive_multiplier = stats.get("avg_adaptive_risk_multiplier", 1.0)
        if avg_adaptive_multiplier is None:
            avg_adaptive_multiplier = 1.0
        adaptive_is_driver = pct_adaptive_active >= threshold_pct and avg_adaptive_multiplier < 0.98
        if adaptive_is_driver:
            direction = (
                f"sized BELOW the configured risk target -- driven mainly by the adaptive-risk "
                f"throttle, active on {pct_adaptive_active:.0f}% of entries with an average size "
                f"multiplier of {avg_adaptive_multiplier:.2f}x (see Adaptive Risk in the report for "
                "which rule(s) fired)"
            )
        else:
            direction = (
                "sized BELOW the configured risk target (a max-position-size cap, whole-contract "
                "rounding on a coarse-granularity instrument, or another sizing constraint -- the "
                "trade data rules out the adaptive-risk throttle as the cause here)"
            )
        pct = pct_capped
    else:
        pct, direction = pct_overshoot, (
            "realized MORE loss than their own actual configured stop risk (almost always a "
            "gap-through fill -- see the gap-loss warning above, if also shown)"
        )
    return (
        f"{_POSITION_SIZING_DEVIATION_MARKER} {pct:.0f}% of trades {direction}. A strategy that "
        "looks fine on INTENDED risk can behave very differently at the REAL, rounded size actually "
        "taken. See the risk reconciliation table in the report (avg_intended_risk_dollars vs "
        "avg_actual_stop_risk_dollars) before trusting the headline eval-pass/risk-of-ruin numbers "
        "at face value."
    )


def instrument_scale_mismatch_message(pip_size: float) -> str:
    """Shared, actionable explanation shown wherever
    has_instrument_scale_mismatch() is True -- one copy of the wording so
    Full Pipeline and Quick Optimize never drift into saying two
    different things about the same problem."""
    return (
        f"Pip-size/instrument-scale mismatch detected (current pip_size: {pip_size}). Every position "
        "size and stop distance this run computed is unreliable -- this almost always means "
        "risk.pip_size doesn't match the instrument actually being tested (e.g. an FX-calibrated "
        "0.0001 run against gold, an index, crypto, or a JPY pair). Set pip_size to match the real "
        "instrument (e.g. 0.01 for gold/JPY pairs, 1.0 for high-priced indices/stocks -- see "
        "suggest_pip_size, or click \"DETECT PIP SIZE FROM DATA\") before trusting this result."
    )
