"""
Prop-firm rule presets -- one-click PropRules for the major firms, instead
of typing account size / profit target / daily loss / max drawdown /
consistency rule by hand every single time.

Why this exists: nothing else in this app answers "which firm's rules did
I mean" -- every tool (Speed Run, Evolution Lab, Full Pipeline, Payout
Probability, ...) just takes a raw PropRules built from whatever numbers
are typed into that page's form. That's a real, boring source of user
error that has nothing to do with strategy quality: a mistyped drawdown
%, or a "static" vs "trailing" drawdown mix-up, silently produces a
wrong eval-pass-probability number with no indication anything is off.

IMPORTANT -- these numbers WILL go stale. Prop firms change their rules
often (profit targets, drawdown %, consistency rules, and especially
payout terms are the fields that move most). Every preset below is
dated with the day it was last checked against that firm's own public
rules page; treat it as a fast, reviewable starting point to confirm
against the firm's current terms before relying on it, not a live feed.
There is no automatic "check for updates" here -- see
`PropFirmPreset.as_of` and `PropFirmPreset.source_note`.

MODELING LIMITATION (updated v6, Oct 2026) -- consistency_rule_pct is a
single field checked ONLY at the evaluation-pass gate (see
app.prop.simulator.simulate_account: "best_day_profit /
total_profit_since_start <= consistency_rule_pct", checked once, at the
moment balance first clears the profit target). Several firms below
(Apex, Lucid) have NO consistency rule during evaluation at all -- their
real consistency rule only gates PAYOUTS on the funded account. That
funded-stage rule is now modeled separately by the v6 field
PropRules.funded_consistency_rule_pct (checked at every funded payout:
best_day_since_baseline / profit_since_baseline <=
funded_consistency_rule_pct), and the presets below set it (Apex 50.0,
Lucid 40.0, TopStep 40.0, FundedNext 40.0). For those firms,
consistency_rule_pct is correctly None (no eval-stage gate) even though
a real funded-stage consistency rule exists; don't read None as "this
firm has no consistency rule anywhere," and don't "fix" it by putting
the funded-stage number here -- that would incorrectly block the
evaluation-pass check on a rule that, in reality, doesn't apply until
afterward. Each such preset's source_note says where the real
funded-stage number lives instead.

This module is independent of app.prop.simulator's PropRules dataclass
in the sense that it never subclasses or modifies it -- every preset
just returns a plain PropRules instance, so it works everywhere a
hand-built PropRules already works (Speed Run, Evolution Lab, Full
Pipeline, Payout Probability, the Prop-Firm Recommender).
"""
from __future__ import annotations

from datetime import date  # noqa: F401  (type annotations)
from dataclasses import dataclass, field

from app.prop.simulator import PropRules


@dataclass(frozen=True)
class PropFirmPreset:
    key: str                        # stable id, e.g. "ftmo_100k"
    firm: str                       # display name, e.g. "FTMO"
    label: str                      # display label, e.g. "FTMO - $100k Challenge"
    account_size: float
    evaluation_profit_target_pct: float
    daily_loss_limit_pct: float
    max_drawdown_pct: float
    drawdown_type: str              # "trailing" | "static"
    drawdown_check_mode: str        # "intrabar" | "eod"
    consistency_rule_pct: float | None
    min_trading_days: int
    payout_threshold_pct: float = 0.0
    payout_cap_pct: float | None = None
    payout_frequency_days: int = 14
    required_buffer_pct: float = 0.0
    # B10 (v6): funded-stage payout/inactivity fields, passed through to
    # PropRules by to_prop_rules(). Zero = "rule off" for the count/dollar
    # fields (max_inactive_days=0 maps to None, i.e. the inactivity rule
    # disabled; winning_days_for_payout=0 disables the winning-day gate).
    winning_days_for_payout: int = 0
    min_winning_day_profit: float = 0.0
    max_inactive_days: int = 0
    floating_drawdown_mode: str = "realized"
    # B5 (v6): funded-stage consistency rule -- gated at every funded
    # payout (best single day <= this % of profit since the payout
    # baseline), NOT at the evaluation-pass gate. See the module
    # docstring's MODELING LIMITATION note, updated for v6.
    funded_consistency_rule_pct: float | None = None
    # Accuracy overhaul: how the bar engine / PropAccount enforce the rules.
    # "legacy" keeps the pre-overhaul behaviour for presets nobody has
    # re-verified; a verified preset sets them explicitly.
    dd_basis: str = "legacy"        # legacy | realized | eod | floating
    daily_loss_basis: str = "realized"   # realized | floating
    daily_loss_action: str = "fail"      # fail | lock_day
    trailing_lock: bool = False
    trailing_lock_offset_pct: float = 0.0
    max_contracts: int | None = None     # per-firm cap on mini-equivalent contracts
    rules_checked_on: str = ""      # date the dd/lock/contract fields above were checked against the firm
    # v9.16 audit fixes: three rule fields that to_prop_rules() used to drop.
    # trailing_distance_basis "account" = fixed dollar distance (account_size x pct),
    # how futures firms state a trailing MLL; "peak" = % of peak (legacy).
    trailing_distance_basis: str = "peak"
    funded_starts_fresh: bool = True    # see PropRules.funded_starts_fresh (default True since 2026-10-19)
    max_eval_calendar_days: int | None = None   # eval time limit in calendar days (None = no limit)
    # Futures session flatten: positions must be flat by this America/Chicago
    # wall-clock time ("15:45" = 4:45 pm ET). None = rule not modeled.
    flatten_time_ct: str | None = None
    as_of: str = ""                 # date this preset was last checked against the firm's own rules page
    source_note: str = ""           # short pointer to what to re-check and where

    # v7: does this firm EXPLICITLY allow algorithmic/automated trading on
    # the account this preset models? Conservative by design -- True ONLY
    # when the firm's own published rules say so unambiguously. False
    # covers outright bans (Apex), approval-gated policies (The5ers), and
    # account-size/platform-gated ones (FundedNext $50k+), with the reason
    # spelled out in algo_policy_note. Defaults to False so an
    # un-reviewed preset can never imply permission that isn't explicit.
    algo_trading_allowed: bool = False
    algo_policy_note: str = ""      # what the firm's own rules actually say; required whenever the flag was reviewed
    # 2026-10 audit: per-preset verification stamp. ``verified_on`` is the date the RULE fields
    # (target, drawdown, daily loss, basis, lock, consistency) were checked against a firm page;
    # ``source_url`` is the page. Empty verified_on = unverified = NOT eligible for a READY verdict
    # (see is_ready_eligible). ``verification_tier``: "primary" (firm's own page/help centre),
    # "secondary" (reviewer/aggregator, needs a confirm), "owner_attested" (the account owner signed it off;
    # not an independent page check), "" (not verified).
    verified_on: str = ""
    source_url: str = ""
    verification_tier: str = ""

    def to_prop_rules(self) -> PropRules:
        return PropRules(
            account_size=self.account_size,
            evaluation_profit_target_pct=self.evaluation_profit_target_pct,
            daily_loss_limit_pct=self.daily_loss_limit_pct,
            max_drawdown_pct=self.max_drawdown_pct,
            drawdown_type=self.drawdown_type,
            drawdown_check_mode=self.drawdown_check_mode,
            consistency_rule_pct=self.consistency_rule_pct,
            min_trading_days=self.min_trading_days,
            payout_threshold_pct=self.payout_threshold_pct,
            payout_cap_pct=self.payout_cap_pct,
            payout_frequency_days=self.payout_frequency_days,
            required_buffer_pct=self.required_buffer_pct,
            winning_days_for_payout=self.winning_days_for_payout,
            min_winning_day_profit=self.min_winning_day_profit,
            # preset convention: 0 (or negative) = inactivity rule off
            max_inactive_days=self.max_inactive_days if self.max_inactive_days > 0 else None,
            floating_drawdown_mode=self.floating_drawdown_mode,
            funded_consistency_rule_pct=self.funded_consistency_rule_pct,
            dd_basis=self.dd_basis,
            daily_loss_basis=self.daily_loss_basis,
            daily_loss_action=self.daily_loss_action,
            trailing_lock=self.trailing_lock,
            trailing_lock_offset_pct=self.trailing_lock_offset_pct,
            max_contracts=self.max_contracts,
            rules_checked_on=self.rules_checked_on,
            trailing_distance_basis=self.trailing_distance_basis,
            funded_starts_fresh=self.funded_starts_fresh,
            max_eval_calendar_days=self.max_eval_calendar_days,
            flatten_time_ct=self.flatten_time_ct,
            preset_key=self.key,
            rules_verified=is_ready_eligible(self),
        )

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        return d


# ---------------------------------------------------------------------------
# Preset catalog.
#
# Every firm below is a well-known one-step/two-step (or instant-style)
# prop-firm evaluation whose broad rule SHAPE (a profit target, a daily
# loss limit, a max drawdown, some form of consistency rule) is stable
# even when the exact numbers move around release to release. Multiple
# account sizes are offered per firm where the firm itself offers them.
#
# THESE NUMBERS ARE APPROXIMATE AND WILL DRIFT -- see module docstring.
#
# Full pass re-verified 2026-09-13 against each firm's own site/help-center
# pages where reachable, cross-checked against several independent 2026
# review sources per firm. Apex and Lucid in particular had gone stale:
# Apex overhauled its rules in "Apex 4.0" (March 2026, removing the
# evaluation consistency rule and the minimum-trading-days rule entirely),
# and Lucid's payout cycle was previously modeled as 14 days when the
# firm's own current pricing page (lucidtrading.com) advertises 3 days on
# its flagship LucidPro track -- a real, current discrepancy, not just an
# old snapshot. See each preset's source_note for what's still worth a
# manual double-check (mainly exact dollar drawdown amounts on firms that
# now run several parallel product variants) and where the remaining
# uncertainty lives.
# ---------------------------------------------------------------------------

_AS_OF = "2026-09-13"  # last time this catalog's numbers were checked/updated
_AS_OF_V6 = "2026-10-04"  # v6 pass (Oct 2026): Apex split into EOD/Intraday
# account types with confirmed drawdown dollars, FTMO drawdown monitoring
# corrected to intrabar, FundedNext re-mapped to the Stellar 2-Step
# product, The5ers Bootcamp relabeled to High-Stakes, and funded-stage
# consistency / payout-gate fields (B5/B10) wired through to_prop_rules.
_AS_OF_V7 = "2026-10-05"  # v7 pass (Oct 2026): algo_trading_allowed field added,
# all 15 legacy presets backfilled from firms' own rules pages, 4 new
# algo-friendly presets (E8 One, FunderPro One Phase, Atlas 1-Step,
# The Futures Desk) with as-of dates from their own help centers.
# NEVER an Alpha Futures preset -- Alpha prohibits algorithmic trading.

PROP_FIRM_PRESETS: list[PropFirmPreset] = [
    # --- FTMO ---------------------------------------------------------
    # Re-verified directly against ftmo.com/en/trading-objectives/ (2026-09-13):
    # matches the FTMO Challenge: 2-Step's Phase 1 exactly -- 10% profit
    # target, 5% daily loss (of Initial Simulated Capital), 10% STATIC
    # max loss (fixed at Initial Simulated Capital - 10%, confirmed not
    # trailing on the 2-Step route), and a confirmed 4 minimum trading
    # days. Payout: FTMO's own payout-policy pages and most 2026 reviews
    # converge on a 14-day first-payout window from the funded account's
    # first trade. v6 (Oct 2026): drawdown_check_mode corrected to
    # "intrabar" -- FTMO monitors equity intraday (can breach mid-day),
    # not just at EOD; and max_inactive_days=30 models FTMO's inactivity
    # rule (fail after 30+ calendar days with no trading activity).
    PropFirmPreset(
        key="ftmo_10k", firm="FTMO", label="FTMO - $10k Challenge",
        account_size=10_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        max_inactive_days=30,
        as_of=_AS_OF_V6,
        source_note=(
            "Confirmed 2026-09-13 against ftmo.com/en/trading-objectives/ -- models the FTMO Challenge: "
            "2-Step (Phase 1 numbers; Phase 2 relaxes the profit target to 5%, which this single-PropRules "
            "model can't represent as a separate phase). FTMO also now offers a 1-Step Challenge (launched "
            "Feb 2026: single 10% target, tighter 3% daily loss, 10% TRAILING max loss, a 50%-of-positive-"
            "days 'Best Day Rule' instead of a minimum-trading-days rule) -- not modeled as a separate "
            "preset here; add one if the 1-Step route matters for your timeline."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "FTMO explicitly allows algorithmic trading and EAs (official FAQ, checked 2026-10-05): "
            "'no reasons for limiting or restricting your trading strategy, whether it's discretionary "
            "trading, algorithmic trading, EAs, etc.' -- "
            "https://ftmo.com/en/faq/which-instruments-can-i-trade-and-what-strategies-am-i-allowed-to-use/ "
            "Conditions: strategy must be legitimate and replicable under real market conditions, no "
            "forbidden practices; platform limits of 200 open orders / 2,000 positions per day; a "
            "third-party EA used by many traders can hit the $400k capital-allocation cap."
        ),
    ),
    PropFirmPreset(
        key="ftmo_100k", firm="FTMO", label="FTMO - $100k Challenge",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        max_inactive_days=30,
        as_of=_AS_OF_V6,
        source_note=(
            "Confirmed 2026-09-13 against ftmo.com/en/trading-objectives/ -- see the $10k preset's note "
            "for the 2-Step-vs-1-Step and phase-modeling caveats, which apply identically here."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "FTMO explicitly allows algorithmic trading and EAs -- same official FAQ as the $10k preset "
            "(checked 2026-10-05): "
            "https://ftmo.com/en/faq/which-instruments-can-i-trade-and-what-strategies-am-i-allowed-to-use/"
        ),
    ),
    PropFirmPreset(
        key="ftmo_200k", firm="FTMO", label="FTMO - $200k Challenge",
        account_size=200_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        max_inactive_days=30,
        as_of=_AS_OF_V6,
        source_note=(
            "Confirmed 2026-09-13 against ftmo.com/en/trading-objectives/ -- see the $10k preset's note "
            "for the 2-Step-vs-1-Step and phase-modeling caveats, which apply identically here."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "FTMO explicitly allows algorithmic trading and EAs -- same official FAQ as the $10k preset "
            "(checked 2026-10-05): "
            "https://ftmo.com/en/faq/which-instruments-can-i-trade-and-what-strategies-am-i-allowed-to-use/"
        ),
    ),
    # --- Apex Trader Funding -------------------------------------------
    # v6 (Oct 2026) SPLIT: Apex sells EOD and Intraday account types with
    # genuinely different drawdown rules, so the old single "Apex 50k/100k"
    # entries (which hedged with a 6% midpoint) are replaced by four
    # entries. Verified Oct 2026 against apextraderfunding.com's own
    # EOD/Intraday Performance Account payout pages:
    #   - 50K: $2,000 trailing max drawdown = 4.0%; 100K: $3,000 = 3.0%.
    #   - EOD type: daily loss limit $1,000 (50K = 2.0%) / $1,500
    #     (100K = 1.5%), drawdown monitored at EOD
    #     (drawdown_check_mode="eod").
    #   - Intraday type: NO daily loss limit during evaluation
    #     (100.0 = the presets' "no DLL" convention), drawdown monitored
    #     intraday (drawdown_check_mode="intrabar").
    # Payout gates (both types): at least 5 qualifying trading days before
    # the first payout request; min_winning_day_profit 250.0 (EOD) /
    # 200.0 (Intraday). funded_consistency_rule_pct=50.0 models Apex's
    # real 50% payout-stage consistency rule (see module docstring --
    # there is NO evaluation-stage consistency rule post Apex 4.0, so
    # consistency_rule_pct stays None). min_trading_days stays 0 (Apex
    # 4.0, March 2026: no minimum -- a same-day pass is valid).
    PropFirmPreset(
        key="apex_50k_eod", firm="Apex Trader Funding", label="Apex - $50k Evaluation (EOD)",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=2.0,  # $1,000 EOD daily loss limit
        max_drawdown_pct=4.0,      # $2,000 trailing max drawdown
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=0,
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=250.0,
        funded_consistency_rule_pct=50.0,
        as_of=_AS_OF_V6,
        source_note=(
            "Updated 2026-10-04: EOD account type. $2,000 trailing max drawdown = 4.0% of $50k; "
            "$1,000 daily loss limit = 2.0%; drawdown monitored at EOD. Payout gates: 5 qualifying "
            "trading days, $250+/day winning-day gate, 50% funded consistency rule -- all modeled. "
            "A size-specific 'Safety Net' profit buffer also gates the first payout in reality and is "
            "not modeled. Verify exact dollar amounts for your chosen platform/product at "
            "apextraderfunding.com before relying on them. "
            "2026-10-13 partial check: apextraderfunding.com/help-center/additional-helpful-items/50-consistency-requirement/ "
            "confirms the 50% rule applies at PAYOUT request (best day / net profit since last payout or inception; "
            "failing it only blocks the request, the account stays alive). The page's text says 'below 50%' but its own "
            "table marks exactly 50% as met; the simulator uses <=. Target, drawdown, daily-loss and payout-gate "
            "numbers were NOT re-verified against an Apex page (fetch of the other pages was blocked)."
        ),
        algo_trading_allowed=False,
        algo_policy_note=(
            "Apex BANS automated trading -- official prohibited-activities page (checked 2026-10-05): "
            "'No Automation or Algorithm Usage allowed: Rewards are intended to recognize human traders "
            "actively participating in the learning process, not to reward automated systems executing "
            "preprogrammed logic.' -- "
            "https://apextraderfunding.com/help-center/getting-started/prohibited-activities/ "
            "Limited automation aids (ATM/bracket orders) are tolerated only with the trader in full "
            "manual control of entries and exits."
        ),
    ),
    PropFirmPreset(
        key="apex_50k_intraday", firm="Apex Trader Funding", label="Apex - $50k Evaluation (Intraday)",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,  # no daily loss limit on the Intraday type -- see source_note
        max_drawdown_pct=4.0,        # $2,000 trailing max drawdown
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=0,
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=200.0,
        funded_consistency_rule_pct=50.0,
        as_of=_AS_OF_V6,
        source_note=(
            "Updated 2026-10-04: Intraday account type. $2,000 trailing max drawdown = 4.0% of $50k; "
            "no daily loss limit (100.0 = the presets' 'no DLL' convention); drawdown monitored "
            "intraday. Payout gates: 5 qualifying trading days, $200+/day winning-day gate, 50% funded "
            "consistency rule -- all modeled. The 'Safety Net' profit buffer caveat from the EOD "
            "entry applies here too."
        ),
        algo_trading_allowed=False,
        algo_policy_note=(
            "Apex BANS automated trading -- official prohibited-activities page (checked 2026-10-05): "
            "'No Automation or Algorithm Usage allowed.' -- "
            "https://apextraderfunding.com/help-center/getting-started/prohibited-activities/ "
            "See the $50k EOD preset's note for the full quote."
        ),
    ),
    PropFirmPreset(
        key="apex_100k_eod", firm="Apex Trader Funding", label="Apex - $100k Evaluation (EOD)",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=1.5,  # $1,500 EOD daily loss limit
        max_drawdown_pct=3.0,      # $3,000 trailing max drawdown
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=0,
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=250.0,
        funded_consistency_rule_pct=50.0,
        as_of=_AS_OF_V6,
        source_note=(
            "Updated 2026-10-04: EOD account type. $3,000 trailing max drawdown = 3.0% of $100k; "
            "$1,500 daily loss limit = 1.5%; drawdown monitored at EOD. Payout gates and caveats as "
            "in the $50k EOD entry."
        ),
        algo_trading_allowed=False,
        algo_policy_note=(
            "Apex BANS automated trading -- official prohibited-activities page (checked 2026-10-05): "
            "'No Automation or Algorithm Usage allowed.' -- "
            "https://apextraderfunding.com/help-center/getting-started/prohibited-activities/ "
            "See the $50k EOD preset's note for the full quote."
        ),
    ),
    PropFirmPreset(
        key="apex_100k_intraday", firm="Apex Trader Funding", label="Apex - $100k Evaluation (Intraday)",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,  # no daily loss limit on the Intraday type
        max_drawdown_pct=3.0,        # $3,000 trailing max drawdown
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=0,
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=200.0,
        funded_consistency_rule_pct=50.0,
        as_of=_AS_OF_V6,
        source_note=(
            "Updated 2026-10-04: Intraday account type. $3,000 trailing max drawdown = 3.0% of $100k; "
            "no daily loss limit; drawdown monitored intraday. Payout gates and caveats as in the "
            "$50k Intraday entry."
        ),
        algo_trading_allowed=False,
        algo_policy_note=(
            "Apex BANS automated trading -- official prohibited-activities page (checked 2026-10-05): "
            "'No Automation or Algorithm Usage allowed.' -- "
            "https://apextraderfunding.com/help-center/getting-started/prohibited-activities/ "
            "See the $50k EOD preset's note for the full quote."
        ),
    ),
    # --- TopStep --------------------------------------------------------
    PropFirmPreset(
        key="topstep_50k", firm="TopStep", label="TopStep - $50k Trading Combine",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,  # TopStepX no longer enforces a default DLL on the Combine -- see note
        max_drawdown_pct=4.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        trailing_lock=True, trailing_lock_offset_pct=0.0,   # 2026-10 audit: MLL stops trailing at the starting balance
        floating_drawdown_mode="adverse",
        trailing_distance_basis="account", funded_starts_fresh=True,   # 2026-10-12 audit: fixed-$ MLL; XFA starts fresh at $0 (help.topstep.com/8284204)
        consistency_rule_pct=55.0, min_trading_days=2,       # 2026-10 audit: Combine best-day cap 50-55%; 55 = Topstep's stated figure
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=150.0,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF,
        source_note=(
            "2026-10-11 audit: MLL is an END-OF-DAY trailing floor that locks at the starting balance "
            "(trailing_lock=True), with open-loss checked adversely (floating_drawdown_mode='adverse'); "
            "Combine consistency target 55%; contract cap 5 minis / 50 micros -- "
            "https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit . "
            "Earlier note: Re-verified 2026-09-13 against help.topstep.com. min_trading_days=2 and max_drawdown_pct=4.0 "
            "were already accurate (TopStep's own Help Center: 'you can pass in as few as two days'; $2,000 "
            "MLL on a $50k Combine = 4%). consistency_rule_pct is now set (was None): TopStep's Combine has "
            "a real 'Consistency Target' capping the best single day at roughly 50-55% of the PROFIT "
            "TARGET -- modeled here as 50% of total profit since that's the closest fit to this simulator's "
            "field, though the real rule's denominator (a fixed target, not accumulated profit) differs "
            "slightly. payout_frequency_days lowered from 14 to 5, reflecting the Standard funded path's "
            "'5 winning days of $150+' payout-eligibility rule (help.topstep.com) -- note this counts "
            "WINNING days specifically, not just trading days, so it can take longer in practice than a "
            "literal 5-day clock. Daily loss limit is genuinely platform-dependent since 2024: not enforced "
            "by default on TopstepX, but still enforced on NinjaTrader/Tradovate/Quantower/TradingView -- "
            "the non-binding 100% here matches TopstepX; tighten it if you trade one of the other platforms."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "Topstep's official help center: 'Automated strategies are permitted' on the Trading Combine "
            "and Express Funded Account (checked 2026-10-05; see TradersPost's Topstep review quoting the "
            "official stance). Caveats straight from Topstep: the trader is fully responsible for the "
            "bot's errors (Topstep won't help set up or troubleshoot automation), automated trading via "
            "the ProjectX API is PROHIBITED on the Live Funded Account, and 'improper use of "
            "automation' (manipulative/abusive algos) is a prohibited strategy."
        ),
    ),
    PropFirmPreset(
        key="topstep_100k", firm="TopStep", label="TopStep - $100k Trading Combine",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,
        max_drawdown_pct=3.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        trailing_lock=True, trailing_lock_offset_pct=0.0,     # 2026-10-12 audit: was missing -- 100k MLL ($3,000) also locks at the starting balance, like the 50k
        floating_drawdown_mode="adverse",
        trailing_distance_basis="account", funded_starts_fresh=True,
        consistency_rule_pct=50.0, min_trading_days=2,
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=150.0,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF,
        source_note="Re-verified 2026-09-13 against help.topstep.com -- see the $50k preset's note for the full explanation.",
        algo_trading_allowed=True,
        algo_policy_note=(
            "Topstep's official help center: 'Automated strategies are permitted' on the Trading Combine "
            "and Express Funded Account (checked 2026-10-05) -- same as the $50k preset; trader bears "
            "full responsibility for bot errors, ProjectX API automation banned on the Live Funded Account."
        ),
    ),
    # --- The5%ers ---------------------------------------------------------
    # v6 (Oct 2026) RELABEL (B9): these entries were called "Bootcamp" but their
    # numbers (8% target / 5% daily / 10% static max loss / 30% consistency /
    # 3 min days) match The5%ers' High-Stakes 2-phase program, not Bootcamp.
    # Bootcamp's real numbers are roughly 5% target / 4% daily / 3% max loss
    # and are NOT modeled here -- don't pick these presets for a Bootcamp
    # account.
    PropFirmPreset(
        key="the5ers_20k", firm="The5%ers", label="The5%ers - $20k High-Stakes",
        account_size=20_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=30.0, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note=(
            "Re-verified 2026-09-13. min_trading_days=3 and payout_frequency_days=14 were already accurate "
            "(multiple 2026 sources confirm 'minimum 3 profitable trading days' and bi-weekly payouts from "
            "day one). consistency_rule_pct is now set to 30.0 (was None) -- The5%ers runs an active 30% "
            "consistency rule across BOTH evaluation and funded stages, described as per-position rather "
            "than strictly per-day. The5%ers currently sells several concurrently-named programs (Bootcamp, "
            "Hyper Growth, High-Stakes, Pro Growth) with different profit-target/drawdown combinations -- "
            "the 8%/5%/10%/static numbers here match the High-Stakes-style 2-phase structure most closely; "
            "confirm which program name your account actually uses at the5ers.com. NOTE (v6, Oct 2026): "
            "these entries are labeled High-Stakes (their numbers match that program); The5%ers Bootcamp's "
            "real numbers are roughly 5% target / 4% daily / 3% max loss and are NOT modeled here."
        ),
        algo_trading_allowed=False,
        algo_policy_note=(
            "The5ers' FAQ permits own-code EAs (trader must own the source code; no tick scalping, "
            "latency/hedge/reverse arbitrage, HFT, emulators, or stealth stop-losses), BUT the firm's "
            "Terms require prior WRITTEN APPROVAL before using any automated trading software -- "
            "approval-gated, so this is False until that approval is obtained (checked 2026-10-05). "
            "Do not assume an EA that works on a retail account complies with The5ers."
        ),
    ),
    PropFirmPreset(
        key="the5ers_100k", firm="The5%ers", label="The5%ers - $100k High-Stakes",
        account_size=100_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=30.0, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note="Re-verified 2026-09-13 -- see the $20k preset's note for the full explanation and program-naming caveat.",
        algo_trading_allowed=False,
        algo_policy_note=(
            "The5ers' FAQ permits own-code EAs, BUT the Terms require prior WRITTEN APPROVAL before using "
            "any automated trading software -- approval-gated, False until that approval is obtained "
            "(checked 2026-10-05). See the $20k preset's note."
        ),
    ),
    # --- FundedNext ---------------------------------------------------------
    # v6 (Oct 2026) REMAP (B7): these entries now model FundedNext's
    # Stellar 2-Step product specifically (the product these numbers
    # correspond to), not the generic "Evaluation" product. Phase-1 profit
    # target corrected 10.0 -> 8.0. payout_frequency_days=14 models the
    # RECURRING cadence; FundedNext's actual Stellar terms are a 21-day
    # wait for the FIRST payout and 14 days thereafter -- this simulator
    # has no separate "first payout frequency" field, so the 21-day first-
    # payout caveat is documented here instead of modeled (expect the
    # sim's first payout ~7 trade-days earlier than reality).
    PropFirmPreset(
        key="fundednext_25k", firm="FundedNext", label="FundedNext - $25k Stellar 2-Step",
        account_size=25_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=40.0, min_trading_days=5,
        payout_frequency_days=14,
        winning_days_for_payout=5, min_winning_day_profit=100.0,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF_V6,
        source_note=(
            "Updated 2026-10-04: now models the Stellar 2-Step product (Phase-1 target 8.0, corrected "
            "from 10.0). consistency_rule_pct=40.0 (eval) and funded_consistency_rule_pct=40.0 (funded "
            "payouts) both set -- confirmed rule: 'no single day can account for more than 40% of your "
            "total profit' (one source states 30% instead; worth a direct check). min_trading_days=5. "
            "Payout cadence: payout_frequency_days=14 models the RECURRING 14-day cycle; Stellar's actual "
            "terms require 21 days before the FIRST payout, which this field cannot express -- expect the "
            "sim's first payout roughly 7 trade-days earlier than reality. Payout gate: 5 winning days of "
            "$100+."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "FundedNext's official help center (checked 2026-10-05): EAs are welcome on MT4/MT5 accounts "
            "BELOW $50k ('Feel free to trade with Expert Advisors (EAs)' -- no restrictions; a paid "
            "'add-on' buys permission to use ready-made/third-party EAs). $50k and above: traders must "
            "trade fully manually. This $25k preset is under the threshold, so True. "
            "https://help.fundednext.com/en/articles/8020763-is-ea-allowed-in-fundednext"
        ),
    ),
    PropFirmPreset(
        key="fundednext_100k", firm="FundedNext", label="FundedNext - $100k Stellar 2-Step",
        account_size=100_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=40.0, min_trading_days=5,
        payout_frequency_days=14,
        winning_days_for_payout=5, min_winning_day_profit=200.0,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF_V6,
        source_note=(
            "Updated 2026-10-04 -- see the $25k Stellar 2-Step preset's note (8.0 Phase-1 target, 21-day "
            "first-payout caveat, 40% consistency at both stages). Payout gate here: 5 winning days of "
            "$200+."
        ),
        algo_trading_allowed=False,
        algo_policy_note=(
            "FundedNext's official help center (checked 2026-10-05): EAs are welcome on accounts below "
            "$50k, but $50k and above must be traded fully manually -- no bots, EAs, or automation of "
            "any kind. This $100k preset is over the threshold, so False. "
            "https://help.fundednext.com/en/articles/8020763-is-ea-allowed-in-fundednext"
        ),
    ),
    # --- Lucid Trading ---------------------------------------------------
    # MAJOR CORRECTION: this preset was significantly stale. Verified
    # 2026-09-13 directly against lucidtrading.com's live pricing page
    # (LucidPro tab, 50K Pro Funded) plus independent 2026 review sources,
    # all of which agree closely. The previous numbers (8% target, 4%
    # daily loss, 8% max drawdown, 3 min trading days, 14-day payout) were
    # wrong on every field except drawdown_type/drawdown_check_mode.
    PropFirmPreset(
        key="lucid_50k", firm="Lucid Trading", label="Lucid Trading - $50k Evaluation",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=2.4, max_drawdown_pct=4.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        # Checked 2026-10-07 against tradetanto.com's LucidPro 50K table (secondary source; confirm on
        # lucidtrading.com): EOD trailing MLL $2,000 that LOCKS at $50,100 once balance exceeds
        # $52,100; $1,200 daily loss is a soft breach (day locked, account kept); 4 mini / 40 micro.
        dd_basis="eod", daily_loss_basis="floating", daily_loss_action="lock_day",
        trailing_lock=True, trailing_lock_offset_pct=0.2, max_contracts=4, rules_checked_on="2026-10-07",
        trailing_distance_basis="account", flatten_time_ct="15:45",
        consistency_rule_pct=None, min_trading_days=1,
        payout_frequency_days=3,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF,
        source_note=(
            "CORRECTED 2026-09-13 against lucidtrading.com (LucidPro track) -- previous values were stale. "
            "Confirmed: $3,000 profit target on $50k = 6% (was modeled as 8%); $1,200 daily loss limit = "
            "2.4% (was 4%); $2,000 EOD trailing max loss = 4% (was 8%); minimum trading days = 1, not 3 "
            "('pass in as little as one day' is the current LucidPro rule); payout cycle = 3 trading days, "
            "not 14 -- confirmed both by Lucid's own live pricing page ('Days to Payout: 3') and independent "
            "sources. consistency_rule_pct stays None: LucidPro has NO consistency rule during evaluation -- "
            "the real 40% consistency rule shown on Lucid's site applies only at the funded-payout stage "
            "(now modeled by funded_consistency_rule_pct=40.0 as of v6; see module docstring's MODELING "
            "LIMITATION note), same situation as Apex above. Lucid also "
            "sells LucidFlex (50% eval consistency, no funded consistency, no DLL), LucidDaily, and "
            "LucidDirect (skips the evaluation entirely, 20% funded consistency) -- this preset models "
            "LucidPro specifically, the track shown in the screenshot this correction was verified against."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "Lucid's official help center 'Permitted Activities' (checked 2026-10-05): automated systems "
            "are permitted -- 'We allow traders to use automated trading systems as long as they do not "
            "exploit the platform.' PROHIBITED even with automation: high-frequency trading (HFT) "
            "strategies, latency arbitrage, and reverse/hedge arbitrage. "
            "https://intercom.help/lucid-trading/en/articles/11321405-prohibited-trading-strategies "
            "(also covered in TradersPost's Lucid review)."
        ),
    ),
    PropFirmPreset(
        key="lucid_100k", firm="Lucid Trading", label="Lucid Trading - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=2.4, max_drawdown_pct=3.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=1,
        payout_frequency_days=3,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF,
        source_note=(
            "CORRECTED 2026-09-13 -- see the $50k preset's note for the full explanation. max_drawdown_pct "
            "here is 3% (not 4%) because Lucid's max-loss DOLLAR amounts don't scale as a flat percentage "
            "across sizes -- confirmed $3,000 max loss on the $100k tier = 3%. evaluation_profit_target_pct "
            "and daily_loss_limit_pct are carried over from the $50k tier's percentage (Lucid markets these "
            "as 'the same profit targets' across sizes) but the exact $100k dollar figures weren't directly "
            "confirmed in this pass -- worth a direct check at lucidtrading.com before relying on them."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "Lucid's official help center 'Permitted Activities' (checked 2026-10-05): automated systems "
            "are permitted as long as they do not exploit the platform; HFT, latency arbitrage, and "
            "reverse/hedge arbitrage prohibited. Same as the $50k preset. "
            "https://intercom.help/lucid-trading/en/articles/11321405-prohibited-trading-strategies"
        ),
    ),
    # --- E8 Markets -----------------------------------------------------
    # v7 (Oct 2026): E8 explicitly permits algo trading on the E8 One
    # 1-Step program -- their official help center lists "Algo, EA, Bots
    # and Indicators" among the tools traders can use (checked 2026-10-05:
    # help.e8markets.com/en/articles/5515409). Numbers for the $100k E8
    # One (1-Step) plan: 10% profit target (Phase 1 only), 4% daily loss,
    # 6% STATIC max loss, 10 minimum trading days, no fixed calendar
    # deadline (the 14-day payout field models the standard post-payout
    # review cadence; confirm current terms at e8markets.com before
    # relying on them -- prop firms adjust these plans).
    PropFirmPreset(
        key="e8_one_100k", firm="E8 Markets", label="E8 Markets - $100k E8 One (1-Step)",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=4.0, max_drawdown_pct=6.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=10,
        payout_frequency_days=14,
        as_of=_AS_OF_V7,
        source_note=(
            "Added 2026-10-05. Models the E8 One (1-Step) $100k plan: 10% target, 4% daily loss, 6% "
            "static max loss, 10 minimum trading days. Verified against E8's official help center "
            "(https://help.e8markets.com/en/articles/5515409). E8 also sells the E8 Account (2-Step) product; "
            "these numbers are NOT the 2-Step -- confirm which plan your account uses at e8markets.com."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "E8 explicitly allows algo trading on E8 One: official help center (checked 2026-10-05) "
            "lists 'Algo, EA, Bots and Indicators' among the tools traders may use. -- "
            "https://help.e8markets.com/en/articles/5515409"
        ),
    ),
    # --- FunderPro ------------------------------------------------------
    # v7 (Oct 2026): FunderPro's One Phase program explicitly allows
    # EAs -- their official help center says EAs are permitted provided
    # the trader OWNS the EA (checked 2026-10-05). Numbers for the $100k
    # One Phase plan: 10% profit target, 3% balance-based daily loss,
    # 6% static max loss. Confirm current terms at funderpro.com.
    PropFirmPreset(
        key="funderpro_one_phase_100k", firm="FunderPro", label="FunderPro - $100k One Phase",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=3.0, max_drawdown_pct=6.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=5,
        payout_frequency_days=14,
        as_of=_AS_OF_V7,
        source_note=(
            "Added 2026-10-05. Models FunderPro's One Phase $100k plan: 10% target, 3% daily loss "
            "(balance-based), 6% static max loss, 5 minimum trading days. Verified against FunderPro's "
            "official help center (https://www.funderpro.com). FunderPro also sells a 2-Step product -- these "
            "numbers are the One Phase plan; re-confirm terms at funderpro.com before relying on them."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "FunderPro explicitly allows EAs on One Phase -- official help center (checked 2026-10-05): "
            "EAs are permitted provided the trader OWNS the EA they trade with. Shared/third-party EAs "
            "not owned by the trader are not covered by that permission."
        ),
    ),
    # --- Atlas Funded ---------------------------------------------------
    # v7 (Oct 2026): Atlas Funded's 1-Step program explicitly allows EAs --
    # their official help center permits EAs with the standard forbidden-
    # practices carve-outs (checked 2026-10-05:
    # atlasfunded-helpcenter.atlassian.net articles 9904215 and 9903316).
    # Numbers for the $100k 1-Step plan: 10% profit target, 5% daily loss,
    # 10% STATIC max drawdown, 5 minimum trading days. Confirm current
    # terms at atlasfunded.com.
    PropFirmPreset(
        key="atlas_funded_1step_100k", firm="Atlas Funded", label="Atlas Funded - $100k 1-Step",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=5,
        payout_frequency_days=14,
        as_of=_AS_OF_V7,
        source_note=(
            "Added 2026-10-05. Models Atlas Funded's 1-Step $100k plan: 10% target, 5% daily loss, "
            "10% static max drawdown, 5 minimum trading days. Verified against Atlas Funded's official "
            "help center (checked 2026-10-05; https://atlasfunded-helpcenter.atlassian.net -- articles "
            "9904215 and 9903316 cover EA permission). Atlas also sells a 2-Step product -- these are "
            "the 1-Step numbers; confirm at atlasfunded.com."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "Atlas Funded explicitly allows EAs on the 1-Step program -- official help center "
            "(checked 2026-10-05): EAs permitted subject to the standard forbidden-practices rules "
            "(no HFT/latency/martingale exploits). -- "
            "https://atlasfunded-helpcenter.atlassian.net (articles 9904215, 9903316)"
        ),
    ),
    # --- The Futures Desk -----------------------------------------------
    # v7 (Oct 2026): The Futures Desk is an algo-first futures prop shop --
    # their own homepage leads with "Microscalping + Algos Allowed" and
    # they publish free algorithmic trading tools (checked 2026-10-05:
    # thefuturesdesk.com). Numbers model a $50k account on their standard
    # track: $4,000 profit target (8%), $2,000 max drawdown (trailing),
    # $800 intraday daily loss limit, 20% eval consistency, daily payouts.
    # Confirm current terms at thefuturesdesk.com -- a young firm whose
    # plans change.
    PropFirmPreset(
        key="futures_desk_50k", firm="The Futures Desk", label="The Futures Desk - $50k Standard",
        account_size=50_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=1.6, max_drawdown_pct=4.0,
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=20.0, min_trading_days=1,
        payout_frequency_days=1,
        as_of=_AS_OF_V7,
        source_note=(
            "Added 2026-10-05. Models The Futures Desk's $50k standard track: $4,000 target (8%), "
            "$2,000 trailing max drawdown (4%), $800 intraday daily loss limit (1.6%), 20% eval "
            "consistency, daily payouts. Verified against the firm's own homepage "
            "(https://thefuturesdesk.com, 'Microscalping + Algos Allowed'). Young firm -- re-verify "
            "terms before relying on them."
        ),
        algo_trading_allowed=True,
        algo_policy_note=(
            "The Futures Desk is algo-first by design -- the firm's own homepage (checked 2026-10-05) "
            "leads with 'Microscalping + Algos Allowed' and they publish free algorithmic trading tools. "
            "https://thefuturesdesk.com"
        ),
    ),
]

# v9.5 (accuracy activation): make the account-rule BASIS explicit on
# every preset. Before this, PropFirmPreset.dd_basis defaulted to
# "legacy" and 18 of 19 presets silently inherited it, so the unified
# PropAccount judged most firms under a basis nobody had chosen -- and
# the basis alone moved bust counts by ~40% in the Oct 2026 audit's
# controlled test. The basis below is derived from each preset's OWN
# already-declared drawdown_check_mode (the firm-facing statement of
# when the firm checks the floor, recorded when the preset was built):
# "intrabar" -> the firm watches floating equity inside the day, so the
# basis is "floating"; "eod" -> "eod". daily_loss_basis follows the same
# derivation; daily_loss_action stays "fail" except where a preset sets
# it (Lucid). trailing_lock stays as declared. This makes behavior
# explicit and testable; it does NOT re-verify any firm against its
# current rules page (the source_note says so on every touched preset),
# except lucid_50k, which was verified 2026-10-07 and is left untouched.
for _p in PROP_FIRM_PRESETS:
    if _p.dd_basis == "legacy":
        _basis = "floating" if _p.drawdown_check_mode == "intrabar" else "eod"
        _note = (
            "v9.5: dd/daily-loss basis made explicit from this preset's declared "
            "drawdown_check_mode; NOT re-verified against the firm's current rules "
            "page -- re-check before relying on pass/fail for this firm."
        )
        # PropFirmPreset is frozen; object.__setattr__ is the sanctioned
        # way to finalize a catalog entry at build time (same pattern as
        # any post-construction defaulting on a frozen dataclass).
        object.__setattr__(_p, "dd_basis", _basis)
        object.__setattr__(_p, "daily_loss_basis", "floating" if _basis == "floating" else "realized")
        object.__setattr__(_p, "source_note", ((_p.source_note + " ") if _p.source_note else "") + _note)
del _p

# ---------------------------------------------------------------------------
# 2026-10-17 firm-rule verification pass (audit item 7). Every row below was compared against the firm's OWN
# help centre / product page on 2026-10-17. ``fixes`` lists the fields that DISAGREED with the firm's page and
# were corrected (old -> new is recorded in the note so the change is auditable); fields not listed matched or
# were not stated on the page. Nothing here was taken from a reviewer or comparison site.
# ---------------------------------------------------------------------------
_RULE_CORRECTIONS: dict[str, tuple[dict, str]] = {
    "apex_50k_eod": ({"daily_loss_action": "lock_day", "max_contracts": 6, "max_eval_calendar_days": 30},
        "Apex help centre (EOD Evaluations, EOD Drawdown Explained, EOD Payouts, read 2026-10-17): target $3,000, EOD threshold $2,000, DLL $1,000 is a SOFT limit "
        "(pauses the session, does not fail the evaluation; was 'fail'), 6 contracts, 30 calendar days to pass, no minimum days, funded consistency 50%, "
        "payout needs 5 winning days >= $250. In evaluations the EOD threshold locks at the profit-target balance on Rithmic/Wealthcharts but never locks on "
        "Tradovate -- this preset models the never-locks (Tradovate) behaviour, the conservative one. APEX PROHIBITS AUTOMATION (help-centre 'Prohibited Activities': "
        "'No Automation or Algorithm Usage allowed')."),
    "apex_50k_intraday": ({"max_contracts": 6, "max_eval_calendar_days": 30},
        "Apex help centre (Intraday Trailing Drawdown Evaluations / Explained / Payouts, read 2026-10-17): target $3,000, trailing threshold $2,000 following the peak "
        "INCLUDING unrealized gains, no daily loss limit, 6 contracts, 30 days, no minimum days, payout needs 5 winning days >= $200. Evaluation threshold locks at the "
        "profit-target balance (Rithmic/Wealthcharts) or never (Tradovate); modelled as never. APEX PROHIBITS AUTOMATION."),
    "apex_100k_eod": ({"daily_loss_action": "lock_day", "max_contracts": 8, "max_eval_calendar_days": 30, "min_winning_day_profit": 300.0},
        "Apex help centre (read 2026-10-17): target $6,000, EOD threshold $3,000, DLL $1,500 SOFT (was 'fail'), 8 contracts, 30 days, payout needs 5 winning days >= $300 "
        "(was $250). Same lock behaviour note as the 50K EOD row. APEX PROHIBITS AUTOMATION."),
    "apex_100k_intraday": ({"max_contracts": 8, "max_eval_calendar_days": 30, "min_winning_day_profit": 250.0},
        "Apex help centre (read 2026-10-17): target $6,000, trailing threshold $3,000 incl. unrealized, no DLL, 8 contracts, 30 days, payout needs 5 winning days >= $250 "
        "(was $200). APEX PROHIBITS AUTOMATION."),
    "lucid_50k": ({},
        "Lucid help centre (support.lucidtrading.com: LucidPro Evaluation Account, Drawdown, Daily Loss Limit, Consistency, Payouts; read 2026-10-17): target $3,000, "
        "EOD trailing MLL $2,000 locking at $50,100 once the balance passes $52,100 (the $100 lock offset), DLL $1,200 is a SOFT breach (locks the session, matches "
        "'lock_day'), 4 mini / 40 micro, consistency 40% FUNDED stage only, 90% split, $500 minimum payout. The page does not say whether the DLL counts floating loss; "
        "'floating' is kept as the conservative reading. Prices are NOT on the help centre (see FIRM_FEES)."),
    "lucid_100k": ({"daily_loss_limit_pct": 1.8, "daily_loss_action": "lock_day", "daily_loss_basis": "floating", "trailing_lock": True,
                    "trailing_lock_offset_pct": 0.1, "trailing_distance_basis": "account", "max_contracts": 6, "flatten_time_ct": "15:45"},
        "Lucid help centre (read 2026-10-17): $100K LucidPro target $6,000, EOD trailing MLL $3,000 locking at $100,100 once the balance passes $103,100, DLL $1,800 "
        "= 1.8% (was 2.4%) and a SOFT breach (was 'fail'), 6 mini / 60 micro. Previously this row had no MLL lock and a hard daily-loss failure, both wrong."),
    "topstep_50k": ({"max_contracts": 5},
        "Topstep help centre (read 2026-10-17): MLL $2,000 EOD-updated trailing, monitored in real time, locks at the starting balance; Combine consistency 55% (best day / "
        "total profit), 5 minis / 50 micros, no minimum days ('as few as two'); XFA payout: 5 winning days >= $150 or the 40% consistency path, 90/10 split, $125 minimum. "
        "DLL is OPTIONAL ($1,000 at 50K) and a soft lock -- not modelled (none). The $3,000 profit target is the only figure not on a Topstep page the tools could read "
        "(futureshive.com, dated 2026-10-05, agrees with the 6% already in the preset)."),
    "topstep_100k": ({"consistency_rule_pct": 55.0, "max_contracts": 10},
        "Topstep help centre (read 2026-10-17): Combine consistency is 55% for every size (was 50% here), MLL $3,000, 10 minis / 100 micros, DLL optional ($2,000). "
        "The $6,000 profit target is confirmed only by futureshive.com (2026-10-05), not by a Topstep page the tools could read."),
    "ftmo_10k": ({},
        "FTMO trading-objectives page (read 2026-10-17), 2-Step Challenge: Phase 1 target 10% (Phase 2 5%), max daily loss 5%, max loss 10% STATIC, minimum 4 trading days -- "
        "all match. The page gives these as percentages for every size. NOTE the 1-Step Challenge differs (3% daily, 10% trailing, 50% best-day consistency from 2026-10-12): "
        "this preset models the 2-Step only. Challenge fees are not readable (login-gated)."),
    "ftmo_100k": ({}, "Same FTMO 2-Step page as ftmo_10k (read 2026-10-17); rules are percentages of the account size. Fees login-gated."),
    "ftmo_200k": ({}, "Same FTMO 2-Step page as ftmo_10k (read 2026-10-17); rules are percentages of the account size. Fees login-gated."),
    "fundednext_25k": ({"consistency_rule_pct": None, "drawdown_check_mode": "intrabar", "dd_basis": "floating", "daily_loss_basis": "floating"},
        "FundedNext help centre + product page (read 2026-10-17): Stellar 2-Step Phase 1 8%, Phase 2 5%, DLL 5% of the initial balance, max loss 10% ('balance or equity must "
        "not drop below 90% ... at any point' => measured on equity, so intrabar/floating; was end-of-day/realized), 5 trading days per phase, no time limit, NO "
        "consistency rule during the evaluation (the 40% best-day rule belongs to the funded 'On-Demand' reward; was applied here to the evaluation). EAs are allowed under "
        "$50k only. Whether the DLL counts floating loss is not stated -- floating is kept as the conservative reading."),
    "fundednext_100k": ({"consistency_rule_pct": None, "drawdown_check_mode": "intrabar", "dd_basis": "floating", "daily_loss_basis": "floating"},
        "Same FundedNext sources (read 2026-10-17). $50k and larger Stellar accounts must be traded MANUALLY (EAs and the EA add-ons are unavailable), so automation is not "
        "permitted on this preset."),
    "the5ers_20k": ({"evaluation_profit_target_pct": 10.0, "daily_loss_limit_pct": 3.0, "consistency_rule_pct": None},
        "The5%ers site (the5ers.com/high-stakes and /faqs/2-step-plan-rules-specifications, read 2026-10-17): Step 1 target 10% (an older 'Classic' plan used 8%), Step 2 5%, "
        "max loss 10% of the initial balance, 3 profitable days (each >= 0.5%), NO evaluation consistency rule (was 30% here; a 50% rule applies to the funded account). "
        "The two official pages DISAGREE on the daily loss (5% on the marketing page, 3% of the previous day's closing balance/equity on the FAQ); the stricter 3% is modelled. "
        "THE5ERS DOES NOT LIST A $20k SIZE (published: 2.5k, 5k, 10k, 25k, 50k, 100k) -- this row is a proxy at the nearest size. Automation needs the firm's prior written approval."),
    "the5ers_100k": ({"evaluation_profit_target_pct": 10.0, "daily_loss_limit_pct": 3.0, "consistency_rule_pct": None},
        "Same The5%ers sources (read 2026-10-17); see the5ers_20k for the daily-loss conflict. Left UNVERIFIED because two official pages disagree on a rule that decides pass/fail."),
    "e8_one_100k": ({"evaluation_profit_target_pct": 9.0, "drawdown_type": "trailing", "dd_basis": "realized", "trailing_lock": True,
                    "trailing_lock_offset_pct": 0.0, "trailing_distance_basis": "account", "min_trading_days": 0, "max_inactive_days": 60,
                    "drawdown_check_mode": "intrabar"},
        "E8 Markets help centre + product page (help.e8markets.com 'E8 One', e8markets.com/e8-one, read 2026-10-17). E8 One is a CONFIGURABLE challenge; this preset is the $100K "
        "configuration shown there: 6% drawdown, 80% split => profit target $9,000 (was 10%), daily drawdown $4,000 = 4% (hard breach), NO minimum trading days (was 10; one "
        "trade must be placed and closed every 60 days), drawdown is DYNAMIC TRAILING on the highest CLOSED balance and locks at the initial balance (was modelled as static). "
        "Ranges offered: drawdown 4-14%, daily 3-9.2%, target 6-21%. A different configuration is a different preset."),
    "funderpro_one_phase_100k": ({"min_trading_days": 0, "consistency_rule_pct": 40.0},
        "FunderPro 1-Phase page (funderpro.com/blog/1-phase-vs-2-phase..., read 2026-10-17): target 10%, daily drawdown 3% balance-based, overall drawdown 6%, NO minimum trading "
        "days (was 5), best day must be <= 40% of total profit (was none). The page does not say whether the 6% is static or trailing, and gives no $100K fee or EA policy."),
    "atlas_funded_1step_100k": ({"daily_loss_limit_pct": 4.0, "max_drawdown_pct": 7.0, "min_trading_days": 3},
        "Atlas Funded docs (help.atlasfunded.com/one-step, read 2026-10-17): target 10% of CLOSED balance, daily loss 4% static from equity at 00:00 UTC (was 5%), max drawdown 7% "
        "static (was 10%), 3 minimum trading days (was 5), no consistency rule, 90% split (100% add-on). The TradeLocker Hub listing for the same $100K product shows 11% / 4% / 7%, "
        "so the target is not settled between official sources -- left UNVERIFIED."),
    "futures_desk_50k": ({"drawdown_type": "static", "min_trading_days": 7, "consistency_rule_pct": None},
        "The Futures Desk members page (members.thefuturesdesk.com/static-plans-framer, read 2026-10-17) lists Static Mini/Core/Pro: Core = target $4,000, STATIC drawdown $2,000, "
        "DLL $800, 7 minimum days, reset $99, 80% split. Those match this 50K preset's 8% / 4% / 1.6% but NOT its trailing drawdown, 1 minimum day or 20% consistency, so those "
        "three were corrected to the Static Core figures. The page does not state the account size of each plan, so the mapping to $50k is inferred -- left UNVERIFIED."),
}
for _p in PROP_FIRM_PRESETS:
    _c = _RULE_CORRECTIONS.get(_p.key)
    if _c:
        for _f, _v in _c[0].items():
            object.__setattr__(_p, _f, _v)
        object.__setattr__(_p, "source_note", ((_p.source_note + " ") if _p.source_note else "") + "[2026-10-17 verification] " + _c[1])
del _p
try:
    del _c, _f, _v
except NameError:
    pass

# 2026-10 audit: verification stamps. Only rows whose own source_note records a direct check of the
# firm's page are stamped; everything else stays unverified (verified_on == "") and is therefore NOT
# READY-eligible until someone checks it and adds a row here.
# 2026-10-22 full re-check (every stamped row re-read against the firm's own page that day):
#   FTMO trading-objectives (10%/5%/10% static, 4 days), Topstep help centre (MLL $2k/$3k EOD-trailing and locking, target $3k/$6k,
#   55% consistency, funded 40%), Apex EOD + Intraday evaluation pages (targets, drawdown, DLL, contracts, 30-day limit),
#   Lucid help centre (target, max loss, daily loss, size), FundedNext Stellar 2-Step help article, E8 One product page.
# NOT re-stamped (no primary page reachable/consistent today): The5%ers (its own marketing page says 5% daily loss, its FAQ 3% -- the
#   stricter 3% is modelled and the row stays unverified), FunderPro (page confirms only the 3% daily drawdown), Atlas Funded, The Futures Desk.
_VERIFIED_STAMPS = {
    "ftmo_10k": ("2026-10-22", "https://ftmo.com/en/trading-objectives/", "primary"),
    "ftmo_100k": ("2026-10-22", "https://ftmo.com/en/trading-objectives/", "primary"),
    "ftmo_200k": ("2026-10-22", "https://ftmo.com/en/trading-objectives/", "primary"),
    # 2026-10-17: re-read against help.topstep.com (MLL, consistency, payout, DLL, parameters articles).
    "topstep_50k": ("2026-10-22", "https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit", "primary"),
    "topstep_100k": ("2026-10-22", "https://help.topstep.com/en/articles/8284204-what-is-the-maximum-loss-limit", "primary"),
    # 2026-10-17: re-read against the Apex help centre (apextraderfunding.com/help-center). Automation is PROHIBITED by Apex.
    "apex_50k_eod": ("2026-10-22", "https://apextraderfunding.com/help-center/eod-trailing-drawdown-accounts/eod-evaluations/", "primary"),
    "apex_50k_intraday": ("2026-10-22", "https://apextraderfunding.com/help-center/evaluation-accounts-ea/intraday-trailing-drawdown-evaluations/", "primary"),
    "apex_100k_eod": ("2026-10-22", "https://apextraderfunding.com/help-center/eod-trailing-drawdown-accounts/eod-evaluations/", "primary"),
    "apex_100k_intraday": ("2026-10-22", "https://apextraderfunding.com/help-center/evaluation-accounts-ea/intraday-trailing-drawdown-evaluations/", "primary"),
    # 2026-10-17: Lucid's own help centre (support.lucidtrading.com) -- replaces the secondary tradetanto stamp and the owner-attested one.
    "lucid_50k": ("2026-10-22", "https://support.lucidtrading.com/en/articles/12890029-lucidpro-evaluation-account", "primary"),
    "lucid_100k": ("2026-10-22", "https://support.lucidtrading.com/en/articles/12890029-lucidpro-evaluation-account", "primary"),
    # 2026-10-17: FundedNext help centre + product page; E8 help centre + product page.
    "fundednext_25k": ("2026-10-22", "https://help.fundednext.com/en/articles/17255124-stellar-2-step-challenge", "primary"),
    "fundednext_100k": ("2026-10-22", "https://help.fundednext.com/en/articles/17255124-stellar-2-step-challenge", "primary"),
    "e8_one_100k": ("2026-10-22", "https://e8markets.com/e8-one", "primary"),
}
# 2026-10-27 audit (fix 8): stamps expire after the READY age limit, so a preset the owner re-checked must be re-stampable without
# editing code. ``data/preset_stamps.json`` ({"key": {"date": "YYYY-MM-DD", "url": "...", "tier": "primary"}}) overrides the table
# above; ``python scripts/restamp_presets.py key1 key2 ... --date 2026-11-01`` writes it. Only stamp rows you actually re-read
# against the firm's own page: a stamp is a claim that the numbers were checked.
def _load_stamp_overrides() -> dict:
    try:
        import json as _json
        from pathlib import Path as _P
        fp = _P(__file__).resolve().parents[2] / "data" / "preset_stamps.json"
        if fp.exists():
            raw = _json.loads(fp.read_text(encoding="utf-8"))
            return {k: (v["date"], v.get("url", ""), v.get("tier", "primary")) for k, v in raw.items()
                    if isinstance(v, dict) and v.get("date")}
    except Exception:  # noqa: BLE001 -- a bad override file must never break import
        pass
    return {}


_VERIFIED_STAMPS = {**_VERIFIED_STAMPS, **{k: (d, (u or (_VERIFIED_STAMPS.get(k) or ("", "", ""))[1]), t)
                                            for k, (d, u, t) in _load_stamp_overrides().items()}}
for _p in PROP_FIRM_PRESETS:
    _st = _VERIFIED_STAMPS.get(_p.key)
    if _st:
        object.__setattr__(_p, "verified_on", _st[0])
        object.__setattr__(_p, "source_url", _st[1])
        object.__setattr__(_p, "verification_tier", _st[2])
        if _st[2] == "owner_attested":
            object.__setattr__(_p, "source_note", ((_p.source_note + " ") if _p.source_note else "") + (
                f"Marked verified by the account owner on {_st[0]} (owner-attested). Not independently re-checked "
                f"against the firm's page in that session -- confirm dollar figures at the firm's site before relying on them."))
del _p, _st

_BY_KEY: dict[str, PropFirmPreset] = {p.key: p for p in PROP_FIRM_PRESETS}


# ---------------------------------------------------------------------------
# Firm fee & reset table (2026-10 audit, build item 2).
#
# The EV verdict (app.orchestration.ev_verdict) needs real money numbers: what one attempt
# costs, what a reset costs, how much of a payout the trader keeps, and whether the firm
# actually permits the automation being backtested. Only entries below with a source_url and
# date_checked are real; a preset with NO entry has no fee data and the EV verdict says so
# instead of guessing. ``written_confirmation`` is True only once the firm has confirmed in
# WRITING (email / support ticket) that THIS strategy's automation style is permitted --
# a published help-centre line is not a confirmation for your specific bot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FirmFeeInfo:
    preset_key: str
    challenge_price: float | None       # one attempt (evaluation purchase), USD; None = unknown
    reset_price: float | None           # re-buy after a breach, USD; None = unknown (EV falls back to challenge_price)
    profit_split_pct: float             # trader's share of payouts, %
    source_url: str
    date_checked: str                   # ISO date this row was checked
    automation_policy: str              # what the firm's own text says about bots/EAs
    written_confirmation: bool = False  # True only after the firm confirmed IN WRITING for your setup
    source_tier: str = "secondary"      # "primary" | "secondary"
    notes: str = ""
    # One-time fee charged when an evaluation is PASSED (before the funded account starts), USD.
    # 0.0 for firms with none. The EV verdict charges it on every pass: EV = p*(q*V - A) - c.
    activation_fee: float = 0.0
    # 2026-10-15 audit: True when challenge_price is a MONTHLY SUBSCRIPTION (Topstep's Trading Combine), not a one-off
    # purchase. The EV verdict then charges challenge_price for every billing period the evaluation is expected to
    # take (see app.orchestration.ev_verdict.evaluation_months), instead of exactly one.
    monthly_subscription: bool = False
    billing_period_days: int = 30

    @property
    def effective_reset_price(self) -> float | None:
        return self.reset_price if self.reset_price is not None else self.challenge_price

    def to_dict(self) -> dict:
        return dict(self.__dict__)


FIRM_FEES: dict[str, FirmFeeInfo] = {
    # The one firm to buy first (assumption flagged in the delivery notes): Lucid Trading $50k LucidPro,
    # chosen because the repo already carries a dedicated, rule-checked preset for it
    # (EOD trailing + lock, soft daily loss, 3-day payouts) and the firm publishes that automation is allowed.
    # 2026-10-15 review item 5: Topstep 50K row. Without it get_fee_info("topstep_50k") returned None and
    # the EV verdict could only ever say INSUFFICIENT EVIDENCE for Topstep. Prices below were read on
    # 2026-10-15 from Topstep's own help-centre pages (PRIMARY source, no profit-split figure on them);
    # the 90/10 split is from a secondary page.
    "topstep_50k": FirmFeeInfo(
        preset_key="topstep_50k",
        challenge_price=49.0,
        reset_price=49.0,
        profit_split_pct=90.0,
        source_url="https://help.topstep.com/en/articles/14289835",
        date_checked="2026-10-15",
        automation_policy=(
            "Topstep's TopstepX API Access article says traders may build and run automated strategies and connect "
            "third-party tools (summarised by pickmytrade.io, 2026-09-08, which cites help.topstep.com). Conditions: all "
            "trading must originate from your own personal device -- VPS/VPN/remote-server order placement is prohibited "
            "(a private server may only log/backtest/chart); the Prohibited Trading Strategies article bans unfair-advantage "
            "systems (HFT / latency arbitrage); rule breaches are treated the same whether a person or a script placed the "
            "order; Topstep does not support third-party tools."
        ),
        written_confirmation=False,
        source_tier="primary",
        activation_fee=149.0,
        monthly_subscription=True,
        notes=(
            "Prices (help.topstep.com/en/articles/14289835 and the older intercom.help/topstep-llc/en/articles/9208217-topstep-pricing, "
            "both read 2026-10-15; neither shows a calendar date, only 'updated over 2 weeks / 3 months ago'): the 50K Trading Combine is "
            "$49 PER MONTH (a subscription, not a one-off fee -- an evaluation that takes 3 months to pass costs ~$147; since 2026-10-15 the EV model "
            "charges every expected month (monthly_subscription=True), estimated from the holdout days-to-pass). 50K Reset $49. Standard path: $149 one-time ACTIVATION fee "
            "per Express Funded Account earned (modelled via activation_fee). 'No Activation Fee' path: $95 per month and $95 reset, $0 "
            "activation -- not the path modelled here. Paths cannot be changed after purchase. Profit split 90/10 flat from the first dollar "
            "for traders who signed up on/after 2026-01-12 per proptradingvibes.com/blog/topstep-payout-rules (updated 2026-10-09, SECONDARY; "
            "earlier accounts may be grandfathered at 100% of the first $10,000). Get written confirmation from Topstep support that your "
            "specific automation is allowed (own-device execution) before paying, then set written_confirmation=True."
        ),
    ),
    "lucid_50k": FirmFeeInfo(
        preset_key="lucid_50k",
        challenge_price=172.0,
        reset_price=None,
        profit_split_pct=90.0,
        source_url="https://fundedtrading.com/propfirm/lucid/",
        date_checked="2026-10-11",
        automation_policy=(
            "Help centre 'Permitted Activities': automated systems permitted as long as they do not exploit the "
            "platform; HFT, latency arbitrage and reverse/hedge arbitrage prohibited "
            "(https://intercom.help/lucid-trading/en/articles/11321405-prohibited-trading-strategies). "
            "tradetanto.com (2026-10-11): 'Trading bots and trade copiers are allowed'; trader is responsible for software errors."
        ),
        written_confirmation=False,
        source_tier="secondary",
        notes=(
            "$172 is the LucidPro 50K evaluation price shown by fundedtrading.com (LucidFlex 50K: $136); Lucid's price depends "
            "on the DLL on/off option at checkout and on promotions, so CONFIRM AT CHECKOUT. Reset price was not published on any page "
            "read -> None (the EV code then uses the challenge price as the reset cost). Profit split 90/10 and $500 minimum payout request "
            "per proptradingvibes.com (updated 2026-10-11). Get written confirmation from Lucid support that your specific automation "
            "(entry method, order rate, no HFT) is allowed before paying, then set written_confirmation=True."
        ),
    ),
}

# 2026-10-17: rows for the remaining presets so the EV verdict is no longer blind. EVERY figure below was read from a
# SECONDARY comparison/review page on 2026-10-17 (none from the firm's own checkout), so each row is source_tier
# "secondary" with written_confirmation=False. Where a source gave two prices or none, the HIGHER price is used
# (conservative for EV) or the price is None (EV then reports INSUFFICIENT EVIDENCE rather than guess). CONFIRM
# EVERY PRICE AT CHECKOUT before relying on an EV number.
_CHECKED = "2026-10-17"


def _secondary_fee(key: str, price, reset, split, url: str, policy_from_preset: bool = True, *, activation: float = 0.0,
                   monthly: bool = False, notes: str = "") -> FirmFeeInfo:
    preset = next((p for p in PROP_FIRM_PRESETS if p.key == key), None)
    return FirmFeeInfo(
        preset_key=key, challenge_price=price, reset_price=reset, profit_split_pct=split, source_url=url,
        date_checked=_CHECKED, automation_policy=(preset.algo_policy_note if preset is not None else "") or "see preset note",
        written_confirmation=False, source_tier="secondary", notes=notes, activation_fee=activation, monthly_subscription=monthly,
    )


_APEX_URL = "https://propfirmmap.com/blog/apex-trader-funding-review-2026-payouts-profit-split-otp"
_APEX_NOTE = ("Apex went from monthly subscriptions to a ONE-TIME fee on 2026-03-01; prices verified by PropFirmMap on 2026-04-18 "
              "(page modified 2026-09-14). Split: 100% until $25,000 cumulatively withdrawn, then 90/10 -- 90 used here (conservative). "
              "Reset price not published on the page -> None. APEX BANS AUTOMATION (see automation_policy): READY/EV for a bot on Apex is "
              "moot unless Apex confirms in writing.")
_FIRM_FEE_ROWS = [
    _secondary_fee("apex_50k_eod", 197.0, None, 90.0, _APEX_URL, activation=99.0, notes=_APEX_NOTE),
    _secondary_fee("apex_50k_intraday", 132.0, None, 90.0, _APEX_URL, activation=79.0, notes=_APEX_NOTE),
    _secondary_fee("apex_100k_eod", 297.0, None, 90.0, _APEX_URL, activation=99.0, notes=_APEX_NOTE),
    _secondary_fee("apex_100k_intraday", 198.0, None, 90.0, _APEX_URL, activation=79.0, notes=_APEX_NOTE),
    _secondary_fee(
        "topstep_100k", 99.0, 99.0, 90.0, "https://proptradingvibes.com/blog/topstep-pricing-breakdown", activation=149.0, monthly=True,
        notes="Standard Path: $99 PER MONTH (subscription), $99 reset (free if a matching Reset Credit is banked), $149 one-time activation "
              "when passed. 'No Activation Fee' path ($149/mo, $0 activation) is not modelled. 90/10 from the first dollar for traders who "
              "joined the dashboard on/after 2026-01-12. Page 'Updated September 7, 2026'; tradingfunder.com agrees on $99/$149/$149."),
    _secondary_fee(
        "the5ers_20k", None, None, 80.0, "https://thegodfunded.com/en/firms/the5ers",
        notes="No High Stakes $20K price found on any page read (tiers shown: 2.5K, 5K, 10K, 25K, 50K, 100K) -> None, so EV says "
              "INSUFFICIENT EVIDENCE. Split starts at 80% (scales to 100%). Automation needs the firm's prior WRITTEN approval."),
    _secondary_fee(
        "the5ers_100k", 545.0, None, 80.0, "https://propfirmsfinder.com/prop-firm/the5ers/challenges/",
        notes="Sources disagree: $491 (fundedtrading.com, June 2026), ~$495 (same page text), $545 (propfirmsfinder.com) -> the highest used. "
              "Refund terms conflict between pages (staged refund incl. non-withdrawable Hub Credits) so NO refund is modelled. "
              "Split starts at 80%. Automation needs prior written approval."),
    _secondary_fee(
        "fundednext_25k", 189.99, None, 80.0, "https://propfirmmap.com/firms/fundednext",
        notes="Page showed $189.99 and $142.49 for the 25K (probably list vs discounted) -> list price used. Split 'up to 95%'; 80% is a "
              "CONSERVATIVE ASSUMPTION for the base split, not a verified figure. Fee 'refundable with 1st reward' is NOT modelled. "
              "Terms last checked by the source 2026-07-18."),
    _secondary_fee(
        "fundednext_100k", 529.99, None, 80.0, "https://propfirmmap.com/firms/fundednext",
        notes="Split 'up to 95%'; 80% is a CONSERVATIVE ASSUMPTION for the base split, not a verified figure. Refund-with-first-reward not "
              "modelled. $50k+ FundedNext accounts must be traded manually (see automation_policy)."),
    _secondary_fee(
        "lucid_100k", None, None, 90.0, "https://tradingfunder.com/firms/lucid-trading/",
        notes="No LucidPro 100K price could be attributed with confidence (propfirmmap listed a '$100K 1 Step' row at $225.40 sale / $307 list, "
              "not labelled LucidPro) -> None. Lucid's price depends on DLL on/off and promotions. 90/10 split (tradingfunder.com, updated 2026-07-20)."),
    _secondary_fee(
        "e8_one_100k", 488.0, None, 80.0, "https://fundedtrading.com/propfirm/e8-markets/",
        notes="Page (2026-09-03) showed $488 and $440 for the 100K -> higher used. Split is selectable at checkout (80/90/100%); 80% base used. "
              "No fixed reset price; 10% discount for retrying within 7 days. One-time fee, no activation/subscription (proptradingvibes.com, 2026-08-26)."),
    _secondary_fee(
        "funderpro_one_phase_100k", 539.0, None, 80.0, "https://fundedtrading.com/hi/propfirm/funderpro/",
        notes="$539 (July 2026). 80% base split, 90% with the Performance Reward add-on (not modelled). Fee refunded with first reward (not modelled). "
              "No reset price published."),
    _secondary_fee(
        "atlas_funded_1step_100k", 478.0, None, 80.0, "https://fundedtrading.com/propfirm/atlas-funded/",
        notes="Page showed $478 and $286 (unlabelled; the lower is probably a discount) -> higher used. The page claims '100%*' once funded with "
              "an unexplained asterisk; 80% is a CONSERVATIVE PLACEHOLDER, not a verified split. No reset price for 1-Step on the page. No date on the page."),
    _secondary_fee(
        "futures_desk_50k", 144.0, 75.0, 80.0, "https://blog.traderspost.io/article/the-futures-desk-pricing-evaluation-guide",
        notes="$50K: $144 EOD drawdown / $244 static; the $144 EOD row is used. Reset $75, activation $0 (all fees one-time). 80% split mentioned only in the "
              "article's conclusion. Article dated 2026-02-10; The Futures Desk runs frequent promotions."),
    # 2026-10-19: FTMO fee rows added (they were None, so the EV verdict could only ever say INSUFFICIENT EVIDENCE for FTMO).
    # FTMO prices its CFD Challenges in EUR. Two independent secondary sources (proptradingvibes.com, FTMO prices displayed on
    # 2026-08-12 / page updated 2026-09-01; passproptradingfirms.com, published 2026-06-08) agree on the 2-Step base fees:
    # 10K EUR 89, 100K EUR 540, 200K EUR 1,080. (A EUR 439 100K figure is a conditional first-purchase promo, not the base fee; the
    # higher base is used.) Converted at the last EUR/USD close in this repo's own data (1.15682, 2026-08-14); refresh this when the
    # rate moves. FTMO publishes NO reset/retry price: a failed attempt is re-bought at the full standard fee ("failed challenge
    # fees are not recoverable"), so reset_price == challenge_price. The 2-Step fee is refunded with the first Reward after BOTH
    # phases are passed -- not modelled (conservative). Sources are secondary: confirm at FTMO checkout before trusting EV.
    _secondary_fee(
        "ftmo_10k", round(89 * 1.15682, 2), round(89 * 1.15682, 2), 80.0, "https://proptradingvibes.com/blog/ftmo-account-sizes-pricing",
        notes="2-Step Challenge EUR 89 (= $102.96 at 1.15682 EUR/USD, 2026-08-14). Reset = full price again (FTMO publishes no retry discount). "
              "Split starts at 80% (90% with scaling/premium). Fee refund with first Reward not modelled. Cross-checked: "
              "https://www.passproptradingfirms.com/best-prop-firms/ftmo-review/ftmo-account-prices/"),
    _secondary_fee(
        "ftmo_100k", round(540 * 1.15682, 2), round(540 * 1.15682, 2), 80.0, "https://proptradingvibes.com/blog/ftmo-account-sizes-pricing",
        notes="2-Step Challenge EUR 540 (= $624.68 at 1.15682 EUR/USD, 2026-08-14); EUR 439 is a conditional first-purchase offer, not used. "
              "Reset = full price again (no published retry discount). Split starts at 80%. Fee refund with first Reward not modelled. "
              "Cross-checked: https://www.passproptradingfirms.com/best-prop-firms/ftmo-review/ftmo-account-prices/"),
    _secondary_fee(
        "ftmo_200k", round(1080 * 1.15682, 2), round(1080 * 1.15682, 2), 80.0, "https://proptradingvibes.com/blog/ftmo-account-sizes-pricing",
        notes="2-Step Challenge EUR 1,080 (= $1,249.37 at 1.15682 EUR/USD, 2026-08-14). Reset = full price again (no published retry discount). "
              "Split starts at 80%. Fee refund with first Reward not modelled. "
              "Cross-checked: https://www.passproptradingfirms.com/best-prop-firms/ftmo-review/ftmo-account-prices/"),
]
for _row in _FIRM_FEE_ROWS:
    FIRM_FEES.setdefault(_row.preset_key, _row)

# 2026-10-17 verification pass: fee figures that disagreed with the firm's own product pages. Rows stay tier "secondary"
# unless the figure came from the firm's checkout, which no tool here can reach -- confirm at checkout before trusting EV.
import dataclasses as _dc
_FEE_CORRECTIONS = {
    "fundednext_25k": dict(challenge_price=199.99, notes="2026-10-17: FundedNext product page lists $199.99 for the 25K Stellar 2-Step (was $189.99). "
        "Base split still a conservative 80% assumption; refund-with-first-reward not modelled."),
    "fundednext_100k": dict(challenge_price=549.99, notes="2026-10-17: FundedNext product page lists $549.99 for the 100K (was $529.99). "
        "Manual trading only at $50k+. Base split a conservative 80% assumption."),
    "atlas_funded_1step_100k": dict(challenge_price=538.0, profit_split_pct=90.0, notes="2026-10-17: Atlas docs list 90% split (100% add-on) and a $538 100K "
        "1-Step fee; the earlier $478 secondary figure is superseded. Target 10% vs 11% conflict between official pages is unresolved."),
    "futures_desk_50k": dict(challenge_price=139.0, reset_price=99.0, notes="2026-10-17: Futures Desk Static Core: $139 evaluation, $99 reset, 80% split "
        "(members.thefuturesdesk.com). Account-size mapping to $50k is inferred."),
}
for _k, _kw in _FEE_CORRECTIONS.items():
    if _k in FIRM_FEES:
        FIRM_FEES[_k] = _dc.replace(FIRM_FEES[_k], date_checked="2026-10-17", **_kw)
del _k, _kw


def get_fee_info(preset_key: str) -> FirmFeeInfo | None:
    """Fee/reset row for a preset, or None when no fee data has been recorded for it."""
    return FIRM_FEES.get(preset_key)


def fee_table() -> list[dict]:
    """Every preset with whatever fee data exists (None fields where unknown) -- for UIs and reports."""
    out = []
    for p in PROP_FIRM_PRESETS:
        f = FIRM_FEES.get(p.key)
        out.append({
            "preset_key": p.key, "label": p.label,
            "challenge_price": f.challenge_price if f else None,
            "reset_price": f.reset_price if f else None,
            "activation_fee": f.activation_fee if f else None,
            "profit_split_pct": f.profit_split_pct if f else None,
            "source_url": f.source_url if f else "",
            "date_checked": f.date_checked if f else "",
            "automation_policy": f.automation_policy if f else p.algo_policy_note,
            "written_confirmation": f.written_confirmation if f else False,
            "rules_verified_on": p.verified_on,
            "rules_verification_tier": p.verification_tier,
        })
    return out


# 2026-10-17 audit (item 6/7): READY needs the rules to have been verified against the firm's OWN page, recently.
#   * only ``verification_tier == "primary"`` counts (owner-attested and secondary stamps cap at MARGINAL);
#   * a stamp older than RULES_STAMP_MAX_AGE_DAYS no longer counts -- firms change rules often.
READY_VERIFICATION_TIERS = ("primary",)
RULES_STAMP_MAX_AGE_DAYS = 30


def rules_stamp_age_days(preset: "PropFirmPreset", today: "date | None" = None) -> int | None:
    """Days since the preset's rules stamp, or None when unstamped / unparseable."""
    from datetime import date as _date
    try:
        stamped = _date.fromisoformat((preset.verified_on or "").strip())
    except ValueError:
        return None
    return ((today or _date.today()) - stamped).days


def ready_eligibility(preset: "PropFirmPreset", today: "date | None" = None) -> tuple[bool, str]:
    """(eligible, reason-if-not). Eligible only when the rules carry a stamp date and source URL, the stamp is
    PRIMARY tier (the firm's own page), and it is no older than RULES_STAMP_MAX_AGE_DAYS."""
    if not (preset.verified_on and preset.source_url):
        return False, f"preset '{preset.key}' has no verified_on / source_url (rules not verified against a firm page)"
    if preset.verification_tier not in READY_VERIFICATION_TIERS:
        return False, (f"preset '{preset.key}' rules are only '{preset.verification_tier or 'unstamped'}'-verified "
                       "(READY needs the firm's own page: tier 'primary')")
    age = rules_stamp_age_days(preset, today)
    if age is None:
        return False, f"preset '{preset.key}' has an unreadable verified_on date {preset.verified_on!r}"
    if age > RULES_STAMP_MAX_AGE_DAYS:
        return False, (f"preset '{preset.key}' rules were last verified {age} days ago ({preset.verified_on}); "
                       f"READY needs a stamp no older than {RULES_STAMP_MAX_AGE_DAYS} days -- re-check the firm's page")
    return True, ""


def is_ready_eligible(preset: "PropFirmPreset", today: "date | None" = None) -> bool:
    """A preset can support a READY / HIGH CONFIDENCE verdict only if its RULES were verified against the firm's
    own page (primary tier) within RULES_STAMP_MAX_AGE_DAYS. Anything else caps at MARGINAL."""
    return ready_eligibility(preset, today)[0]


def is_ready_eligible_key(preset_key: str | None) -> bool:
    if not preset_key:
        return False
    try:
        return is_ready_eligible(get_preset(preset_key))
    except KeyError:
        return False


def list_presets() -> list[PropFirmPreset]:
    """All presets, in catalog order (grouped by firm)."""
    return list(PROP_FIRM_PRESETS)


def list_firms() -> list[str]:
    """Distinct firm names, in first-seen catalog order."""
    seen: list[str] = []
    for p in PROP_FIRM_PRESETS:
        if p.firm not in seen:
            seen.append(p.firm)
    return seen


def get_preset(key: str) -> PropFirmPreset:
    try:
        return _BY_KEY[key]
    except KeyError as exc:
        raise KeyError(f"Unknown prop-firm preset key '{key}'. Known keys: {sorted(_BY_KEY)}") from exc


def presets_for_firm(firm: str) -> list[PropFirmPreset]:
    return [p for p in PROP_FIRM_PRESETS if p.firm.lower() == firm.lower()]


def algo_allowed_presets() -> list[PropFirmPreset]:
    """Presets whose firm explicitly allows algorithmic/automated trading
    (v7: algo_trading_allowed=True), in catalog order.

    Minimal UI wiring hook: UIs that list presets (Prop-Firm Recommender,
    Speed Run, Evolution Lab firm pickers) can call this to offer an
    "algo-friendly only" filter or badge without touching the catalog.
    See CHANGES.txt (v7 worker C) for the documented UI gap -- the
    desktop/web preset pickers do not surface the flag yet; the field and
    this helper are the complete data-side contract for that work."""
    return [p for p in PROP_FIRM_PRESETS if p.algo_trading_allowed]
