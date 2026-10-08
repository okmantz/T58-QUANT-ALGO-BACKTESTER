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
            "apextraderfunding.com before relying on them."
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
        consistency_rule_pct=50.0, min_trading_days=2,
        payout_frequency_days=5,
        winning_days_for_payout=5, min_winning_day_profit=150.0,
        funded_consistency_rule_pct=40.0,
        as_of=_AS_OF,
        source_note=(
            "Re-verified 2026-09-13 against help.topstep.com. min_trading_days=2 and max_drawdown_pct=4.0 "
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

_BY_KEY: dict[str, PropFirmPreset] = {p.key: p for p in PROP_FIRM_PRESETS}


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
