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

MODELING LIMITATION -- consistency_rule_pct is a single field checked
ONLY at the evaluation-pass gate (see app.prop.simulator.simulate_account:
"best_day_profit / total_profit_since_start <= consistency_rule_pct",
checked once, at the moment balance first clears the profit target).
Several firms below (Apex, Lucid) have NO consistency rule during
evaluation at all -- their real consistency rule only gates PAYOUTS on
the funded account, which this simulator does not separately model. For
those firms, consistency_rule_pct is correctly set to None (no eval-stage
gate) even though a real funded-stage consistency rule exists; don't
read None as "this firm has no consistency rule anywhere," and don't
"fix" it by putting the funded-stage number here -- that would incorrectly
block the evaluation-pass check on a rule that, in reality, doesn't apply
until afterward. Each such preset's source_note says where the real
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
    as_of: str = ""                 # date this preset was last checked against the firm's own rules page
    source_note: str = ""           # short pointer to what to re-check and where

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

PROP_FIRM_PRESETS: list[PropFirmPreset] = [
    # --- FTMO ---------------------------------------------------------
    # Re-verified directly against ftmo.com/en/trading-objectives/ (2026-09-13):
    # matches the FTMO Challenge: 2-Step's Phase 1 exactly -- 10% profit
    # target, 5% daily loss (of Initial Simulated Capital), 10% STATIC
    # max loss (fixed at Initial Simulated Capital - 10%, confirmed not
    # trailing on the 2-Step route), and a confirmed 4 minimum trading
    # days. Payout: FTMO's own payout-policy pages and most 2026 reviews
    # converge on a 14-day first-payout window from the funded account's
    # first trade. No numeric changes needed here -- this preset was
    # already accurate.
    PropFirmPreset(
        key="ftmo_10k", firm="FTMO", label="FTMO - $10k Challenge",
        account_size=10_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note=(
            "Confirmed 2026-09-13 against ftmo.com/en/trading-objectives/ -- models the FTMO Challenge: "
            "2-Step (Phase 1 numbers; Phase 2 relaxes the profit target to 5%, which this single-PropRules "
            "model can't represent as a separate phase). FTMO also now offers a 1-Step Challenge (launched "
            "Feb 2026: single 10% target, tighter 3% daily loss, 10% TRAILING max loss, a 50%-of-positive-"
            "days 'Best Day Rule' instead of a minimum-trading-days rule) -- not modeled as a separate "
            "preset here; add one if the 1-Step route matters for your timeline."
        ),
    ),
    PropFirmPreset(
        key="ftmo_100k", firm="FTMO", label="FTMO - $100k Challenge",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note=(
            "Confirmed 2026-09-13 against ftmo.com/en/trading-objectives/ -- see the $10k preset's note "
            "for the 2-Step-vs-1-Step and phase-modeling caveats, which apply identically here."
        ),
    ),
    PropFirmPreset(
        key="ftmo_200k", firm="FTMO", label="FTMO - $200k Challenge",
        account_size=200_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note=(
            "Confirmed 2026-09-13 against ftmo.com/en/trading-objectives/ -- see the $10k preset's note "
            "for the 2-Step-vs-1-Step and phase-modeling caveats, which apply identically here."
        ),
    ),
    # --- Apex Trader Funding -------------------------------------------
    # MAJOR UPDATE: Apex overhauled its rules on March 1, 2026 ("Apex 4.0").
    # Per Apex's own Help Center (apextraderfunding.com/help-center/...):
    # the evaluation now has NO consistency rule at all (previously 30%,
    # removed) and NO minimum-trading-days rule (previously ~1 day,
    # removed -- "pass in one day if you hit the target cleanly" is now
    # explicitly the documented behavior, not just a loophole). The 50%
    # payout-stage consistency rule and the "at least 5 qualifying
    # trading days" payout-eligibility rule are real, but apply to the
    # FUNDED Performance Account, not the evaluation -- see this module's
    # docstring's MODELING LIMITATION note for why consistency_rule_pct
    # stays None here rather than being set to 50.
    PropFirmPreset(
        key="apex_50k", firm="Apex Trader Funding", label="Apex - $50k Evaluation",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,  # Apex has no daily loss limit during evaluation -- see source_note
        max_drawdown_pct=6.0,
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=0,
        payout_frequency_days=5,
        as_of=_AS_OF,
        source_note=(
            "Updated 2026-09-13 for the Apex 4.0 overhaul (March 1, 2026): min_trading_days is now 0 (no "
            "minimum -- Apex's own Help Center confirms a same-day pass is valid) and consistency_rule_pct "
            "is None (no evaluation-stage consistency rule; see module docstring). payout_frequency_days=5 "
            "reflects Apex's own EOD/Intraday Performance Account payout pages: 'at least 5 qualifying "
            "trading days' before the first payout request -- a real 50% consistency rule and a "
            "size-specific 'Safety Net' profit buffer ALSO gate that first payout and are not modeled by "
            "this simulator's fields. The exact trailing-drawdown DOLLAR amount is genuinely ambiguous "
            "across sources as of this check ($2,000-$2,500 depending on EOD vs Intraday and Rithmic vs "
            "Tradovate vs Wealthcharts) -- the 6% here is a reasonable midpoint, not a confirmed figure; "
            "verify the exact number for your chosen platform/product at apextraderfunding.com before "
            "relying on it."
        ),
    ),
    PropFirmPreset(
        key="apex_100k", firm="Apex Trader Funding", label="Apex - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,
        max_drawdown_pct=6.0,
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=None, min_trading_days=0,
        payout_frequency_days=5,
        as_of=_AS_OF,
        source_note=(
            "Updated 2026-09-13 for the Apex 4.0 overhaul -- see the $50k preset's note for the full "
            "explanation (no eval consistency rule, no minimum trading days, 5-day payout eligibility, "
            "and the same trailing-drawdown-dollar-amount ambiguity across EOD/Intraday/platform variants)."
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
    ),
    PropFirmPreset(
        key="topstep_100k", firm="TopStep", label="TopStep - $100k Trading Combine",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,
        max_drawdown_pct=3.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=50.0, min_trading_days=2,
        payout_frequency_days=5,
        as_of=_AS_OF,
        source_note="Re-verified 2026-09-13 against help.topstep.com -- see the $50k preset's note for the full explanation.",
    ),
    # --- The5%ers ---------------------------------------------------------
    PropFirmPreset(
        key="the5ers_20k", firm="The5%ers", label="The5%ers - $20k Bootcamp",
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
            "confirm which program name your account actually uses at the5ers.com."
        ),
    ),
    PropFirmPreset(
        key="the5ers_100k", firm="The5%ers", label="The5%ers - $100k Bootcamp",
        account_size=100_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=30.0, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note="Re-verified 2026-09-13 -- see the $20k preset's note for the full explanation and program-naming caveat.",
    ),
    # --- FundedNext ---------------------------------------------------------
    PropFirmPreset(
        key="fundednext_25k", firm="FundedNext", label="FundedNext - $25k Evaluation",
        account_size=25_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=40.0, min_trading_days=5,
        payout_frequency_days=7,
        as_of=_AS_OF,
        source_note=(
            "Re-verified 2026-09-13. min_trading_days=5, and the 10%/5%/10% target/daily/max-drawdown "
            "numbers were already accurate for FundedNext's 2-phase Evaluation product. consistency_rule_pct "
            "is now set to 40.0 (was None) -- confirmed rule: 'no single day can account for more than 40% "
            "of your total profit' (one source states 30% instead; 40% was the more directly-quoted figure "
            "and is used here, but this is worth a direct check). payout_frequency_days lowered from 15 to "
            "7 as a rough midpoint for the standard Evaluation product's 'weekly payouts' framing -- of "
            "everything re-checked in this pass, THIS is the least confidently sourced number: FundedNext "
            "now runs Evaluation/Express/Stellar 1-Step/Stellar 2-Step/Rapid Pro/Rapid Daily products with "
            "payout cycles documented anywhere from 3 days to daily to bi-weekly depending on which one you "
            "actually bought. Confirm the exact figure for your specific product at fundednext.com."
        ),
    ),
    PropFirmPreset(
        key="fundednext_100k", firm="FundedNext", label="FundedNext - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=40.0, min_trading_days=5,
        payout_frequency_days=7,
        as_of=_AS_OF,
        source_note="Re-verified 2026-09-13 -- see the $25k preset's note, including the payout-cadence caveat (least confident figure in this catalog).",
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
        consistency_rule_pct=None, min_trading_days=1,
        payout_frequency_days=3,
        as_of=_AS_OF,
        source_note=(
            "CORRECTED 2026-09-13 against lucidtrading.com (LucidPro track) -- previous values were stale. "
            "Confirmed: $3,000 profit target on $50k = 6% (was modeled as 8%); $1,200 daily loss limit = "
            "2.4% (was 4%); $2,000 EOD trailing max loss = 4% (was 8%); minimum trading days = 1, not 3 "
            "('pass in as little as one day' is the current LucidPro rule); payout cycle = 3 trading days, "
            "not 14 -- confirmed both by Lucid's own live pricing page ('Days to Payout: 3') and independent "
            "sources. consistency_rule_pct stays None: LucidPro has NO consistency rule during evaluation -- "
            "the real 40% consistency rule shown on Lucid's site applies only at the funded-payout stage "
            "(see module docstring's MODELING LIMITATION note), same situation as Apex above. Lucid also "
            "sells LucidFlex (50% eval consistency, no funded consistency, no DLL), LucidDaily, and "
            "LucidDirect (skips the evaluation entirely, 20% funded consistency) -- this preset models "
            "LucidPro specifically, the track shown in the screenshot this correction was verified against."
        ),
    ),
    PropFirmPreset(
        key="lucid_100k", firm="Lucid Trading", label="Lucid Trading - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=2.4, max_drawdown_pct=3.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=1,
        payout_frequency_days=3,
        as_of=_AS_OF,
        source_note=(
            "CORRECTED 2026-09-13 -- see the $50k preset's note for the full explanation. max_drawdown_pct "
            "here is 3% (not 4%) because Lucid's max-loss DOLLAR amounts don't scale as a flat percentage "
            "across sizes -- confirmed $3,000 max loss on the $100k tier = 3%. evaluation_profit_target_pct "
            "and daily_loss_limit_pct are carried over from the $50k tier's percentage (Lucid markets these "
            "as 'the same profit targets' across sizes) but the exact $100k dollar figures weren't directly "
            "confirmed in this pass -- worth a direct check at lucidtrading.com before relying on them."
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
