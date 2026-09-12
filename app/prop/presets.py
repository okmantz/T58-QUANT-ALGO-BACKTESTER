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
# ---------------------------------------------------------------------------

_AS_OF = "2026-09"  # last time this catalog's numbers were checked/updated

PROP_FIRM_PRESETS: list[PropFirmPreset] = [
    # --- FTMO ---------------------------------------------------------
    PropFirmPreset(
        key="ftmo_10k", firm="FTMO", label="FTMO - $10k Challenge",
        account_size=10_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current target/drawdown % and the Swing/Normal account variant at ftmo.com.",
    ),
    PropFirmPreset(
        key="ftmo_100k", firm="FTMO", label="FTMO - $100k Challenge",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current target/drawdown % and the Swing/Normal account variant at ftmo.com.",
    ),
    PropFirmPreset(
        key="ftmo_200k", firm="FTMO", label="FTMO - $200k Challenge",
        account_size=200_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=4,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current target/drawdown % and the Swing/Normal account variant at ftmo.com.",
    ),
    # --- Apex Trader Funding -------------------------------------------
    PropFirmPreset(
        key="apex_50k", firm="Apex Trader Funding", label="Apex - $50k Evaluation",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,  # Apex has no hard daily loss limit on most plans -- see source_note
        max_drawdown_pct=6.0,
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=30.0, min_trading_days=1,
        payout_frequency_days=8,
        as_of=_AS_OF,
        source_note=(
            "Apex's trailing threshold/consistency-rule terms change frequently -- verify current "
            "trailing-drawdown %, consistency rule, and minimum trading days at apextraderfunding.com. "
            "daily_loss_limit_pct is set to a non-binding 100% here since Apex evaluations are not "
            "built around a firm daily-loss cutoff the way FTMO-style challenges are."
        ),
    ),
    PropFirmPreset(
        key="apex_100k", firm="Apex Trader Funding", label="Apex - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,
        max_drawdown_pct=6.0,
        drawdown_type="trailing", drawdown_check_mode="intrabar",
        consistency_rule_pct=30.0, min_trading_days=1,
        payout_frequency_days=8,
        as_of=_AS_OF,
        source_note="Verify current trailing-drawdown %, consistency rule, and minimum trading days at apextraderfunding.com.",
    ),
    # --- TopStep --------------------------------------------------------
    PropFirmPreset(
        key="topstep_50k", firm="TopStep", label="TopStep - $50k Trading Combine",
        account_size=50_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,  # TopStep uses a daily loss LIMIT that halts trading, not a hard fail -- see note
        max_drawdown_pct=4.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=2,
        payout_frequency_days=14,
        as_of=_AS_OF,
        source_note=(
            "TopStep's daily loss limit stops new trades for the day rather than failing the account "
            "outright, which this simplified model doesn't distinguish -- verify current target/trailing "
            "drawdown % and minimum trading days at topstep.com."
        ),
    ),
    PropFirmPreset(
        key="topstep_100k", firm="TopStep", label="TopStep - $100k Trading Combine",
        account_size=100_000, evaluation_profit_target_pct=6.0,
        daily_loss_limit_pct=100.0,
        max_drawdown_pct=3.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=2,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current target/trailing drawdown % and minimum trading days at topstep.com.",
    ),
    # --- The5%ers ---------------------------------------------------------
    PropFirmPreset(
        key="the5ers_20k", firm="The5%ers", label="The5%ers - $20k Bootcamp",
        account_size=20_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current program (Bootcamp/Instant/Hyper Growth) target and drawdown terms at the5ers.com.",
    ),
    PropFirmPreset(
        key="the5ers_100k", firm="The5%ers", label="The5%ers - $100k Bootcamp",
        account_size=100_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current program (Bootcamp/Instant/Hyper Growth) target and drawdown terms at the5ers.com.",
    ),
    # --- FundedNext ---------------------------------------------------------
    PropFirmPreset(
        key="fundednext_25k", firm="FundedNext", label="FundedNext - $25k Evaluation",
        account_size=25_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=5,
        payout_frequency_days=15,
        as_of=_AS_OF, source_note="Verify current Evaluation/Stellar/Express model terms at fundednext.com.",
    ),
    PropFirmPreset(
        key="fundednext_100k", firm="FundedNext", label="FundedNext - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=10.0,
        daily_loss_limit_pct=5.0, max_drawdown_pct=10.0,
        drawdown_type="static", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=5,
        payout_frequency_days=15,
        as_of=_AS_OF, source_note="Verify current Evaluation/Stellar/Express model terms at fundednext.com.",
    ),
    # --- Lucid Trading ---------------------------------------------------
    PropFirmPreset(
        key="lucid_50k", firm="Lucid Trading", label="Lucid Trading - $50k Evaluation",
        account_size=50_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=4.0, max_drawdown_pct=8.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current target/drawdown % and account model at lucidtrading.com.",
    ),
    PropFirmPreset(
        key="lucid_100k", firm="Lucid Trading", label="Lucid Trading - $100k Evaluation",
        account_size=100_000, evaluation_profit_target_pct=8.0,
        daily_loss_limit_pct=4.0, max_drawdown_pct=8.0,
        drawdown_type="trailing", drawdown_check_mode="eod",
        consistency_rule_pct=None, min_trading_days=3,
        payout_frequency_days=14,
        as_of=_AS_OF, source_note="Verify current target/drawdown % and account model at lucidtrading.com.",
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
