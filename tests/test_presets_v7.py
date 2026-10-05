"""v7 regression tests -- algorithmic-trading policy flags on prop presets.

Covers workstream A:
  1. The `algo_trading_allowed` / `algo_policy_note` fields exist on every
     preset and are backfilled on all 15 pre-v7 presets (never defaulted
     blindly -- each flag below was verified against the firm's own
     official rules/help-center page, 2026-10-05).
  2. The four new v7 presets (firms that explicitly allow algos) carry all
     required fields, an as-of date, a source note with checkable URLs,
     and build a valid PropRules.
  3. Standing rule: no Alpha Futures preset anywhere.
"""

import re

import pytest

from app.prop.presets import (
    PROP_FIRM_PRESETS,
    algo_allowed_presets,
    get_preset,
    list_presets,
)

# The 15 presets that existed before v7 -- every one must be backfilled.
LEGACY_KEYS = [
    "ftmo_10k",
    "ftmo_100k",
    "ftmo_200k",
    "apex_50k_eod",
    "apex_50k_intraday",
    "apex_100k_eod",
    "apex_100k_intraday",
    "topstep_50k",
    "topstep_100k",
    "the5ers_20k",
    "the5ers_100k",
    "fundednext_25k",
    "fundednext_100k",
    "lucid_50k",
    "lucid_100k",
]

# Verified 2026-10-05 against each firm's own official pages (see
# CHANGES.txt for the per-firm source list). "False" here does NOT mean
# "banned" -- it means "not explicitly, unconditionally allowed" (banned,
# conditional, approval-gated, or account-size-gated all land here with
# the reason spelled out in algo_policy_note).
EXPECTED_ALGO_FLAGS = {
    # FTMO: official FAQ -- "no reasons for limiting or restricting your
    # trading strategy, whether it's discretionary trading, algorithmic
    # trading, EAs, etc." (ftmo.com/en/faq)
    "ftmo_10k": True,
    "ftmo_100k": True,
    "ftmo_200k": True,
    # Apex: official prohibited-activities page -- "No Automation or
    # Algorithm Usage allowed."
    "apex_50k_eod": False,
    "apex_50k_intraday": False,
    "apex_100k_eod": False,
    "apex_100k_intraday": False,
    # Topstep: official help center -- "Automated strategies are permitted"
    # on Combine / Express Funded (ProjectX API automation banned on LFA).
    "topstep_50k": True,
    "topstep_100k": True,
    # The5ers: official FAQ permits own-code EAs, BUT the Terms require
    # prior WRITTEN APPROVAL for automated trading software -> approval-
    # gated -> False with note.
    "the5ers_20k": False,
    "the5ers_100k": False,
    # FundedNext Stellar 2-Step: official help center -- EAs welcome on
    # MT4/MT5 for accounts BELOW $50k (paid EA add-on); $50k+ must trade
    # fully manually.
    "fundednext_25k": True,
    "fundednext_100k": False,
    # Lucid: official help center "Permitted Activities" -- automated
    # trading systems permitted (HFT prohibited, trader responsible for
    # software errors).
    "lucid_50k": True,
    "lucid_100k": True,
}

# New v7 presets -- firms whose official pages explicitly allow algos.
NEW_ALGO_KEYS = [
    "e8_one_100k",
    "funderpro_one_phase_100k",
    "atlas_funded_1step_100k",
    "futures_desk_50k",
]

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def test_every_preset_carries_algo_fields():
    for p in PROP_FIRM_PRESETS:
        assert isinstance(p.algo_trading_allowed, bool), p.key
        assert isinstance(p.algo_policy_note, str), p.key


def test_legacy_backfill_matches_verified_flags():
    by_key = {p.key: p for p in PROP_FIRM_PRESETS}
    for key in LEGACY_KEYS:
        assert key in by_key, f"legacy preset {key} missing from catalog"
    for key, expected in EXPECTED_ALGO_FLAGS.items():
        actual = by_key[key].algo_trading_allowed
        assert actual is expected, (
            f"{key}: algo_trading_allowed={actual}, expected {expected} "
            f"(verified 2026-10-05 -- see CHANGES.txt)"
        )


def test_allowed_presets_always_carry_a_policy_note():
    for p in PROP_FIRM_PRESETS:
        if p.algo_trading_allowed:
            assert p.algo_policy_note.strip(), (
                f"{p.key}: allowed but has no policy note explaining the conditions"
            )


def test_disallowed_presets_explain_why():
    # A False flag must never be a silent default -- the note says why.
    for p in PROP_FIRM_PRESETS:
        if not p.algo_trading_allowed:
            assert p.algo_policy_note.strip(), (
                f"{p.key}: not algo-allowed but no explanatory note"
            )


def test_new_algo_presets_exist_and_are_flagged():
    for key in NEW_ALGO_KEYS:
        p = get_preset(key)
        assert p.algo_trading_allowed is True, key
        assert p.algo_policy_note.strip(), key


def test_new_algo_presets_have_all_required_fields():
    for key in NEW_ALGO_KEYS:
        p = get_preset(key)
        assert p.firm.strip(), key
        assert p.label.strip(), key
        assert p.account_size > 0, key
        assert p.evaluation_profit_target_pct > 0, key
        assert p.daily_loss_limit_pct > 0, key
        assert p.max_drawdown_pct > 0, key
        assert p.drawdown_type in ("trailing", "static"), key
        assert p.drawdown_check_mode in ("intrabar", "eod"), key
        assert p.min_trading_days >= 0, key
        assert DATE_RE.match(p.as_of), f"{key}: as_of={p.as_of!r} is not YYYY-MM-DD"
        assert "http" in p.source_note, f"{key}: source_note has no checkable URL"


def test_new_algo_presets_build_valid_prop_rules():
    for key in NEW_ALGO_KEYS:
        rules = get_preset(key).to_prop_rules()
        assert rules.account_size == get_preset(key).account_size
        assert rules.max_drawdown_pct == get_preset(key).max_drawdown_pct


def test_algo_allowed_presets_helper():
    allowed = algo_allowed_presets()
    allowed_keys = {p.key for p in allowed}
    for key, expected in EXPECTED_ALGO_FLAGS.items():
        if expected:
            assert key in allowed_keys, key
        else:
            assert key not in allowed_keys, key
    for key in NEW_ALGO_KEYS:
        assert key in allowed_keys, key
    # helper returns the same objects, in catalog order
    assert allowed == [p for p in list_presets() if p.algo_trading_allowed]


def test_preset_keys_unique():
    keys = [p.key for p in PROP_FIRM_PRESETS]
    assert len(keys) == len(set(keys))


def test_no_alpha_futures_preset():
    for p in PROP_FIRM_PRESETS:
        assert "alpha" not in p.firm.lower(), p.key
        assert "alpha" not in p.key.lower(), p.key
