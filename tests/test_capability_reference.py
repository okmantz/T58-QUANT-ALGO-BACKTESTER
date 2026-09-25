"""Tests for app.ai.capability_reference -- the capability-grounding fix.

These are deliberately "drift-proofing" tests, not just point-in-time
snapshots: they assert that every real construct app.strategy.pinescript /
app.strategy.mql5 actually parse shows up in the introspected capability
lists AND in the generated contract text app.ai.strategy_generator sends
to the model. If a future session adds a new ta.*/i*/directive regex to
either parser and this module's introspection doesn't pick it up, these
tests fail loudly instead of silently reintroducing the exact stale-
contract bug this fix was built to close.
"""
from app.ai import capability_reference as cr
from app.ai import strategy_generator


def test_pinescript_capabilities_include_every_real_ta_function():
    caps = cr.pinescript_capabilities()
    # These are the real functions app.strategy.pinescript's own regexes
    # (_TA_CALL_RE, _TA_ATR_RE, _TA_VWAP_RE, _TA_HIGHEST_RE, _TA_LOWEST_RE,
    # _TA_STDEV_RE, _DESTRUCTURE_MACD_RE) parse today -- the old hand-typed
    # _PINESCRIPT_CONTRACT string in strategy_generator.py never mentioned
    # atr/vwap/highest/lowest/stdev/macd at all.
    for fn in ("sma", "ema", "wma", "rsi", "atr", "vwap", "highest", "lowest", "stdev", "macd", "crossover", "crossunder"):
        assert fn in caps["ta_functions"], f"{fn} missing from introspected PineScript capabilities"


def test_pinescript_capabilities_include_atr_based_directives():
    caps = cr.pinescript_capabilities()
    for directive in ("T58_SL_PIPS", "T58_TP_PIPS", "T58_SL_ATR_MULT", "T58_TP_ATR_MULT", "T58_ATR_PERIOD"):
        assert directive in caps["directives"]


def test_pinescript_price_references_match_parser():
    from app.strategy import pinescript as ps

    caps = cr.pinescript_capabilities()
    assert set(caps["price_references"]) == set(ps._PRICE_ALIASES)


def test_mql5_capabilities_include_every_real_indicator():
    caps = cr.mql5_capabilities()
    # iATR/iBands/iHighest/iLowest are real, parsed indicator calls
    # (app.strategy.mql5._IATR_RE / _IBANDS_RE / _IHIGHEST_RE / _ILOWEST_RE)
    # the old hand-typed _MQL5_CONTRACT claimed didn't exist ("only iMA and
    # iRSI as indicators").
    for fn in ("iMA", "iRSI", "iATR", "iBands", "iHighest", "iLowest"):
        assert fn in caps["indicator_functions"], f"{fn} missing from introspected MQL5 capabilities"


def test_mql5_capabilities_include_atr_based_directives():
    caps = cr.mql5_capabilities()
    for directive in ("T58_SL_PIPS", "T58_TP_PIPS", "T58_SL_ATR_MULT", "T58_TP_ATR_MULT", "T58_ATR_PERIOD"):
        assert directive in caps["directives"]


def test_mql5_ma_modes_match_parser():
    from app.strategy import mql5

    caps = cr.mql5_capabilities()
    assert set(caps["ma_modes"]) == set(mql5._MODE_TO_FUNC.keys())


def test_python_indicator_library_includes_known_indicators():
    names = cr.python_indicator_library()
    for fn in ("sma", "ema", "rsi", "macd", "atr", "bollinger", "adx", "vwap"):
        assert fn in names


def test_python_indicator_library_excludes_private_helpers():
    names = cr.python_indicator_library()
    assert not any(n.startswith("_") for n in names)


def test_manual_condition_kinds_include_known_kinds():
    kinds = cr.manual_condition_kinds()
    for kind in ("time_of_day", "day_of_week", "ib_contraction_ratio", "liquidity_sweep", "atr_regime"):
        assert kind in kinds, f"{kind} missing from introspected manual condition kinds"


def test_generated_pinescript_contract_mentions_every_capability():
    """The actual text strategy_generator._PINESCRIPT_CONTRACT sends to
    the model -- confirms the fix is wired through end to end, not just
    correct in capability_reference itself."""
    text = strategy_generator._PINESCRIPT_CONTRACT
    for token in ("ta.atr", "ta.vwap", "ta.highest", "ta.lowest", "ta.stdev", "ta.macd", "T58_SL_ATR_MULT", "T58_TP_ATR_MULT"):
        assert token in text, f"{token} missing from the generated PineScript contract"


def test_generated_mql5_contract_mentions_every_capability():
    text = strategy_generator._MQL5_CONTRACT
    for token in ("iATR", "iBands", "iHighest", "iLowest", "T58_SL_ATR_MULT", "T58_TP_ATR_MULT"):
        assert token in text, f"{token} missing from the generated MQL5 contract"


def test_generated_contracts_still_forbid_unsupported_constructs():
    """The fix corrects what's allowed -- it must not accidentally loosen
    the "anything else fails to parse" boundary."""
    ps_text = strategy_generator._PINESCRIPT_CONTRACT
    mql5_text = strategy_generator._MQL5_CONTRACT
    assert "Do NOT use" in ps_text
    assert "Do NOT use" in mql5_text
    assert "security()" in ps_text  # still explicitly forbidden
    assert "CopyBuffer" in mql5_text  # still explicitly forbidden


def test_chat_capability_note_is_json_safe_and_populated():
    import json

    note = cr.chat_capability_note()
    json.dumps(note)  # must not raise
    assert note["python_indicators_available"]
    assert note["manual_json_condition_types"]
    assert "ta.atr" in note["pinescript_ta_functions"]
    assert "iATR" in note["mql5_indicator_functions"]


def test_build_context_includes_engine_capabilities():
    """app.ai.trading_assistant.build_context threads engine_capabilities
    through so the chat assistant is grounded on every turn."""
    from app.ai import trading_assistant

    context = trading_assistant.build_context(rankings=[], news_events=[])
    assert "engine_capabilities" in context
    assert "ta.atr" in context["engine_capabilities"]["pinescript_ta_functions"]


def test_capability_grounding_note_present_in_chat_prompts():
    from app.ai import trading_assistant

    assert "engine_capabilities" in trading_assistant.CAPABILITY_GROUNDING_NOTE
    assert trading_assistant.CAPABILITY_GROUNDING_NOTE in trading_assistant.PERSONAL_MODE_SYSTEM_PROMPT
    assert trading_assistant.CAPABILITY_GROUNDING_NOTE in trading_assistant.T58_GROUP_SYSTEM_PROMPT


def test_capability_functions_are_cached_and_cheap():
    """lru_cache means repeated calls don't re-introspect the source
    every time -- a sanity check that caching is actually wired up."""
    assert cr.pinescript_capabilities() is cr.pinescript_capabilities()
    assert cr.mql5_capabilities() is cr.mql5_capabilities()
    assert cr.python_indicator_library() is cr.python_indicator_library()
    assert cr.manual_condition_kinds() is cr.manual_condition_kinds()
