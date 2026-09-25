"""Tests for app.ai.ai_director -- the deterministic portfolio-level
priority engine behind the AI Director panel, plus a smoke test of the
web routes that wire it up (app.web.ai_assistant_routes)."""
import time

import pytest

from app.ai import ai_director


def _fake_progress(status_map):
    """Builds a pipeline_progress_fn stand-in: metadata carries a
    "_stage_key" the fake looks up in status_map, so tests can drive
    score_one() without touching the real filesystem-backed
    app.strategy.library.compute_pipeline_progress at all."""

    def fn(metadata):
        return status_map[metadata["_stage_key"]]

    return fn


READY_PROGRESS = {"current_stage": "ready", "next_stage": None, "verdict": "READY"}
MARGINAL_PROGRESS = {"current_stage": "champion_check", "next_stage": None, "verdict": "MARGINAL"}
NOT_READY_PROGRESS = {"current_stage": "champion_check", "next_stage": None, "verdict": "NOT READY"}
TESTED_ONLY_PROGRESS = {"current_stage": "test", "next_stage": "optimize", "verdict": None}
OPTIMIZED_ONLY_PROGRESS = {"current_stage": "optimize", "next_stage": "validate", "verdict": None}
DRAFT_PROGRESS = {"current_stage": "create", "next_stage": "test", "verdict": None}
DONE_PROGRESS = {"current_stage": "ready", "next_stage": None, "verdict": "READY"}


def _strategy(name, stage_key, status="draft", market=None, modified_ts=None, now=None):
    now = now if now is not None else time.time()
    modified_ts = modified_ts if modified_ts is not None else now
    return {
        "name": name, "strategy_type": "python", "status": status,
        "metadata": {"_stage_key": stage_key, **({"market": market} if market else {})},
        "modified": modified_ts,
    }


def test_ready_not_live_becomes_promote_top_priority():
    now = time.time()
    strategies = [
        _strategy("ready_strat.py", "ready", status="draft", now=now),
        _strategy("draft_strat.py", "draft", status="draft", now=now),
    ]
    progress_fn = _fake_progress({"ready": READY_PROGRESS, "draft": DRAFT_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    assert directives[0].strategy_name == "ready_strat.py"
    assert directives[0].action == "PROMOTE"
    assert directives[0].priority == max(d.priority for d in directives)


def test_ready_and_already_live_is_not_promoted_again():
    now = time.time()
    strategies = [_strategy("live_strat.py", "ready", status="live", now=now)]
    progress_fn = _fake_progress({"ready": READY_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    assert all(d.action != "PROMOTE" for d in directives)


def test_marginal_maps_to_run_optimize():
    now = time.time()
    strategies = [_strategy("marginal.py", "marginal", now=now)]
    progress_fn = _fake_progress({"marginal": MARGINAL_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    assert directives[0].action == "RUN_OPTIMIZE"


def test_not_ready_maps_to_iterate_or_archive_and_scales_with_idle_time():
    now = time.time()
    fresh = _strategy("fresh_not_ready.py", "not_ready", modified_ts=now, now=now)
    stale = _strategy("stale_not_ready.py", "not_ready", modified_ts=now - 20 * 86400, now=now)
    progress_fn = _fake_progress({"not_ready": NOT_READY_PROGRESS})
    directives = ai_director.compute_directives(
        [fresh, stale], pipeline_progress_fn=progress_fn, now=now,
    )
    by_name = {d.strategy_name: d for d in directives}
    assert by_name["fresh_not_ready.py"].action == "ITERATE_OR_ARCHIVE"
    assert by_name["stale_not_ready.py"].priority > by_name["fresh_not_ready.py"].priority


def test_tested_but_never_optimized_maps_to_run_optimize():
    now = time.time()
    strategies = [_strategy("tested_only.py", "tested_only", now=now)]
    progress_fn = _fake_progress({"tested_only": TESTED_ONLY_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    assert directives[0].action == "RUN_OPTIMIZE"


def test_optimized_but_not_validated_maps_to_validate_further():
    now = time.time()
    strategies = [_strategy("optimized_only.py", "optimized_only", now=now)]
    progress_fn = _fake_progress({"optimized_only": OPTIMIZED_ONLY_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    assert directives[0].action == "VALIDATE_FURTHER"


def test_fresh_draft_never_backtested_produces_no_directive_yet():
    """A brand-new draft (idle 0 days, nothing computed on it yet) isn't
    worth nagging Owen about immediately -- it should still surface (not
    silently dropped), but at the lowest priority in the list."""
    now = time.time()
    strategies = [
        _strategy("brand_new_draft.py", "draft", modified_ts=now, now=now),
        _strategy("ready_strat.py", "ready", now=now),
    ]
    progress_fn = _fake_progress({"draft": DRAFT_PROGRESS, "ready": READY_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    names_in_order = [d.strategy_name for d in directives]
    assert names_in_order[0] == "ready_strat.py"
    assert "brand_new_draft.py" in names_in_order
    assert names_in_order[-1] == "brand_new_draft.py"


def test_market_alignment_bumps_priority_and_adds_reason_note():
    now = time.time()
    strategies = [
        _strategy("gold_strategy.py", "not_ready", market="XAUUSD", now=now),
        _strategy("other_strategy.py", "not_ready", market="EURUSD", now=now),
    ]
    progress_fn = _fake_progress({"not_ready": NOT_READY_PROGRESS})
    rankings = [
        {"symbol": "XAUUSD", "score": 85, "status": "READY"},
        {"symbol": "EURUSD", "score": 40, "status": "WAIT"},
    ]
    directives = ai_director.compute_directives(
        strategies, rankings=rankings, pipeline_progress_fn=progress_fn, now=now,
    )
    by_name = {d.strategy_name: d for d in directives}
    assert by_name["gold_strategy.py"].priority > by_name["other_strategy.py"].priority
    assert "XAUUSD" in by_name["gold_strategy.py"].reason


def test_market_alignment_alone_creates_market_aligned_directive():
    """A strategy with nothing else pending (fully Ready+live, i.e. truly
    done) but whose market is hot today still surfaces as a
    MARKET_ALIGNED nudge instead of disappearing entirely."""
    now = time.time()
    strategies = [_strategy("done_strategy.py", "done", status="live", market="XAUUSD", now=now)]
    progress_fn = _fake_progress({"done": DONE_PROGRESS})
    rankings = [{"symbol": "XAUUSD", "score": 90, "status": "READY"}]
    directives = ai_director.compute_directives(
        strategies, rankings=rankings, pipeline_progress_fn=progress_fn, now=now,
    )
    assert len(directives) == 1
    assert directives[0].action == "MARKET_ALIGNED"


def test_empty_library_returns_empty_list_never_raises():
    assert ai_director.compute_directives([]) == []
    assert ai_director.compute_directives(None) == []


def test_malformed_entries_are_skipped_not_fatal():
    now = time.time()
    good = _strategy("good.py", "ready", now=now)
    progress_fn = _fake_progress({"ready": READY_PROGRESS})
    bad_entries = [{"no_name_field": True}, None, 12345, object()]
    directives = ai_director.compute_directives(
        bad_entries + [good], pipeline_progress_fn=progress_fn, now=now,
    )
    assert len(directives) == 1
    assert directives[0].strategy_name == "good.py"


def test_top_n_truncates():
    now = time.time()
    progress_fn = _fake_progress({"ready": READY_PROGRESS})
    strategies = [_strategy(f"s{i}.py", "ready", now=now) for i in range(20)]
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now, top_n=5)
    assert len(directives) == 5


def test_accepts_dict_or_object_shaped_strategies():
    """StoredStrategy is an object with .name/.strategy_type/.status/
    .metadata/.modified attributes, not a dict -- compute_directives must
    accept either without special-casing by the caller."""
    now = time.time()

    class FakeStoredStrategy:
        name = "obj_shaped.py"
        strategy_type = "python"
        status = "draft"
        metadata = {"_stage_key": "ready"}
        modified = now

    progress_fn = _fake_progress({"ready": READY_PROGRESS})
    directives = ai_director.compute_directives([FakeStoredStrategy()], pipeline_progress_fn=progress_fn, now=now)
    assert len(directives) == 1
    assert directives[0].strategy_name == "obj_shaped.py"


def test_build_deterministic_briefing_empty_library():
    text = ai_director.build_deterministic_briefing([])
    assert "Nothing in the Strategy Library" in text


def test_build_deterministic_briefing_lists_every_directive():
    now = time.time()
    strategies = [_strategy("a.py", "ready", now=now), _strategy("b.py", "not_ready", now=now)]
    progress_fn = _fake_progress({"ready": READY_PROGRESS, "not_ready": NOT_READY_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    text = ai_director.build_deterministic_briefing(directives, {"total": 5, "by_verdict": {"KEEP": 3, "DISCARD": 2}})
    assert "a.py" in text and "b.py" in text
    assert "5 experiments recorded" in text


def test_build_director_prompt_includes_full_payload_and_never_reorders_instruction():
    now = time.time()
    strategies = [_strategy("a.py", "ready", now=now)]
    progress_fn = _fake_progress({"ready": READY_PROGRESS})
    directives = ai_director.compute_directives(strategies, pipeline_progress_fn=progress_fn, now=now)
    prompt = ai_director.build_director_prompt(directives, {"total": 1, "by_verdict": {"KEEP": 1}})
    assert "a.py" in prompt
    assert "do not add, remove, or reorder" in prompt.lower()


def test_score_one_never_raises_on_missing_metadata_fields():
    now = time.time()
    progress_fn = _fake_progress({"ready": READY_PROGRESS})
    d = ai_director.score_one(
        name="minimal.py", strategy_type="python", status="draft", metadata={"_stage_key": "ready"},
        modified_ts=now, now=now, hot_markets={}, pipeline_progress_fn=progress_fn,
    )
    assert d.strategy_name == "minimal.py"
    assert d.market is None


# ---------------------------------------------------------------------------
# Web route smoke tests
# ---------------------------------------------------------------------------


def test_director_route_works_with_empty_library_and_ollama_disabled():
    from app.ai.ollama_settings import OllamaSettings

    from app.web.server import app as flask_app
    import app.web.ai_assistant_routes as routes

    def _fake_load_settings():
        return OllamaSettings(enabled=False, host="", model="", api_key="")

    orig = routes.load_ollama_settings
    routes.load_ollama_settings = _fake_load_settings
    try:
        client = flask_app.test_client()
        resp = client.get("/assistant/api/director")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "directives" in data
        assert "memory_counts" in data
        assert data["error"] is None
        assert "Ollama isn't enabled" in data["text"]
    finally:
        routes.load_ollama_settings = orig


def test_director_stream_route_reports_ollama_disabled_without_crashing():
    from app.ai.ollama_settings import OllamaSettings

    from app.web.server import app as flask_app
    import app.web.ai_assistant_routes as routes

    def _fake_load_settings():
        return OllamaSettings(enabled=False, host="", model="", api_key="")

    orig = routes.load_ollama_settings
    routes.load_ollama_settings = _fake_load_settings
    try:
        client = flask_app.test_client()
        resp = client.post("/assistant/api/director/stream", json={})
        assert resp.status_code == 200
        body = b"".join(resp.response).decode()
        assert "Ollama isn't enabled" in body
    finally:
        routes.load_ollama_settings = orig
