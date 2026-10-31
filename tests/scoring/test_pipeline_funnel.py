"""Dashboard "Pipeline in motion": Candidates -> Tested -> Optimized -> Validated -> Champion Candidate -> Forward Tested -> Production Ready."""
from app.scoring import champion_board as cb
from app.strategy.library import VALIDATION_STEP_METHODS


def _meta(**kw):
    return dict(kw)


def _all_validation():
    return {m: {"recorded_at": 1.0, "efficiency": 0.7} for m in VALIDATION_STEP_METHODS}


STRONG_RUN = {"eval_pass_probability": 75.0, "first_payout_probability": 70.0}


def test_stage_order_and_titles_match_the_dashboard_spec():
    assert [cb.FUNNEL_TITLES[k] for k in cb.FUNNEL_STAGES] == [
        "Candidates", "Tested", "Optimized", "Validated", "Champion Candidate", "Forward Tested", "Production Ready"]


def test_rank_climbs_with_what_is_on_record():
    assert cb.funnel_rank({}) == 0
    assert cb.funnel_rank(_meta(last_run=STRONG_RUN)) == 1
    assert cb.funnel_rank(_meta(last_run=STRONG_RUN, last_optimize={"improved": True})) == 2


def test_passing_validation_makes_a_champion_candidate_without_clicking_promote():
    m = _meta(last_run=STRONG_RUN, validation_methods_run=_all_validation(), last_validation={"efficiency": 0.7},
              last_champion_check={"verdict": "READY"})
    assert cb.funnel_rank(m) == 4                      # validated AND meets the Eval/Payout/OOS thresholds -> Champion Candidate


def test_validation_that_does_not_meet_the_gate_stays_validated():
    weak = {"eval_pass_probability": 20.0, "first_payout_probability": 10.0}
    m = _meta(last_run=weak, validation_methods_run=_all_validation(), last_validation={"efficiency": 0.7})
    assert cb.funnel_rank(m) == 3


def test_a_not_ready_verdict_is_never_a_champion_candidate():
    m = _meta(last_run=STRONG_RUN, validation_methods_run=_all_validation(), last_validation={"efficiency": 0.7},
              last_champion_check={"verdict": "NOT READY"})
    assert cb.funnel_rank(m) == 3


def test_already_promoted_strategies_count_at_their_promoted_stage_or_beyond():
    for stage, want in (("validated", 4), ("champion_candidate", 4), ("forward_testing", 5), ("production_ready", 6)):
        assert cb.funnel_rank(_meta(promotion={"stage": stage, "history": {}})) == want, stage


def test_a_forward_test_on_record_makes_it_forward_tested():
    assert cb.funnel_rank(_meta(last_run=STRONG_RUN, last_forward_test={"trades": 3})) == 5


def test_flow_counts_never_rise_and_first_node_is_every_strategy():
    rows = [
        {"filename": "a.json", "funnel_rank": 0},
        {"filename": "b.json", "funnel_rank": 1},
        {"filename": "c.json", "funnel_rank": 4},
        {"filename": "d.json", "funnel_rank": 5},
        {"filename": "e.json", "funnel_rank": 6},
    ]
    nodes = cb.pipeline_flow_nodes(rows, tested_files={"a.json"}, run_only=2)
    ns = [n["n"] for n in nodes]
    assert [n["title"] for n in nodes][0] == "Candidates" and ns[0] == 5 + 2          # total strategies, including the unsaved ones
    assert ns == sorted(ns, reverse=True)                                              # a funnel: never more at a later step
    assert ns[1] == 5 + 2                                                              # 'a' counts as tested via its run history; run-only ones are tested
    assert ns[-1] == 1 and ns[5] == 2 and ns[4] == 3


def test_board_rows_carry_the_rank(tmp_path, monkeypatch):
    rows = cb.list_board()
    assert all("funnel_rank" in r for r in rows)
