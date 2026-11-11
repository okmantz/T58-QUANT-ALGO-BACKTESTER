"""2026-11-04: the journey Create -> Optimize -> Validate -> Champion -> Forward test -> Deploy tells the person where they are and what is next."""
from app.scoring.champion_board import evaluate_promotion, row_next_step
from app.strategy.library import VALIDATION_STEP_METHODS


def _meta(**over):
    md = {"last_run": {"eval_pass_probability": 75, "first_payout_probability": 65}, "last_validation": {"efficiency": 0.8, "is_robust": True},
          "validation_methods_run": {m: {} for m in VALIDATION_STEP_METHODS}, "last_champion_check": {"verdict": "MARGINAL"}}
    md.update(over)
    return md


def test_fully_validated_strategy_is_promoted_straight_to_champion_in_one_step():
    st = evaluate_promotion(_meta()).to_dict()
    assert st["can_promote"] and st["next_stage"] == "champion" and st["next_stage_title"] == "Champion"
    assert st["path"] == ["champion"]


def test_promotion_is_blocked_when_champion_thresholds_are_not_met_and_never_offers_validated():
    st = evaluate_promotion(_meta(last_run={"eval_pass_probability": 30, "first_payout_probability": 20})).to_dict()
    assert not st["can_promote"] and st["next_stage"] == "champion" and "validated" not in st["path"]
    assert any(r["key"] == "eval_threshold" and not r["met"] for r in st["requirements"])


def test_promotion_never_enters_forward_testing():
    md = _meta(promotion={"stage": "candidate", "history": {}}, last_champion_check={"verdict": "READY"})
    st = evaluate_promotion(md).to_dict()
    assert st["next_stage"] == "champion" and "forward_testing" not in st["path"]


def test_forward_test_candidate_ladder_stage_unlocks_forward_testing_even_when_verdict_is_marginal():
    md = _meta(promotion={"stage": "champion", "history": {}}, last_run={"eval_pass_probability": 75},
               last_champion_check={"verdict": "MARGINAL", "ladder_stage": "FORWARD-TEST CANDIDATE"})
    assert evaluate_promotion(md).can_promote
    md["last_champion_check"] = {"verdict": "MARGINAL"}
    assert not evaluate_promotion(md).can_promote


def test_board_row_says_what_one_press_does_and_where_to_go_next():
    row = {"strategy_type": "python", "filename": "x.py", "promotion": evaluate_promotion(_meta()).to_dict()}
    nxt = row_next_step(row)
    assert nxt["kind"] == "promote" and "Promote to Champion" in nxt["text"] and "Validated" not in nxt["text"]


def test_deploy_preflight_names_the_stage_and_a_stale_rules_stamp_needs_acknowledgement():
    from app.live_deploy.connect_flow import preflight_checklist
    from app.live_deploy.live_settings import LiveAccount
    acct = LiveAccount(id="a", platform="MT4/MT5", firm_name="Topstep", nickname="t", login="1", server="s")
    out = preflight_checklist(acct, strategy_meta=_meta(), preset_key="topstep_50k")
    keys = {i["key"]: i for i in out["items"]}
    assert "stage" in keys and "forward" in keys["stage"]["detail"].lower()
    if keys.get("rules", {}).get("level") == "ack":
        assert out["needs_ack"] and "re-check" in keys["rules"]["title"]
