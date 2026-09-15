from app.scoring.parsimony import compute_parsimony, dof_to_score
from app.strategy.manual import ManualStrategy


def test_dof_to_score_zero_is_max_and_decreasing():
    assert dof_to_score(0) == 100.0
    assert dof_to_score(2) > dof_to_score(6) > dof_to_score(20)
    assert dof_to_score(1000) >= 0.0


def test_manual_strategy_counts_indicators_and_visual_conditions():
    config = {
        "indicators": [{"type": "ema", "period": 20}, {"type": "rsi", "period": 14}],
        "entry_conditions": {"long": [{"a": 1}, {"a": 2}], "short": [{"a": 1}]},
        "exit_conditions": {"long": [{"a": 1}]},
    }
    result = compute_parsimony(ManualStrategy(config))
    assert result.degrees_of_freedom == 2 + 3 + 1  # 2 indicators + 3 entry conditions + 1 exit condition
    assert result.score is not None
    assert result.notes


def test_manual_strategy_legacy_expression_counts_and_or_terms():
    config = {
        "indicators": [],
        "long_entry": "sma_fast > sma_slow and rsi < 30",
        "short_entry": "",
    }
    result = compute_parsimony(ManualStrategy(config))
    assert result.degrees_of_freedom == 2  # one 'and' boundary -> two terms


def test_simpler_strategy_scores_higher_than_more_complex_one():
    simple = ManualStrategy({"indicators": [{"type": "ema", "period": 20}], "long_entry": "close > ema_20"})
    complex_ = ManualStrategy({
        "indicators": [{"type": "ema", "period": 20}, {"type": "rsi", "period": 14}, {"type": "atr", "period": 14}],
        "entry_conditions": {"long": [{"a": 1}, {"a": 2}, {"a": 3}], "short": [{"a": 1}]},
    })
    r_simple = compute_parsimony(simple)
    r_complex = compute_parsimony(complex_)
    assert r_simple.score > r_complex.score


def test_uncountable_strategy_type_returns_none_score_not_zero():
    class _WeirdStrategy:
        source_type = "manual"
        config = "not a dict"  # malformed on purpose

    result = compute_parsimony(_WeirdStrategy())
    assert result.score is None
    assert result.notes
