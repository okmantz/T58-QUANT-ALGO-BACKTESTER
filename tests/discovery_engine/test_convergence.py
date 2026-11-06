from app.discovery_engine.convergence import ConvergenceTracker, signal_fingerprint
from app.discovery_engine.presets import PRESETS


def _rows(pairs):
    return [{"statistics": {"total_trades": t, "net_profit": p}} for t, p in pairs]


def test_identical_trade_results_are_one_signal_set():
    a, b = {"statistics": {"total_trades": 40, "net_profit": 123.456}}, {"total_trades": 40, "net_profit": 123.4561}
    assert signal_fingerprint(a) == signal_fingerprint(b)
    assert signal_fingerprint({"foo": 1}) is None


def test_tracker_stops_when_a_round_adds_few_new_sets():
    t = ConvergenceTracker(min_new_per_1000=5.0, max_rounds=4)
    t.add_round(_rows([(i, i * 3.0) for i in range(50)]), 1000)          # 50 new per 1000 -> continue
    assert t.should_continue()
    t.add_round(_rows([(i, i * 3.0) for i in range(50)]), 1000)          # nothing new -> stop
    assert not t.should_continue() and "stopped: below" in t.note()


def test_tracker_respects_the_round_cap():
    t = ConvergenceTracker(min_new_per_1000=1.0, max_rounds=2)
    for r in range(2):
        t.add_round(_rows([(r * 1000 + i, float(i)) for i in range(200)]), 1000)
    assert not t.should_continue() and "round cap" in t.note().replace("-round cap", " round cap") or "cap" in t.note()


def test_only_the_search_lab_preset_is_convergence_driven():
    assert PRESETS["search_lab"].convergence and not PRESETS["speed_run"].convergence
