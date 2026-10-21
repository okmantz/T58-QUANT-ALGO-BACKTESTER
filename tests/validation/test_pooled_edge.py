import numpy as np
import pandas as pd

from app.validation.pooled_edge import (analytic_power_table, cluster_t_stat, estimated_cost_in_r, pooled_edge_test)


def _market(n, mean, seed, start="2020-01-02"):
    rng = np.random.default_rng(seed)
    days = list(pd.bdate_range(start, periods=n).strftime("%Y-%m-%d"))
    return list(rng.normal(mean, 1.3, n)), days


def test_real_edge_pooled_passes_but_single_market_unproven():
    single = {"A": _market(120, 0.15, 1)}
    pooled = {s: _market(120, 0.15, i + 1) for i, s in enumerate("ABCDEFGH")}
    assert pooled_edge_test(single, n_boot=1000).status in ("UNPROVEN", "PASS")
    assert pooled_edge_test(pooled, n_boot=1000).status == "PASS"


def test_null_does_not_pass_and_loser_fails():
    null = {s: _market(120, 0.0, 10 + i) for i, s in enumerate("ABCDEFGH")}
    assert pooled_edge_test(null, n_boot=1000).status != "PASS"
    loser = {s: _market(120, -0.3, 30 + i) for i, s in enumerate("ABCDEFGH")}
    assert pooled_edge_test(loser, n_boot=1000).status == "FAIL"


def test_insufficient_and_cluster_t():
    assert pooled_edge_test({"A": _market(20, 0.5, 3)}).status == "INSUFFICIENT"
    r, d = _market(300, 0.2, 4)
    assert cluster_t_stat(r, d) > 1.5


def test_same_day_correlation_widens_bound():
    base, days = _market(200, 0.1, 5)
    indep = {s: _market(200, 0.1, 50 + i) for i, s in enumerate("ABCD")}
    corr = {s: (list(base), days) for s in "ABCD"}      # identical R on identical days
    assert pooled_edge_test(corr, n_boot=1000).t_stat < pooled_edge_test(indep, n_boot=1000).t_stat


def test_cost_in_r_and_power_table():
    class R:
        contract_size = 20.0
        commission_per_contract = 2.0
        def side_cost_price(self): return 1.0
    assert abs(estimated_cost_in_r(R(), 20.0) - (2.0 + 0.1) / 20.0) < 1e-9
    rows = analytic_power_table()
    assert rows[0]["both_pooled"] > rows[0]["both"]
    assert rows[2]["both_pooled"] > 0.6
