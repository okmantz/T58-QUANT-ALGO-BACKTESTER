"""Oct-2026 audit: Full Pipeline form -> engine inputs (preset rules, instrument-aware costs)."""
from werkzeug.datastructures import MultiDict

from app.backtest.risk import RiskConfig
from app.web import server


def test_posted_preset_brings_the_firms_full_rules():
    typed = server.PropRules(account_size=50000.0)
    got = server._apply_posted_preset(MultiDict({"prop_preset": "topstep_50k"}), typed)
    assert got.dd_basis == "eod" and got.consistency_rule_pct == 50.0
    assert got.account_size == 50000.0


def test_no_preset_keeps_typed_rules_unchanged():
    typed = server.PropRules(account_size=50000.0)
    assert server._apply_posted_preset(MultiDict({}), typed) is typed
    assert server._apply_posted_preset(MultiDict({"prop_preset": "no_such_firm"}), typed) is typed


def _risk(commission=0.0):
    return RiskConfig(initial_balance=50000.0, pip_size=1.0, contract_size=20.0, commission_per_trade=commission,
                      slippage_pips=server._posted_cost(MultiDict({}), "slippage_pips"),
                      spread_pips=server._posted_cost(MultiDict({}), "spread_pips"))


def test_unposted_costs_use_instrument_spec_not_fx_defaults():
    r = server._fill_unposted_costs(MultiDict({}), _risk(4.2), "NQ1! parquet")
    assert (r.spread_pips, r.slippage_pips) == (0.25, 0.25)           # 1 tick each side, not 1.5 points
    assert r.commission_per_trade == 0.0 and r.commission_per_contract == 4.2   # no double billing


def test_typed_costs_are_never_overridden_and_unknown_instruments_keep_legacy_defaults():
    r = server._fill_unposted_costs(MultiDict({"slippage_pips": "0"}), _risk(), "NQ1!")
    assert (r.spread_pips, r.slippage_pips) == (0.0, 0.0)
    r2 = server._fill_unposted_costs(MultiDict({}), _risk(), "totally-unknown-feed")
    assert (r2.spread_pips, r2.slippage_pips) == (1.0, 0.5)


def test_integrity_check_accepts_per_contract_commission():
    import pandas as pd
    from app.strategy.manual import ManualStrategy
    from app.validation.integrity_check import run_integrity_check
    ts = pd.date_range("2024-01-02 14:30", periods=400, freq="5min", tz="UTC")
    px = 18000 + pd.Series(range(400)).astype(float)
    df = pd.DataFrame({"timestamp": ts, "open": px, "high": px + 1, "low": px - 1, "close": px, "volume": 1})
    base = dict(initial_balance=50000.0, pip_size=1.0, contract_size=20.0, slippage_pips=0.25, spread_pips=0.25)
    strat = ManualStrategy({"indicators": [], "long_entry": "close > 0", "stop_loss_pips": 10, "take_profit_pips": 20})
    bad = run_integrity_check(df, strat, RiskConfig(**base), data_label="NQ1!")
    good = run_integrity_check(df, strat, RiskConfig(**base, commission_per_contract=4.2), data_label="NQ1!")
    assert any("Commission" in str(l) and not getattr(l, "ok", True) for l in getattr(bad, "lines", [])) or bad.status == "BLOCKED"
    assert good.status != "BLOCKED"
