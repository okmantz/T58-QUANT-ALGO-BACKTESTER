from __future__ import annotations

import numpy as np
import pandas as pd

from app.backtest.engine import run_backtest
from app.backtest.risk import RiskConfig
from app.strategy.manual import ManualStrategy
from app.strategy.zones import build_zones, check_zone_causality, zone_orders_from_config

CFG_ZONE = {"kind": "fvg", "atr_period": 3, "min_size_atr": 0.3, "expiry_bars": 20, "stop_buffer_atr": 0.25, "target_r": 2.0}


def _hand_day(touch_low=101.8):
    rows = [(100, 100.5, 99.5, 100)] * 10
    rows += [
        (100, 100.4, 99.6, 100.2),      # 10  candle i-2
        (100.2, 103.0, 100.1, 102.9),   # 11  impulse
        (102.9, 103.5, 102.0, 103.2),   # 12  candle i -> gap [100.4, 102.0], known at close of 12
        (103.2, 103.6, 102.5, 103.0),   # 13  no touch
        (103.0, 103.1, touch_low, 102.2),  # 14  retrace through 102.0
        (102.2, 104.0, 102.0, 103.9),   # 15
        (103.9, 106.5, 103.8, 106.4),   # 16  target 106.0667
        (106.4, 106.6, 106.0, 106.5),   # 17
    ]
    df = pd.DataFrame(rows, columns=["open", "high", "low", "close"])
    df.insert(0, "timestamp", pd.date_range("2024-01-02 15:00", periods=len(df), freq="15min"))
    df["volume"] = 1
    return df


def _risk():
    return RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=1000.0, pip_size=0.01, contract_size=1.0,
                      spread_pips=1.0, slippage_pips=1.0, commission_per_contract=0.0, max_trades_per_day=10)


def test_fvg_zone_is_known_only_at_the_close_of_its_third_candle():
    z = build_zones(_hand_day(), "fvg", atr_period=3)
    assert z[0].created_bar == 12 and z[0].side == 1 and all(x.created_bar >= 12 for x in z)
    assert (z[0].near, z[0].far) == (102.0, 100.4)


def test_hand_checked_fvg_retrace_fills_and_dollars():
    df = _hand_day()
    strat = ManualStrategy({"name": "fvg retest", "zone_entry": CFG_ZONE})
    res = run_backtest(df, strat, _risk())
    assert len(res.trades) == 1
    t = res.trades[0]
    atr = (0.8 + 2.9 + 1.5) / 3
    stop = 100.4 - 0.25 * atr
    tgt = 102.0 + 2.0 * (102.0 - stop)
    assert t.entry_time == df["timestamp"][14] and abs(t.entry_price - 102.0) < 1e-9
    assert t.exit_time == df["timestamp"][16] and t.exit_reason == "take_profit" and abs(t.exit_price - tgt) < 1e-9
    assert abs(t.pnl - t.size * (tgt - 102.0)) < 1e-6
    # nothing before bar 14 and the zone never produces a market-order signal
    assert res.equity_curve["equity"].iloc[13] == 50_000.0


def test_touch_without_trade_through_does_not_fill():
    df = _hand_day(touch_low=102.0)
    res = run_backtest(df, ManualStrategy({"name": "x", "zone_entry": CFG_ZONE}), _risk())
    assert res.trades == []


def test_zone_orders_are_causal_on_random_walk_and_a_leak_is_caught(monkeypatch):
    rng = np.random.default_rng(3)
    n = 900
    px = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame(dict(timestamp=pd.date_range("2024-01-02", periods=n, freq="15min"), open=px, high=px + rng.random(n), low=px - rng.random(n), close=px, volume=1))
    cfg = dict(CFG_ZONE, atr_period=14)
    assert check_zone_causality(cfg, df) == []
    import app.strategy.zones as zmod
    real = zmod.zone_orders_from_config

    def leaky(d, c):
        out = real(d, c)
        for o in out:                      # peeks at the LAST bar of whatever history it is given
            o.stop_price = o.stop_price + 1e-3 * float(d["close"].iloc[-1])
        return out
    monkeypatch.setattr(zmod, "zone_orders_from_config", leaky)
    assert check_zone_causality(cfg, df)


def test_fast_path_refuses_resting_order_strategies():
    from app.backtest.vectorized_fastpath import is_vectorizable
    res = ManualStrategy({"name": "x", "zone_entry": CFG_ZONE}).generate(_hand_day())
    assert not is_vectorizable(res)


def test_translator_renders_limit_orders_not_market_orders():
    from app.strategy.translator import to_mql5, to_pinescript
    cfg = {"name": "FVG retest", "zone_entry": CFG_ZONE}
    pine, mql = to_pinescript(cfg), to_mql5(cfg)
    assert "limit=near" in pine and "strategy.cancel" in pine and "barstate.isconfirmed" in pine
    assert "BuyLimit" in mql and "SellLimit" in mql and "ORDER_TIME_SPECIFIED" in mql
    assert "order_block" not in pine
    ob = to_pinescript({"name": "x", "zone_entry": dict(CFG_ZONE, kind="order_block")})
    assert "not translated" in ob


def test_paper_claims_become_cited_deduped_hypotheses_and_link_to_memory(tmp_path, monkeypatch):
    from app.ai.llm_client import ScriptedClient
    from app.discovery.hypothesis import HypothesisStore
    from app.discovery.paper_hypotheses import extract_hypotheses
    store = HypothesisStore(tmp_path / "h.json")
    reply = '[{"claim":"Does a breakout of the 40-bar high keep going when volatility is high?","conditions":"high vol","expected_sign":"positive","regime":"high volatility"},' \
            '{"claim":"Do analyst upgrades lead returns for small caps?","conditions":"","expected_sign":"unclear","regime":""}]'
    hs = extract_hypotheses("text", llm=ScriptedClient(reply), store=store, source_file="momentum/jt93.pdf", chunk_index=7)
    assert len(hs) == 2 and all(h.source == "paper:momentum/jt93.pdf#7" for h in hs)
    assert hs[0].spec and not hs[1].spec and "not yet testable" in hs[1].warnings[0]
    again = extract_hypotheses("text", llm=ScriptedClient(reply), store=store, source_file="momentum/jt93.pdf")
    assert again == []                                  # near-duplicates are merged, not stored twice
    mined = extract_hypotheses("Momentum persists after earnings surprises in the following quarter according to the sample. Other text.", store=HypothesisStore(tmp_path / "m.json"))
    assert mined and mined[0].source.startswith("paper:")


def test_experiment_memory_links_rows_to_hypothesis(tmp_path, monkeypatch):
    import app.ai.experiment_memory as em
    monkeypatch.setattr(em, "_db_path", lambda: tmp_path / "e.db")
    eid = em.record_experiment(origin="discovery", strategy_name="x", hypothesis_id="h123", cell="NQ/15min", verdict="BROKEN",
                               settings=type("S", (), {"is_usable": False})())
    assert eid
    rows = em.already_tested("h123")
    assert len(rows) == 1 and rows[0]["cell"] == "NQ/15min"
    assert em.already_tested("h123", cell="ES/1h") == [] and em.already_tested("nope") == []
