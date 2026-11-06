"""Trade copier: one strategy on a leader account, every trade mirrored onto follower accounts.

Covers sizing per follower, the pre-start group checks, the fan-out (open / close / retry), failure
isolation (one follower halting does not stop the others; a leader halt flattens everyone), all-or-nothing
start, the bar-feed fallback for platforms with no bars (Tradovate), and the per-platform account pick that
makes several accounts under one login addressable.
"""
from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from app.backtest.risk import RiskConfig
from app.forward_test.journal import ForwardTestJournal
from app.live_deploy.broker_base import BrokerAdapter, ConnectionResult, OpenPosition, OrderResult
from app.live_deploy.broker_registry import adapter_class_for, build_adapter
from app.live_deploy.broker_tradelocker import TradeLockerBrokerAdapter
from app.live_deploy.broker_tradovate import TradovateBrokerAdapter
from app.live_deploy.execution_engine import LiveExecutionConfig, LiveExecutionSession
from app.live_deploy.guardrails import futures_prop_safe_rules
from app.live_deploy.live_settings import LiveAccount
from app.live_deploy.trade_copier import (
    FollowerSpec, SIZE_FIXED, SIZE_MULTIPLIER, SIZE_RISK, build_copier, compute_follower_units,
    parse_follower_specs, validate_group,
)


# ------------------------------------------------------------------ fakes

class _Strategy:
    """Signal is switchable; 10-point stop."""

    def __init__(self):
        self.sig = 0

    def generate(self, df):
        n = len(df)
        return SimpleNamespace(
            signals=pd.Series([self.sig] * n, index=df.index),
            stop_loss_distance=pd.Series([10.0] * n, index=df.index),
            stop_loss_pips=None, take_profit_distance=None, take_profit_pips=None,
        )


class _Broker(BrokerAdapter):
    """In-memory broker with a position book. Whole contracts, like Tradovate."""

    platform_name = "Fake"

    def __init__(self, name, equity=50_000.0, bars=True, stops=True, fail_connect=False):
        self.name = name
        self.equity = equity
        self.supports_bars = bars
        self.supports_broker_stops = stops
        self.fail_connect = fail_connect
        self.alive = True
        self.fail_close = False
        self.connected = False
        self.tick = 0
        self.positions: list[OpenPosition] = []
        self.orders: list[tuple] = []
        self.flatten_calls = 0
        self._n = 0

    def connect(self):
        if self.fail_connect:
            return ConnectionResult(ok=False, message=f"{self.name} refused the login")
        self.connected = True
        return ConnectionResult(ok=True, message="ok", account_login=self.name, account_server="fake",
                                balance=self.equity, equity=self.equity)

    def disconnect(self):
        self.connected = False

    def is_alive(self):
        return self.alive

    def ensure_connected(self, *a, **k):
        return ConnectionResult(ok=True, message="ok") if self.alive else ConnectionResult(ok=False, message="link down")

    def account_summary(self):
        return {"balance": self.equity, "equity": self.equity}

    def fetch_completed_bars(self, symbol, timeframe_minutes, count):
        if not self.supports_bars:
            raise ValueError("no bars on this platform")
        ts = pd.date_range("2026-10-01 00:00", periods=10, freq="15min", tz="UTC") + pd.Timedelta(minutes=15 * self.tick)
        return pd.DataFrame({"timestamp": ts, "open": 5000.0, "high": 5001.0, "low": 4999.0, "close": 5000.0, "volume": 1.0})

    def get_open_positions(self, symbol=None):
        return [p for p in self.positions if symbol is None or p.symbol == symbol]

    def place_market_order(self, symbol, direction, volume, sl_price=None, tp_price=None, comment="", deviation=20):
        self._n += 1
        ticket = f"{self.name}-{self._n}"
        self.orders.append((symbol, direction, volume, sl_price, tp_price))
        self.positions.append(OpenPosition(ticket=ticket, symbol=symbol, direction=direction, volume=volume,
                                           open_price=5000.0, sl=sl_price or 0.0, tp=tp_price or 0.0,
                                           open_time=pd.Timestamp.now(tz="UTC"), profit=0.0))
        return OrderResult(ok=True, message="ok", ticket=ticket, price=5000.0, volume=volume)

    def close_position(self, ticket, comment=""):
        if self.fail_close:
            return OrderResult(ok=False, message="exchange rejected the close")
        self.positions = [p for p in self.positions if p.ticket != ticket]
        return OrderResult(ok=True, message="closed", ticket=ticket, price=5002.0)

    def close_all(self, symbol=None):
        self.flatten_calls += 1
        out = [OrderResult(ok=True, message="flat", ticket=p.ticket) for p in self.get_open_positions(symbol)]
        self.positions = [p for p in self.positions if symbol is not None and p.symbol != symbol]
        return out

    def to_broker_qty(self, units, contract_size, units_per_lot):
        import math
        if not contract_size or contract_size <= 0:
            return 0.0
        return float(math.floor(units / contract_size + 1e-9))


def _acct(name, platform="Tradovate", **extra):
    return LiveAccount(id=name, nickname=name, firm_name="", platform=platform, login=name, server="", password="pw",
                       extra_credentials=extra)


def _risk():
    return RiskConfig(initial_balance=50_000.0, risk_mode="percent", risk_value=1.0, pip_size=1.0, contract_size=50.0,
                      spread_pips=0.0, slippage_pips=0.0, commission_per_trade=0.0)


class _Rig:
    """A leader session plus followers, driven poll by poll without the background thread."""

    def __init__(self, tmp_path, followers, *, leader_equity=50_000.0, leader_bars=True, halt_group=True, bar_from=None):
        self.strategy = _Strategy()
        self.leader = _Broker("leader", equity=leader_equity, bars=leader_bars, stops=leader_bars)
        self.brokers = {name: broker for name, _spec, broker in followers}
        accounts = {name: _acct(name) for name, _s, _b in followers}
        specs = [spec for _n, spec, _b in followers]
        self.journal = ForwardTestJournal(db_path=tmp_path / "j.db")
        rules = futures_prop_safe_rules(50_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=10.0)
        self.copier = build_copier(
            specs=specs, accounts_by_id=accounts, adapter_factory=lambda a: self.brokers[a.id], leader_symbol="ES",
            timeframe_minutes=15, leader_risk=_risk(), leader_rules=rules, journal=self.journal,
            strategy_type="manual", strategy_filename="long", halt_group_on_leader_halt=halt_group,
            connect_stagger_seconds=0.0,
        )
        cfg = LiveExecutionConfig(symbol="ES", timeframe_minutes=15, risk=_risk(), prop_rules=rules,
                                  contract_size=50.0, units_per_lot=50.0, journal_login="leader")
        self.session = LiveExecutionSession(
            strategy=self.strategy, strategy_type="manual", strategy_filename="long", broker=self.leader,
            journal=self.journal, config=cfg, copier=self.copier,
            bar_source=(self.brokers[bar_from] if bar_from else None), bar_symbol=("ES" if bar_from else None),
        )

    def begin(self, leader_direction=0):
        s = self.session
        conn = self.leader.connect()
        s.status.connected = True
        s.status.balance, s.status.equity = conn.balance, conn.equity
        s._session_id = self.journal.start_session("manual", "long", "ES", 15, "leader", "fake")
        s._dd_peak = conn.equity
        ok, msg = self.copier.start(leader_direction=leader_direction)
        assert ok, msg
        s._sync_followers()
        s._stop_flag.clear()
        s.status.running = True
        return self

    def poll(self, sig=None):
        if sig is not None:
            self.strategy.sig = sig
        for b in [self.leader, *self.brokers.values()]:
            b.tick += 1
        self.session._poll_once()


def _spec(name, **kw):
    return FollowerSpec(account_id=name, **kw)


# ------------------------------------------------------------------ sizing

def test_risk_mode_sizes_on_the_follower_accounts_own_equity():
    risk = _risk()
    # $25k at 1% = $250 over a 10-pt stop = 25 units; MES contract = 5 units -> 25/5 = 5 contracts
    u = compute_follower_units(mode=SIZE_RISK, value=0, risk=_replace(risk, initial_balance=25_000.0, contract_size=5.0),
                               equity=25_000.0, stop_distance=10.0, leader_units=50.0, contract_size=5.0, units_per_lot=5.0)
    assert u == 25.0
    # $100k -> $1000 / 10 pts = 100 units = 2 ES
    u2 = compute_follower_units(mode=SIZE_RISK, value=0, risk=_replace(risk, initial_balance=100_000.0), equity=100_000.0,
                                stop_distance=10.0, leader_units=50.0, contract_size=50.0, units_per_lot=50.0)
    assert u2 == 100.0


def _replace(risk, **kw):
    import dataclasses
    return dataclasses.replace(risk, **kw)


def test_multiplier_scales_leader_dollar_exposure_across_contract_sizes():
    # Leader holds 1 ES = 50 units. x1 onto an MES account (5 units/contract) is 10 MES, not 1.
    u = compute_follower_units(mode=SIZE_MULTIPLIER, value=1.0, risk=_risk(), equity=1.0, stop_distance=10.0,
                               leader_units=50.0, contract_size=5.0, units_per_lot=5.0)
    assert u == 50.0
    assert compute_follower_units(mode=SIZE_MULTIPLIER, value=0.5, risk=_risk(), equity=1.0, stop_distance=10.0,
                                  leader_units=50.0, contract_size=5.0, units_per_lot=5.0) == 25.0


def test_fixed_mode_is_native_contracts():
    assert compute_follower_units(mode=SIZE_FIXED, value=3, risk=_risk(), equity=1.0, stop_distance=10.0, leader_units=50.0,
                                  contract_size=5.0, units_per_lot=5.0) == 15.0


def test_spec_parsing_validates_input():
    assert parse_follower_specs([{"account_id": "a", "size_mode": "fixed", "size_value": "2"}])[0].size_value == 2.0
    assert parse_follower_specs("") == []
    with pytest.raises(ValueError):
        parse_follower_specs([{"account_id": "a", "size_mode": "yolo"}])
    with pytest.raises(ValueError):
        parse_follower_specs([{"account_id": "a", "size_value": "lots"}])


# ------------------------------------------------------------------ group preflight

def _pre(leader, specs, accounts, symbol="MNQ"):
    return validate_group(leader, specs, {a.id: a for a in accounts}, leader_symbol=symbol, adapter_class_for=adapter_class_for)


def _levels(res, key):
    return [i["level"] for i in res["items"] if i["key"] == key]


def test_preflight_blocks_without_leader_or_followers():
    a = _acct("a", "cTrader", client_id="x", client_secret="y", refresh_token="z", ctid_trader_account_id="1")
    assert not _pre(None, [_spec("a")], [a])["can_start"]
    assert not _pre(a, [], [a])["can_start"]


def test_preflight_two_mt5_accounts_blocked_by_the_single_terminal_rule():
    leader = _acct("m1", "MT4/MT5"); leader.server = "S"
    other = _acct("m2", "MT4/MT5"); other.server = "S"
    res = _pre(leader, [_spec("m2")], [leader, other], symbol="US100")
    assert "block" in _levels(res, "mt5_single") and not res["can_start"]


def test_preflight_shared_tradovate_login_needs_an_account_id_each():
    common = dict(app_id="a", app_secret="b", cid="c", sec="d")
    l1 = _acct("lucid1", **common); l1.login = "owen"
    l2 = _acct("lucid2", **common); l2.login = "owen"
    ft = _acct("ftmo", "cTrader", client_id="x", client_secret="y", refresh_token="z", ctid_trader_account_id="9")
    res = _pre(l1, [_spec("lucid2"), _spec("ftmo")], [l1, l2, ft])
    assert "block" in _levels(res, "account_pick") and not res["can_start"]
    l1.extra_credentials["account_id"] = "LUCID-1"
    l2.extra_credentials["account_id"] = "LUCID-2"
    res2 = _pre(l1, [_spec("lucid2"), _spec("ftmo")], [l1, l2, ft])
    assert "block" not in _levels(res2, "account_pick") and "block" not in _levels(res2, "same_account")


def test_preflight_same_underlying_account_twice_is_blocked():
    common = dict(app_id="a", app_secret="b", cid="c", sec="d", account_id="LUCID-1")
    a = _acct("one", **common); a.login = "owen"
    b = _acct("two", **common); b.login = "owen"
    res = _pre(a, [_spec("two")], [a, b])
    assert "block" in _levels(res, "same_account")


def test_preflight_tradovate_only_group_has_no_bar_feed():
    common = dict(app_id="a", app_secret="b", cid="c", sec="d")
    l1 = _acct("t1", account_id="1", **common); l1.login = "u1"
    l2 = _acct("t2", account_id="2", **common); l2.login = "u2"
    res = _pre(l1, [_spec("t2")], [l1, l2])
    assert "block" in _levels(res, "feed") and not res["can_start"]


def test_preflight_tradovate_leader_takes_bars_from_a_follower_and_warns_about_stops():
    l1 = _acct("lucid", account_id="1", app_id="a", app_secret="b", cid="c", sec="d")
    ft = _acct("ftmo", "cTrader", client_id="x", client_secret="y", refresh_token="z", ctid_trader_account_id="9")
    res = _pre(l1, [_spec("ftmo", symbol="US100")], [l1, ft])
    assert res["can_start"] and res["feed_account_id"] == "ftmo"
    assert _levels(res, "feed") == ["warn"] and _levels(res, "stops") == ["warn"]
    assert res["needs_ack"] and "ack" in _levels(res, "permission")


def test_preflight_tradovate_follower_needs_a_known_contract_size():
    leader = _acct("ct", "cTrader", client_id="x", client_secret="y", refresh_token="z", ctid_trader_account_id="9")
    tv = _acct("tv", account_id="1", app_id="a", app_secret="b", cid="c", sec="d")
    assert "block" in _levels(_pre(leader, [_spec("tv", symbol="WEIRDSYM")], [leader, tv], symbol="US100"), "contract_size")
    assert "block" not in _levels(_pre(leader, [_spec("tv", symbol="MNQ")], [leader, tv], symbol="US100"), "contract_size")


def test_preflight_multiplier_needs_a_value():
    leader = _acct("ct", "cTrader", client_id="x", client_secret="y", refresh_token="z", ctid_trader_account_id="9")
    other = _acct("ct2", "cTrader", client_id="x", client_secret="y", refresh_token="z", ctid_trader_account_id="10")
    res = _pre(leader, [_spec("ct2", size_mode=SIZE_MULTIPLIER, size_value=0)], [leader, other])
    assert "block" in _levels(res, "size")


# ------------------------------------------------------------------ copying end to end

def _three(tmp_path, **kw):
    f_mes = _Broker("mes25k", equity=25_000.0)
    f_es = _Broker("es100k", equity=100_000.0)
    f_mult = _Broker("multmes", equity=25_000.0)
    return _Rig(tmp_path, [
        ("mes25k", _spec("mes25k", symbol="MES", account_size=25_000.0), f_mes),
        ("es100k", _spec("es100k", symbol="ES", account_size=100_000.0), f_es),
        ("multmes", _spec("multmes", symbol="MES", size_mode=SIZE_MULTIPLIER, size_value=0.5, account_size=25_000.0), f_mult),
    ], **kw).begin()


def test_one_signal_opens_the_right_size_on_every_account_and_closes_them_together(tmp_path):
    rig = _three(tmp_path)
    rig.poll(sig=1)
    assert [o[2] for o in rig.leader.orders] == [1.0]                      # leader: 1 ES
    assert [o[2] for o in rig.brokers["mes25k"].orders] == [5.0]          # $250 risk / 10 pts / 5 per MES
    assert [o[2] for o in rig.brokers["es100k"].orders] == [2.0]          # $1000 risk / 10 pts / 50 per ES
    assert [o[2] for o in rig.brokers["multmes"].orders] == [5.0]         # half the leader's 50 units of exposure = 25 units = 5 MES
    assert rig.brokers["mes25k"].orders[0][0] == "MES" and rig.brokers["es100k"].orders[0][0] == "ES"
    # stops are anchored on each account and sent to platforms that take them
    assert rig.brokers["mes25k"].orders[0][3] == pytest.approx(4990.0)
    assert len(rig.session.status.followers) == 3 and all(f["open_position_ticket"] for f in rig.session.status.followers)

    rig.poll(sig=0)
    assert all(not b.positions for b in [rig.leader, *rig.brokers.values()])
    assert all(f["open_position_ticket"] is None and f["n_trades_closed"] == 1 for f in rig.session.status.followers)


def test_follower_obeys_the_per_order_lot_cap(tmp_path):
    f = _Broker("f", equity=25_000.0)
    rig = _Rig(tmp_path, [("f", _spec("f", symbol="MES", size_mode=SIZE_MULTIPLIER, size_value=1.0, account_size=25_000.0), f)]).begin()
    rig.poll(sig=1)
    # x1 of 1 ES is 10 MES, but every session enforces max 5 contracts/lots per single order
    assert [o[2] for o in f.orders] == [5.0]


def test_signal_reversal_flips_every_account(tmp_path):
    rig = _three(tmp_path)
    rig.poll(sig=1)
    rig.poll(sig=-1)
    for b in [rig.leader, *rig.brokers.values()]:
        assert len(b.positions) == 1 and b.positions[0].direction == -1


def test_each_account_is_journaled_separately(tmp_path):
    rig = _three(tmp_path)
    rig.poll(sig=1)
    rows = rig.journal._conn.execute("SELECT mt5_login FROM sessions").fetchall()
    assert sorted(r[0] for r in rows) == ["es100k", "leader", "mes25k", "multmes"]


def test_a_follower_that_cannot_size_one_contract_is_skipped_without_affecting_others(tmp_path):
    tiny = _Broker("tiny", equity=20_000.0)
    big = _Broker("big", equity=100_000.0)
    rig = _Rig(tmp_path, [
        ("tiny", _spec("tiny", symbol="ES", account_size=20_000.0), tiny),   # $200 / 10 pts = 20 units < 1 ES
        ("big", _spec("big", symbol="ES", account_size=100_000.0), big),
    ]).begin()
    rig.poll(sig=1)
    assert tiny.orders == [] and [o[2] for o in big.orders] == [2.0] and len(rig.leader.positions) == 1


def test_follower_drawdown_breach_halts_only_that_follower(tmp_path):
    a, b = _Broker("a", equity=25_000.0), _Broker("b", equity=25_000.0)
    rig = _Rig(tmp_path, [
        ("a", _spec("a", symbol="MES", account_size=25_000.0, max_dd_pct=10.0), a),
        ("b", _spec("b", symbol="MES", account_size=25_000.0, max_dd_pct=10.0), b),
    ]).begin()
    rig.poll(sig=1)
    assert a.positions and b.positions
    a.equity = 22_000.0                       # below the 10% trailing floor of 22.5k
    rig.poll()
    st = {f["account_id"]: f for f in rig.session.status.followers}
    assert st["a"]["alert"] and "MAX DRAWDOWN" in st["a"]["alert"] and not a.positions
    assert st["b"]["alert"] is None and b.positions                      # the rest of the group is untouched
    assert rig.session.status.alert is None and rig.session.status.running
    rig.poll(sig=0)
    rig.poll(sig=1)                                                      # a stays out; b and the leader trade on
    assert not a.positions and b.positions and rig.leader.positions


def test_follower_connection_loss_three_strikes_flattens_that_follower_only(tmp_path):
    a, b = _Broker("a"), _Broker("b")
    rig = _Rig(tmp_path, [("a", _spec("a", symbol="ES"), a), ("b", _spec("b", symbol="ES"), b)]).begin()
    rig.poll(sig=1)
    a.alive = False
    for _ in range(3):
        rig.poll()
    st = {f["account_id"]: f for f in rig.session.status.followers}
    assert st["a"]["alert"] and "CONNECTION LOST" in st["a"]["alert"] and a.flatten_calls == 1
    assert st["b"]["alert"] is None and b.positions


def test_leader_emergency_flattens_and_halts_every_follower(tmp_path):
    rig = _three(tmp_path)
    rig.poll(sig=1)
    rig.leader.equity = 44_000.0              # below the leader's 10% trailing floor (45k)
    rig.poll()
    assert rig.session.status.alert and "MAX DRAWDOWN" in rig.session.status.alert
    assert all(not b.positions for b in rig.brokers.values())
    assert all(f["alert"] and "LEADER HALTED" in f["alert"] for f in rig.session.status.followers)


def test_group_halt_can_be_switched_off(tmp_path):
    f = _Broker("f")
    rig = _Rig(tmp_path, [("f", _spec("f", symbol="ES"), f)], halt_group=False).begin()
    rig.poll(sig=1)
    rig.leader.equity = 44_000.0
    rig.poll()
    assert f.positions and rig.session.status.followers[0]["alert"] is None


def test_close_that_fails_is_retried_every_poll_until_it_works(tmp_path):
    f = _Broker("f")
    rig = _Rig(tmp_path, [("f", _spec("f", symbol="ES"), f)]).begin()
    rig.poll(sig=1)
    f.fail_close = True
    rig.poll(sig=0)                            # leader closes; the follower's close is rejected
    assert not rig.leader.positions and f.positions
    f.fail_close = False
    rig.poll()                                 # next poll: the owed close is retried
    assert not f.positions and rig.session.status.followers[0]["open_position_ticket"] is None


def test_repeated_close_failures_raise_a_loud_alert(tmp_path):
    f = _Broker("f")
    rig = _Rig(tmp_path, [("f", _spec("f", symbol="ES"), f)]).begin()
    rig.poll(sig=1)
    f.fail_close = True
    rig.poll(sig=0)
    rig.poll()
    rig.poll()
    assert "CLOSE FAILED" in rig.session.status.followers[0]["alert"]


def test_existing_unrelated_position_is_parked_not_touched(tmp_path):
    f = _Broker("f")
    f.positions.append(OpenPosition(ticket="manual-1", symbol="ES", direction=1, volume=3, open_price=4990.0, sl=0, tp=0,
                                    open_time=pd.Timestamp.now(tz="UTC"), profit=0.0))
    rig = _Rig(tmp_path, [("f", _spec("f", symbol="ES"), f)]).begin(leader_direction=0)
    assert rig.session.status.followers[0]["blocked_reason"]
    rig.poll(sig=1)
    rig.poll(sig=0)
    assert [p.ticket for p in f.positions] == ["manual-1"] and f.orders == []


def test_existing_position_matching_the_leader_is_adopted_and_closed_with_it(tmp_path):
    f = _Broker("f")
    f.positions.append(OpenPosition(ticket="old-1", symbol="ES", direction=1, volume=1, open_price=4990.0, sl=0, tp=0,
                                    open_time=pd.Timestamp.now(tz="UTC"), profit=0.0))
    rig = _Rig(tmp_path, [("f", _spec("f", symbol="ES"), f)]).begin(leader_direction=1)
    assert rig.session.status.followers[0]["open_position_ticket"] == "old-1"


def test_start_is_all_or_nothing(tmp_path):
    ok_one = _Broker("ok_one")
    bad = _Broker("bad", fail_connect=True)
    rig = _Rig(tmp_path, [("ok_one", _spec("ok_one", symbol="ES"), ok_one), ("bad", _spec("bad", symbol="ES"), bad)])
    started, msg = rig.session.start()
    assert started is False and "bad refused the login" in msg
    assert not ok_one.connected and not rig.leader.connected          # nothing is left connected
    assert rig.session._thread is None and not rig.session.status.running


def test_leader_without_bars_reads_prices_from_a_follower_feed(tmp_path):
    feed = _Broker("feed", equity=25_000.0)
    rig = _Rig(tmp_path, [("feed", _spec("feed", symbol="MES", account_size=25_000.0), feed)],
               leader_bars=False, bar_from="feed").begin()
    rig.poll(sig=1)                      # would raise on the leader's own (barless) adapter
    assert rig.leader.orders and feed.orders
    assert rig.leader.orders[0][3] == pytest.approx(4990.0)      # stop anchored on the feed's price


def test_a_lost_price_feed_counts_as_a_poll_failure(tmp_path):
    feed = _Broker("feed")
    rig = _Rig(tmp_path, [("feed", _spec("feed", symbol="ES"), feed)], leader_bars=False, bar_from="feed").begin()
    feed.alive = False
    rig.poll()
    assert rig.session._consecutive_poll_failures >= 1


def test_graceful_stop_leaves_positions_and_disconnects_everyone(tmp_path):
    rig = _three(tmp_path)
    rig.poll(sig=1)
    rig.session.stop()
    assert all(b.positions for b in rig.brokers.values())
    assert not any(b.connected for b in [rig.leader, *rig.brokers.values()])


def test_kill_switch_flattens_every_account(tmp_path):
    rig = _three(tmp_path)
    rig.poll(sig=1)
    rig.session.flatten_all_and_stop()
    assert all(not b.positions for b in [rig.leader, *rig.brokers.values()])


def test_session_without_a_copier_is_unchanged(tmp_path):
    leader = _Broker("leader")
    journal = ForwardTestJournal(db_path=tmp_path / "j.db")
    cfg = LiveExecutionConfig(symbol="ES", timeframe_minutes=15, risk=_risk(),
                              prop_rules=futures_prop_safe_rules(50_000.0, daily_loss_limit_pct=5.0, max_drawdown_pct=10.0),
                              contract_size=50.0, units_per_lot=50.0)
    strat = _Strategy(); strat.sig = 1
    s = LiveExecutionSession(strategy=strat, strategy_type="manual", strategy_filename="x", broker=leader, journal=journal, config=cfg)
    conn = leader.connect()
    s.status.connected, s.status.balance, s.status.equity = True, conn.balance, conn.equity
    s._session_id = journal.start_session("manual", "x", "ES", 15, "leader", "fake")
    s._dd_peak = conn.equity
    leader.tick = 1
    s._poll_once()
    assert [o[2] for o in leader.orders] == [1.0] and s.status.followers == []


# ------------------------------------------------------------------ several accounts under one login

class _Resp:
    def __init__(self, data, code=200):
        self._d, self.status_code = data, code

    def json(self):
        return self._d


class _Sess:
    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def _hit(self, method, url, **kw):
        path = (url.split("/backend-api")[-1] if "backend-api" in url else url.split("/v1")[-1]).split("?")[0]
        self.calls.append((method, path, kw.get("json")))
        v = self.routes.get((method, path))
        return _Resp(v if v is not None else {}, 200 if v is not None else 404)

    def get(self, url, **kw):
        return self._hit("GET", url, **kw)

    def post(self, url, **kw):
        return self._hit("POST", url, **kw)


def _tv_login(pick):
    a = TradovateBrokerAdapter("owen", "pw", "app", "sec", "1", "s", account_id=pick)
    a._session = _Sess({
        ("POST", "/auth/accesstokenrequest"): {"accessToken": "T"},
        ("GET", "/account/list"): [{"id": 11, "name": "LUCID-A"}, {"id": 22, "name": "LUCID-B"}, {"id": 33, "name": "LUCID-C"}],
        ("POST", "/cashBalance/getcashbalancesnapshot"): {"totalCashValue": 25000, "netLiq": 25000},
        ("POST", "/order/placeorder"): {"orderId": 9},
        ("GET", "/fill/ldeps"): [{"qty": 1, "price": 100.0}],
    })
    return a


def test_tradovate_picks_the_named_account_not_the_first():
    a = _tv_login("LUCID-B")
    assert a.connect().ok and a.account_id == 22 and a.account_name == "LUCID-B"
    a.place_market_order("MNQZ6", 1, 1)
    body = [c for c in a._session.calls if c[1] == "/order/placeorder"][0][2]
    assert body["accountId"] == 22 and body["accountSpec"] == "LUCID-B"


def test_tradovate_picks_by_numeric_id_and_still_learns_the_name():
    a = _tv_login("33")
    assert a.connect().ok and a.account_id == 33 and a.account_name == "LUCID-C"


def test_tradovate_blank_pick_keeps_the_old_first_account_behaviour():
    a = _tv_login("")
    assert a.connect().ok and a.account_id == 11


def test_tradovate_unknown_account_fails_and_lists_the_real_ones():
    r = _tv_login("NOPE").connect()
    assert not r.ok and "LUCID-A" in r.message and "LUCID-C" in r.message


def test_tradelocker_picks_by_account_number():
    a = TradeLockerBrokerAdapter("e@x.com", "pw", "SRV", account_id="7")
    a._session = _Sess({
        ("POST", "/auth/jwt/token"): {"accessToken": "A", "refreshToken": "R"},
        ("GET", "/auth/jwt/all-accounts"): {"accounts": [{"id": 100, "accNum": 1, "currency": "USD"},
                                                           {"id": 200, "accNum": 7, "currency": "USD"}]},
    })
    a._session.request = lambda m, url, **kw: _Resp({"s": "ok", "d": {"accountDetailsData": [1.0, 1.0, 1.0]}})
    r = a.connect()
    assert a.account_id == "200" and a.acc_num == 7


def test_registry_passes_the_account_pick_through():
    acct = LiveAccount(id="x", nickname="x", firm_name="", platform="Tradovate", login="owen", server="", password="pw",
                       extra_credentials={"app_id": "a", "app_secret": "b", "cid": "c", "sec": "d", "account_id": "LUCID-B"})
    assert build_adapter(acct)._account_pick == "LUCID-B"
    dx = LiveAccount(id="y", nickname="y", firm_name="", platform="DXtrade", login="u", server="", password="pw",
                     extra_credentials={"base_url": "https://x", "account_id": "DX-9"})
    assert build_adapter(dx).account_id == "DX-9"
