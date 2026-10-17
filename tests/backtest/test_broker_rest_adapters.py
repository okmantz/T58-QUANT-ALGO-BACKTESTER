"""TradeLocker + Tradovate adapters against fake servers that mimic the real wire shapes."""
from __future__ import annotations

import json

from app.live_deploy.broker_tradelocker import TradeLockerBrokerAdapter
from app.live_deploy.broker_tradovate import TradovateBrokerAdapter


class _Resp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status
        self.text = json.dumps(data)
        self.headers = {}

    def json(self):
        return self._d


class _Session:
    def __init__(self, routes):
        self.routes, self.calls, self.headers = routes, [], {}

    def _hit(self, method, url, **kw):
        path = url.split("/backend-api")[-1] if "backend-api" in url else url.split("/v1")[-1]
        self.calls.append((method, path, kw.get("json"), kw.get("params"), kw.get("headers")))
        for (m, p), v in self.routes.items():
            if m == method and path.split("?")[0] == p:
                return _Resp(*(v(kw) if callable(v) else (v,)))
        return _Resp({"s": "error", "errmsg": f"no route {method} {path}"}, 404)

    def request(self, method, url, **kw):
        return self._hit(method, url, **kw)

    def get(self, url, **kw):
        return self._hit("GET", url, **kw)

    def post(self, url, **kw):
        return self._hit("POST", url, **kw)


_INST = {"d": {"instruments": [
    {"tradableInstrumentId": 77, "name": "EURUSD",
     "routes": [{"id": 11, "type": "TRADE"}, {"id": 12, "type": "INFO"}]}]}}


def _tl():
    a = TradeLockerBrokerAdapter("e@x.com", "pw", "SRV")
    acct = "/trade/accounts/9"
    a._session = _Session({
        ("POST", "/auth/jwt/token"): {"accessToken": "A", "refreshToken": "R"},
        ("GET", "/auth/jwt/all-accounts"): {"accounts": [{"id": 9, "accNum": 3, "currency": "USD"}]},
        ("GET", "/trade/config"): {"d": {
            "positionsConfig": {"columns": [{"id": c} for c in
                                ["id", "tradableInstrumentId", "routeId", "side", "qty", "avgPrice", "stopLossId",
                                 "takeProfitId", "openDate", "unrealizedPl", "strategyId"]]},
            "accountDetailsConfig": {"columns": [{"id": "balance"}, {"id": "projectedBalance"}, {"id": "availableFunds"}]},
            "ordersHistoryConfig": {"columns": [{"id": c} for c in ["id", "status", "avgPrice", "positionId"]]}}},
        ("GET", acct + "/state"): {"s": "ok", "d": {"accountDetailsData": [50000.0, 50100.0, 49000.0]}},
        ("GET", acct + "/instruments"): _INST,
        ("GET", acct + "/positions"): {"d": {"positions": [[501, 77, 11, "buy", 0.5, 1.1, 0, 0, 1700000000000, 12.5, ""]]}},
        ("GET", acct + "/ordersHistory"): {"d": {"ordersHistory": [[42, "Filled", 1.1002, 501]]}},
        ("POST", acct + "/orders"): {"s": "ok", "d": {"orderId": "42"}},
        ("DELETE", "/trade/positions/501"): {"s": "ok"},
    })
    return a


def test_tradelocker_connect_positions_order_close():
    a = _tl()
    c = a.connect()
    assert c.ok and c.balance == 50000.0 and c.equity == 50100.0 and a.acc_num == 3
    pos = a.get_open_positions("EURUSD")
    assert len(pos) == 1 and pos[0].symbol == "EURUSD" and pos[0].direction == 1 and pos[0].volume == 0.5
    r = a.place_market_order("EURUSD", 1, 0.5, sl_price=1.09)
    assert r.ok and r.price == 1.1002 and r.ticket == "501"
    order = [c for c in a._session.calls if c[0] == "POST" and c[1].endswith("/orders")][0]
    assert order[2]["routeId"] == "11" and order[2]["tradableInstrumentId"] == "77"
    assert order[2]["qty"] == "0.5" and order[2]["stopLossType"] == "absolute"
    assert order[4]["accNum"] == "3"
    assert a.close_position("501").ok
    assert not a.place_market_order("NOPE", 1, 1).ok


def test_tradelocker_login_rejected():
    a = TradeLockerBrokerAdapter("e", "p", "S")
    a._session = _Session({("POST", "/auth/jwt/token"): ({"message": "bad"}, 401)})
    assert not a.connect().ok


def _tv():
    a = TradovateBrokerAdapter("user", "pw", "app", "sec", "1", "s")
    a.access_token, a.account_id = "T", 5
    a._session = _Session({
        ("POST", "/order/placeorder"): {"orderId": 900},
        ("GET", "/fill/ldeps"): [{"qty": 2, "price": 100.0}, {"qty": 2, "price": 101.0}],
        ("GET", "/position/list"): [{"id": 3, "accountId": 5, "contractId": 8, "netPos": 2, "netPrice": 100.5, "openPl": 3}],
        ("GET", "/contract/item"): {"name": "NQZ6"},
        ("POST", "/order/liquidateposition"): {"orderId": 901},
        ("POST", "/cashBalance/getcashbalancesnapshot"): {"totalCashValue": 50000, "netLiq": 50100},
    })
    return a


def test_tradovate_order_shape_fill_positions_and_close():
    a = _tv()
    r = a.place_market_order("NQZ6", -1, 4)
    assert r.ok and r.price == 100.5
    body = [c for c in a._session.calls if c[1] == "/order/placeorder"][0][2]
    assert body["accountSpec"] == "user" and body["isAutomated"] is True and body["action"] == "Sell"
    pos = a.get_open_positions("NQZ6")
    assert pos[0].symbol == "NQZ6" and pos[0].volume == 2
    c = a.close_position("3")
    assert c.ok
    liq = [c for c in a._session.calls if c[1] == "/order/liquidateposition"][0][2]
    assert liq["contractId"] == 8 and liq["accountId"] == 5
    assert a.account_summary()["equity"] == 50100


def test_tradovate_rejection_surfaces():
    a = _tv()
    a._session.routes[("POST", "/order/placeorder")] = {"failureReason": "RiskCheck", "failureText": "limit"}
    r = a.place_market_order("NQZ6", 1, 1)
    assert not r.ok and "limit" in r.message
