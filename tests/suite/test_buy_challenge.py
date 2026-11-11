import pytest

from app.buy_challenge import catalog
from app.prop import desk_store as ds
from app.web.server import app


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "offers_path", lambda: tmp_path / "offers.json")
    monkeypatch.setattr(ds, "desk_path", lambda: tmp_path / "desk.json")
    return tmp_path


def test_listed_codes_ship_labelled_and_never_change_a_price(isolated):
    cat = catalog.catalog()
    by = {f["firm"]: f for f in cat["firms"]}
    assert by["FTMO"]["offer"]["code"] == "GORILLA" and by["FTMO"]["offer"]["listed"]
    assert by["FTMO"]["offer"]["source"] == "PropFirmGorilla" and by["FTMO"]["offer"]["up_to_pct"] == 20
    assert by["Apex Trader Funding"]["offer"]["up_to_pct"] == 90
    # an "up to" figure is never subtracted from a price
    assert all(p["price_with_code"] is None for p in cat["plans"])
    assert {m["firm"] for m in cat["more_codes"]} >= {"Tradeify", "My Funded Futures"}
    assert cat["codes_source"]["checked_on"]


def test_every_firm_in_presets_has_a_card_and_algo_badge_follows_presets(isolated):
    from app.prop import presets as P
    firms = {f["firm"]: f for f in catalog.catalog()["firms"]}
    assert {p.firm for p in P.list_presets()} <= set(firms)
    for p in P.list_presets():
        row = next(r for r in catalog.plans() if r["key"] == p.key)
        assert row["algo_allowed"] == p.algo_trading_allowed


def test_offer_round_trip_discount_and_expiry(isolated):
    from datetime import date
    catalog.save_offer("FTMO", {"code": "SAVE10", "discount_pct": 10, "checked_on": "2026-11-01"})
    row = next(r for r in catalog.plans() if r["key"] == "ftmo_100k")
    assert row["offer"]["active"] and row["price_with_code"] == round(row["price"] * 0.9, 2)
    catalog.save_offer("FTMO", {"code": "OLD", "discount_pct": 10, "expires": "2020-01-01"})
    row = next(r for r in catalog.plans(today=date(2026, 11, 10)) if r["key"] == "ftmo_100k")
    assert row["offer"]["expired"] and row["price_with_code"] is None
    catalog.save_offer("FTMO", {})
    assert catalog.load_offers() == {}


def test_offer_validation(isolated):
    with pytest.raises(ValueError):
        catalog.save_offer("Nonexistent Firm", {"code": "X"})
    with pytest.raises(ValueError):
        catalog.save_offer("FTMO", {"discount_pct": 150})
    catalog.save_offer("FTMO", {"code": "A", "checkout_url": "javascript:alert(1)"})
    assert catalog.load_offers()["FTMO"]["checkout_url"] == ""      # only http(s) links are kept


def test_unconfirmed_site_has_no_link(isolated):
    f = {x["firm"]: x for x in catalog.catalog()["firms"]}
    assert f["FunderPro"]["link_confirmed"] is False
    catalog.save_offer("FunderPro", {"site_url": "https://example.com/funderpro"})
    f = {x["firm"]: x for x in catalog.catalog()["firms"]}
    assert f["FunderPro"]["link_confirmed"] and f["FunderPro"]["checkout_url"] == "https://example.com/funderpro"


def test_purchase_logs_cost_and_opens_attempt(isolated):
    acct = catalog.record_purchase("ftmo_100k")
    data = ds.load()
    assert data["accounts"][0]["id"] == acct["id"] and data["costs"][0]["kind"] == "purchase"
    assert data["attempts"][0]["outcome"] == "open"


def test_routes(isolated):
    c = app.test_client()
    assert c.get("/buy-challenge").status_code == 200
    j = c.get("/api/buy-challenge/catalog").json
    assert j["ok"] and len(j["plans"]) >= 20 and "does not take payment" in j["disclaimer"]
    assert c.put("/api/buy-challenge/offers/FTMO", json={"code": "ZED"}).json["ok"]
    assert c.put("/api/buy-challenge/offers/Nope", json={"code": "ZED"}).status_code == 400
    assert c.post("/api/buy-challenge/purchase", json={"preset_key": "nope"}).status_code == 400
    assert c.post("/api/buy-challenge/purchase", json={"preset_key": "lucid_50k"}).json["ok"]
    assert 'href="/buy-challenge"' in c.get("/dashboard").get_data(as_text=True)
