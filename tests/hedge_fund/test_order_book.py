from __future__ import annotations

import pytest

from app.quant_lab.order_book import LimitOrderBook, OrderBookError, Side


def test_resting_limit_orders_dont_cross():
    book = LimitOrderBook()
    r1 = book.submit_limit_order(Side.BUY, 100.0, 10)
    r2 = book.submit_limit_order(Side.SELL, 101.0, 10)
    assert r1.resting and r2.resting
    assert not r1.trades and not r2.trades
    assert book.best_bid() == 100.0
    assert book.best_ask() == 101.0
    assert book.spread() == 1.0
    assert book.mid_price() == 100.5


def test_crossing_limit_order_generates_a_trade_at_the_resting_price():
    book = LimitOrderBook()
    book.submit_limit_order(Side.BUY, 100.0, 10)
    result = book.submit_limit_order(Side.SELL, 99.0, 4)
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.price == 100.0  # fills at the RESTING (maker) price, not the aggressive taker price
    assert trade.quantity == 4.0
    assert result.filled_quantity == 4.0
    assert not result.resting
    depth = book.depth_snapshot()
    assert depth["bids"] == [(100.0, 6.0)]


def test_price_time_priority_fifo_within_a_level():
    book = LimitOrderBook()
    first = book.submit_limit_order(Side.BUY, 100.0, 5)
    second = book.submit_limit_order(Side.BUY, 100.0, 5)
    result = book.submit_limit_order(Side.SELL, 100.0, 7)
    # The first resting order should be filled (fully) before the second.
    makers = [t.maker_order_id for t in result.trades]
    assert makers == [first.order.order_id, first.order.order_id, second.order.order_id] or makers[0] == first.order.order_id
    assert result.filled_quantity == 7.0


def test_market_order_sweeps_multiple_price_levels():
    book = LimitOrderBook()
    book.submit_limit_order(Side.SELL, 101.0, 5)
    book.submit_limit_order(Side.SELL, 102.0, 5)
    result = book.submit_market_order(Side.BUY, 8)
    assert result.filled_quantity == 8.0
    prices_hit = [t.price for t in result.trades]
    assert prices_hit == [101.0, 102.0]


def test_market_order_partial_fill_when_book_runs_dry():
    book = LimitOrderBook()
    book.submit_limit_order(Side.SELL, 101.0, 3)
    result = book.submit_market_order(Side.BUY, 10)
    assert result.filled_quantity == 3.0
    assert result.remaining_quantity == 7.0
    assert not result.fully_filled
    assert not result.resting  # unfilled remainder is dropped, never rests


def test_cancel_removes_resting_order():
    book = LimitOrderBook()
    r = book.submit_limit_order(Side.BUY, 100.0, 5)
    assert book.cancel_order(r.order.order_id) is True
    assert book.best_bid() is None
    assert book.cancel_order(r.order.order_id) is False  # already gone


def test_invalid_orders_raise():
    book = LimitOrderBook()
    with pytest.raises(OrderBookError):
        book.submit_limit_order(Side.BUY, -5.0, 1)
    with pytest.raises(OrderBookError):
        book.submit_limit_order(Side.BUY, 100.0, 0)
    with pytest.raises(OrderBookError):
        book.submit_market_order(Side.SELL, -1)


def test_tape_records_every_trade_in_order():
    book = LimitOrderBook()
    book.submit_limit_order(Side.BUY, 100.0, 5)
    book.submit_limit_order(Side.SELL, 100.0, 5)
    book.submit_limit_order(Side.BUY, 100.0, 3)
    book.submit_limit_order(Side.SELL, 100.0, 3)
    assert len(book.tape) == 2
    assert [t.quantity for t in book.tape] == [5.0, 3.0]
