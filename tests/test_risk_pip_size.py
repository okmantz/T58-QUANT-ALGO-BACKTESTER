import pandas as pd

from app.backtest.risk import suggest_pip_size


def _df_with_close(price):
    return pd.DataFrame({"close": [price] * 10})


def test_suggests_fx_pip_size_for_low_priced_pairs():
    assert suggest_pip_size(_df_with_close(1.10)) == 0.0001


def test_suggests_cent_scale_for_stock_and_jpy_priced_instruments():
    assert suggest_pip_size(_df_with_close(190.0)) == 0.01
    assert suggest_pip_size(_df_with_close(150.0)) == 0.01  # USDJPY-scale


def test_suggests_whole_number_scale_for_index_or_high_value_instruments():
    assert suggest_pip_size(_df_with_close(4500.0)) == 1.0


def test_handles_missing_or_empty_data_gracefully():
    assert suggest_pip_size(None) == 0.0001
    assert suggest_pip_size(pd.DataFrame({"close": []})) == 0.0001
    assert suggest_pip_size(pd.DataFrame({"open": [1.0]})) == 0.0001


def test_has_instrument_scale_mismatch_detects_either_marker():
    from app.backtest.risk import has_instrument_scale_mismatch

    assert has_instrument_scale_mismatch([
        "1 trade(s) had a fixed-pips stop whose price distance doesn't match the instrument actually being tested",
    ])
    assert has_instrument_scale_mismatch([
        "1 trade(s) had a fixed-pips stop under 15% of this instrument's own recent ATR (as little as 6.7%)",
    ])
    assert not has_instrument_scale_mismatch(["some unrelated warning"])
    assert not has_instrument_scale_mismatch([])


def test_instrument_scale_mismatch_message_mentions_pip_size():
    from app.backtest.risk import instrument_scale_mismatch_message

    msg = instrument_scale_mismatch_message(0.0001)
    assert "0.0001" in msg
    assert "pip_size" in msg
