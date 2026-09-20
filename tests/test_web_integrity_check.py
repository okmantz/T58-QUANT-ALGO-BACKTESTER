from pathlib import Path

from app.web.server import REPORTS_DIR, app

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "examples" / "EURUSD_5M_sample.csv"

_BASE_FORM = {
    "strategy_mode": "manual",
    "sma_fast": "20", "sma_slow": "50", "sl_pips": "20", "tp_pips": "40",
    "account_size": "100000", "profit_target": "8", "daily_loss": "5", "max_dd": "10",
    "dd_type": "trailing", "consistency": "30", "min_days": "5", "payout_freq": "14",
    "payout_threshold": "0", "buffer": "0", "payout_cap": "",
    "initial_balance": "100000", "risk_mode": "percent", "risk_value": "1.0",
    "max_trades_day": "10", "commission": "0", "slippage_pips": "0.5",
    "spread_pips": "1.0", "pip_size": "0.0001",
    "n_sims": "100", "mc_method": "bootstrap",
}


def test_run_still_succeeds_and_shows_integrity_check_for_a_normal_request():
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = dict(_BASE_FORM, csv_file=(f, "EURUSD_5M_sample.csv"))
        r = client.post("/run", data=data, content_type="multipart/form-data")

    assert r.status_code == 200
    body = r.get_data(as_text=True)
    assert "Eval Pass Probability" in body
    assert "T58 Backtest Integrity Check -- VALID" in body
    assert "BACKTEST STATUS: VALID" in body

    for f in REPORTS_DIR.glob("report_*"):
        f.unlink()


def test_run_is_blocked_when_requested_timeframe_is_finer_than_native_data():
    """EURUSD_5M_sample.csv is native 5-minute data -- asking for 1m bars
    is asking the engine to manufacture bars finer than the source data,
    which the integrity check should refuse up front rather than let a
    resample silently misbehave."""
    client = app.test_client()
    with open(SAMPLE_CSV, "rb") as f:
        data = dict(_BASE_FORM, csv_file=(f, "EURUSD_5M_sample.csv"), timeframe="1m")
        r = client.post("/run", data=data, content_type="multipart/form-data")

    assert r.status_code == 400
    body = r.get_data(as_text=True)
    assert "BACKTEST BLOCKED" in body
    assert "Resampling: FAILED" in body
    assert "No performance results were produced" in body
