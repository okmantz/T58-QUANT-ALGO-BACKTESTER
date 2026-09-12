from pathlib import Path

from app.web.server import app

_SAMPLE_CSV = Path("data/examples/EURUSD_5M_sample.csv")


def _manual_strategy_form(**overrides):
    form = dict(
        strategy_mode="manual", sma_fast="20", sma_slow="50", sl_pips="20", tp_pips="40",
        initial_balance="10000", risk_value="1.0", pip_size="0.0001", n_sims="50",
    )
    form.update(overrides)
    return form


def test_prop_firm_recommender_form_loads():
    client = app.test_client()
    r = client.get("/prop-firm-recommender")
    assert r.status_code == 200
    assert b"Prop-firm recommender" in r.data
    # every catalog preset should appear as a checkbox option
    assert b"FTMO - $100k Challenge" in r.data
    assert b"Apex - $50k Evaluation" in r.data


def test_prop_firm_recommender_run_produces_ranked_table():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = _manual_strategy_form()
        data["csv_file"] = (f, "EURUSD_5M_sample.csv")
        r = client.post("/prop-firm-recommender/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Firm / preset" in r.data
    assert b"class=\"error\"" not in r.data


def test_prop_firm_recommender_run_respects_firm_selection():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = _manual_strategy_form()
        data["csv_file"] = (f, "EURUSD_5M_sample.csv")
        data["firms"] = "ftmo_100k"
        r = client.post("/prop-firm-recommender/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    # The results TABLE should contain only the selected firm -- extract
    # just the table body so the always-rendered firm-selection checklist
    # (which lists every preset regardless of selection) doesn't confound
    # the assertion.
    text = r.get_data(as_text=True)
    table_start = text.index("<tbody>")
    table_end = text.index("</tbody>")
    table_body = text[table_start:table_end]
    assert "FTMO - $100k Challenge" in table_body
    assert "Apex" not in table_body


def test_payout_probability_form_includes_preset_dropdown():
    client = app.test_client()
    r = client.get("/payout-probability")
    assert r.status_code == 200
    assert b"prop_preset_select" in r.data
    assert b"FTMO" in r.data


def test_payout_probability_run_with_scaling_enabled():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = _manual_strategy_form(
            account_size="10000", profit_target="2", daily_loss="100", max_dd="100",
            max_payouts_tracked="5", funding_approval_pct="100", evaluation_fee="0",
            profit_split="80", max_attempts="3",
            enable_scaling="on", scale_payouts_per="2", scale_multiplier="1.25", scale_max_multiple="4.0",
        )
        data["csv_file"] = (f, "EURUSD_5M_sample.csv")
        r = client.post("/payout-probability/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Probability of any scale-up" in r.data
    assert b"class=\"error\"" not in r.data


def test_payout_probability_run_without_scaling_omits_scaling_card():
    client = app.test_client()
    with open(_SAMPLE_CSV, "rb") as f:
        data = _manual_strategy_form(
            account_size="10000", profit_target="8", daily_loss="5", max_dd="10",
            max_payouts_tracked="5", funding_approval_pct="100", evaluation_fee="0",
            profit_split="80", max_attempts="3",
        )
        data["csv_file"] = (f, "EURUSD_5M_sample.csv")
        r = client.post("/payout-probability/run", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Probability of any scale-up" not in r.data
