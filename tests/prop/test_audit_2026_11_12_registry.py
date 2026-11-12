from app.prop import presets as P
from app.prop.firm_registry import FIRM_REGISTRY, RETIRED_PLANS, algo_permitted, cost_basis_line

REPORT_FIRMS = ["Alpha Futures", "Apex Trader Funding", "TopStep", "Lucid Trading", "MyFundedFutures", "Tradeify", "Take Profit Trader",
                "Earn2Trade", "Bulenox", "Elite Trader Funding", "TradeDay", "OneUp Trader", "Leeloo Trading", "Top One Futures", "TopOne Trader",
                "FTMO", "FundedNext", "E8 Markets", "The5%ers", "FunderPro", "Atlas Funded", "The Futures Desk"]


def test_registry_covers_every_firm_in_the_report():
    assert set(REPORT_FIRMS) <= set(FIRM_REGISTRY)


def test_every_report_futures_firm_has_a_preset_and_a_fee_row():
    firms_with_presets = {p.firm for p in P.PROP_FIRM_PRESETS}
    for f in REPORT_FIRMS:
        if f == "TopOne Trader":          # CFD sister brand -- the report covers it, T58 trades futures
            continue
        assert f in firms_with_presets, f"no preset for {f}"
    for p in P.PROP_FIRM_PRESETS:
        assert p.key in P.FIRM_FEES, f"no fee row for {p.key}"


def test_presets_never_allow_a_bot_the_report_bans():
    for p in P.PROP_FIRM_PRESETS:
        pol = FIRM_REGISTRY.get(p.firm)
        assert pol is not None, p.firm
        if pol["algo"] in ("NOT_ALLOWED", "UNVERIFIED"):
            assert p.algo_trading_allowed is False, f"{p.key} allows a bot but the report says {pol['algo']}"
    assert algo_permitted("Lucid Trading") and not algo_permitted("Apex Trader Funding")


def test_new_presets_are_unstamped_and_retired_plans_listed():
    for k in ("alpha_zero_50k", "mffu_rapid_50k", "tradeify_growth_50k", "tpt_test_50k", "bulenox_qual_50k"):
        p = P.get_preset(k)
        assert not P.is_ready_eligible(p)[0] if isinstance(P.is_ready_eligible(p), tuple) else not P.is_ready_eligible(p)
    assert "Core" in RETIRED_PLANS["MyFundedFutures"] and "Progression Ladder" in RETIRED_PLANS["Earn2Trade"]
    assert "4.78" in cost_basis_line()


def test_report_2026_11_11_fee_updates_present():
    assert P.FIRM_FEES["alpha_zero_50k"].challenge_price == 169.0 and P.FIRM_FEES["alpha_zero_50k"].reset_price == 149.0
    assert P.FIRM_FEES["lucid_50k"].challenge_price == 172.0 and P.FIRM_FEES["lucid_50k"].reset_price == 120.0
    assert P.FIRM_FEES["apex_25k_eod"].challenge_price == 490.0 and P.FIRM_FEES["apex_25k_eod"].activation_fee == 109.0
