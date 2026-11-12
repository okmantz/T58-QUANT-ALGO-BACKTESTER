"""Firm-level facts from the owner's Prop Firm Rules Report (verified 2026-11-11): the one place that says, per firm, whether
automation is allowed, what the published ES/NQ round-trip commission is, and which plans were retired.

Presets (app.prop.presets) carry per-plan rules; this registry carries what is true of the FIRM, so a tool can refuse a bot on a
firm that bans bots even when no preset exists for the plan. ``algo`` is one of ALLOWED / CONDITIONAL / NOT_ALLOWED / UNVERIFIED;
only ALLOWED (and Topstep's SIM-only CONDITIONAL lane) may back ``algo_trading_allowed=True`` on a preset.
"""
from __future__ import annotations

REPORT_DATE = "2026-11-11"
REPORT_NAME = "Prop Firm Rules Report (owner-compiled)"

FIRM_REGISTRY: dict[str, dict] = {
    "Alpha Futures": {"algo": "NOT_ALLOWED", "es_nq_rt": 4.78, "note": "AI/bots/automation banned on every account type; semi-automated (manual placement) only. Zero 50K list $169/mo, reset $149. Premium retired; Standard relaunched Jul 2026; Direct (one-time fee) added."},
    "Apex Trader Funding": {"algo": "NOT_ALLOWED", "es_nq_rt": 3.10, "note": "No AI/algos/bots/scripts/HFT; only automated copy of your own Apex accounts. Apex 4.0: flat 100% split, a Performance Account closes after 6 payouts."},
    "Topstep": {"algo": "CONDITIONAL", "es_nq_rt": 3.78, "note": "Automation allowed in Combine/XFA (SIM) via the TopstepX/ProjectX API ($29/mo); NOT on Live Funded; HFT banned; orders from your own device."},
    "Lucid Trading": {"algo": "ALLOWED", "es_nq_rt": 3.50, "es_nq_rt_tier": "secondary", "note": "Automation and copiers permitted; HFT banned; microscalp (>50% of profit from <=5s holds) flagged; hedging banned. One-time fees, 90/10."},
    "MyFundedFutures": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "Allowed with conditions [secondary]. Core/Scale/Flex RETIRED; current: Rapid, Rapid EOD, Builder, Pro, Studio."},
    "Tradeify": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "Sole-owner provable bots only, Tradeify-only; HFT banned; funded microscalp rule (>50% from holds >10s)."},
    "Take Profit Trader": {"algo": "NOT_ALLOWED", "es_nq_rt": 4.50, "note": "No automated or bot trading of any kind; copiers only between your own accounts."},
    "Earn2Trade": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "No explicit permission; copiers forbidden; get written approval. Progression Ladder RETIRED 2026-11-02; 5 profitable days before every withdrawal."},
    "Bulenox": {"algo": "ALLOWED", "es_nq_rt": 4.18, "note": "Bots/algos/copiers permitted ONLY if user-built for personal use; Rithmic third-party API +$100/mo. Old $250K/$10K sizes withdrawn; monthly billing discontinued Aug 2026."},
    "Elite Trader Funding": {"algo": "CONDITIONAL", "es_nq_rt": 4.08, "note": "EAs/AI/algos NOT allowed unless expressly authorised in writing; 10-second minimum trade duration."},
    "TradeDay": {"algo": "CONDITIONAL", "es_nq_rt": 5.76, "note": "Own systems only inside NinjaTrader/Tradovate/TradingView/Jigsaw/Quantower; purchased bots forbidden; >200 trades/day prohibited."},
    "OneUp Trader": {"algo": "UNVERIFIED", "es_nq_rt": 5.00, "note": "Rules never mention bots; funded rules ban <10s trades, microscalping, HFT. Treat as NOT allowed until written confirmation."},
    "Leeloo Trading": {"algo": "NOT_ALLOWED", "es_nq_rt": None, "note": "Any system trading for you is banned; commissions UNVERIFIED (no official table)."},
    "Top One Futures": {"algo": "NOT_ALLOWED", "es_nq_rt": 5.76, "note": "Automation/EA/HFT strictly forbidden. (TopOne Trader, the CFD sister brand, allows EAs in challenge phases only.)"},
    "TopOne Trader": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "CFD brand: EAs in challenge phases only, never on funded/instant accounts."},
    "FTMO": {"algo": "ALLOWED", "es_nq_rt": None, "note": "EAs and algorithmic trading explicitly allowed (official FAQ)."},
    "FundedNext": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "EAs only on accounts below $50,000; manual-only at $50K+. The 40% best-day rule is On-Demand (90%) only."},
    "E8 Markets": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "Algo allowed, limited to 1 EA strategy per user [secondary]. E8 One preset needs a re-check against current rules."},
    "The5%ers": {"algo": "CONDITIONAL", "es_nq_rt": None, "note": "Own-code EAs only; written approval required; third-party/tick-scalping EAs banned."},
    "FunderPro": {"algo": "ALLOWED", "es_nq_rt": None, "note": "EAs explicitly permitted on One Phase."},
    "Atlas Funded": {"algo": "ALLOWED", "es_nq_rt": None, "note": "EAs and bots fully allowed. CONFLICT kept visible: 1-Step target 10% vs 11%, $100K fee $342 vs $538."},
    "The Futures Desk": {"algo": "ALLOWED", "es_nq_rt": None, "note": "Algo-first; reportedly acquired by Topstep 2026-04-01 [secondary single source] -- re-confirm it still sells."},
}

# Plans the report says are no longer sold (so nobody builds a verdict on a product that cannot be bought).
RETIRED_PLANS = {
    "MyFundedFutures": ("Core", "Scale", "Flex"),
    "Earn2Trade": ("Progression Ladder",),
    "Alpha Futures": ("Premium",),
    "Bulenox": ("$250K size", "$10K size", "monthly subscription billing"),
}

# The cost basis every report states (P2-17): Alpha Trader's published schedule is the HIGHEST ES/NQ round trip among the algo-capable
# futures lanes (Lucid ~$3.50 [secondary]), so a backtest costed on it is conservative for Lucid.
COST_BASIS = {"name": "Alpha Trader schedule", "es_nq_rt": 4.78, "gc_rt": 5.22, "micro_rt": 1.32, "mgc_rt": 1.82,
              "vs_lucid_es_nq_rt": 3.50, "note": "adopted as the conservative basis for Lucid (Lucid publishes ~$3.50 ES/NQ RT [secondary])"}


def firm_policy(firm: str) -> dict | None:
    return FIRM_REGISTRY.get(firm)


def algo_permitted(firm: str) -> bool:
    """True only where the report says automation is allowed outright (Topstep's SIM-only lane counts for the Combine/XFA presets)."""
    p = FIRM_REGISTRY.get(firm)
    return bool(p and p["algo"] in ("ALLOWED",))


def cost_basis_line() -> str:
    c = COST_BASIS
    return (f"Cost basis: {c['name']} (ES/NQ ${c['es_nq_rt']:.2f}, GC ${c['gc_rt']:.2f}, micros ${c['micro_rt']:.2f} round-trip) -- "
            f"{c['note']}.")

# The preset catalog spells Topstep "TopStep"; keep both spellings resolvable.
FIRM_REGISTRY["TopStep"] = FIRM_REGISTRY["Topstep"]
