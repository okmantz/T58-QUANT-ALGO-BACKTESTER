"""
Per-section "Start Here" guides -- ONE source of truth for both builds.

The web app renders these as /start-here/<section> (see app/web/server.py,
section_start_here.html); the desktop app renders the same data as a "Start
Here" page at the top of each sidebar group (app/ui/main_window.py,
_build_start_here_tab). Keeping the text here -- instead of a copy in each
build -- means the two can never drift apart.

Each tool lists its web `href`; DESKTOP_NAV_FOR_HREF maps that href to the
desktop sidebar page key that opens the equivalent tab.
"""
from __future__ import annotations

SECTION_GUIDES = {
    "create": {
        "title": "Create", "accent_color": "#a78bfa",
        "tagline": "Everything that produces a new strategy, from a blank page to a candidate worth testing.",
        "description": (
            "Create is where a strategy comes from in the first place -- either generated broadly by the "
            "system (Forge, Speed Run, Research Agent/Director/Loop, Generate Strategies) when you don't yet "
            "have a specific idea, or built directly (Strategy Library) when you do. Nothing here is validated "
            "yet -- a strategy that comes out of Create still needs to go through Test and Optimize before "
            "it's trustworthy."
        ),
        "roadmap": [
            "No idea yet? Start with Forge Strategy -- one button, generates and screens thousands of "
            "hypotheses, and validates survivors for prop-firm survival.",
            "Want it faster and narrower? Speed Run does a quicker single-pass version of the same idea.",
            "Have a specific idea already? Build or edit it directly in the Strategy Library.",
            "Want an LLM to reason about market structure and propose ideas? Try Research Agent, Research "
            "Director, or the background Research Loop.",
            "Once you have a survivor you like, send it to Search Lab or Evolution Lab (under Optimize) to "
            "sharpen its parameters, or straight to Full Pipeline for a validated verdict.",
        ],
        "primary_buttons": [{"label": "Forge Strategy", "href": "/forge"}, {"label": "Strategy Library", "href": "/library"}],
        "tools": [
            {"name": "Forge Strategy", "href": "/forge", "desc": "One-button broad generate + screen + prop-survival check."},
            {"name": "Generate Strategies (AI)", "href": "/generate-strategies", "desc": "LLM-assisted strategy code generation."},
            {"name": "Research Agent", "href": "/research-agent", "desc": "LLM reasons over market structure for ideas."},
            {"name": "Research Director", "href": "/research", "desc": "Directs a deeper, multi-step research pass."},
            {"name": "Research Loop (background)", "href": "/research-loop", "desc": "Keeps researching unattended, on a schedule."},
            {"name": "Speed Run", "href": "/speed-run", "desc": "Faster, narrower single-pass version of Forge."},
            {"name": "Multi-Instrument Speed Run", "href": "/speed-run/multi-instrument", "desc": "Speed Run across several datasets at once."},
            {"name": "Strategy Library", "href": "/library", "desc": "Build, edit, or browse strategies directly."},
        ],
    },
    "test": {
        "title": "Test", "accent_color": "#22d3ee",
        "tagline": "Run one specific strategy against one dataset and read the report.",
        "description": (
            "Test is the manual, single-run path: you already have a strategy and a dataset in mind, and you "
            "want to run a backtest and read the resulting report -- trades, equity curve, drawdown, prop-firm "
            "pass/payout probability -- without any search or optimization happening. Use Optimize instead once "
            "you want the system to search for better parameters rather than just report on the ones you gave it."
        ),
        "roadmap": [
            "Pick a strategy (from the Strategy Library, or configure one directly) and a dataset.",
            "Set your prop-firm rules and risk/execution settings.",
            "Run & Report gives you the full backtest report for that exact configuration.",
            "Payout Probability adds Monte Carlo-based odds of actually reaching a funded payout.",
            "Not sure which prop firm's rules to test against? Prop-Firm Recommender narrows it down.",
        ],
        "primary_buttons": [{"label": "Run & Report", "href": "/"}],
        "tools": [
            {"name": "Run & Report", "href": "/", "desc": "One strategy, one dataset, one full backtest report."},
            {"name": "Payout Probability", "href": "/payout-probability", "desc": "Monte Carlo odds of reaching a funded payout."},
            {"name": "Prop-Firm Recommender", "href": "/prop-firm-recommender", "desc": "Find which prop firm's rules fit a strategy best."},
        ],
    },
    "champion": {
        "title": "Champion", "accent_color": "#e879f9",
        "tagline": "Once you have strong individual strategies, this is where they're combined into something sturdier.",
        "description": (
            "Champion is about combining strength, not finding it from scratch: it takes strategies that already "
            "did well in Optimize/Validate and asks whether running several together (an ensemble), across "
            "several assets (a portfolio), or from genuinely different families (diversity) produces something "
            "more robust than any single strategy alone."
        ),
        "roadmap": [
            "Check Family Diversity first -- confirms your surviving strategies aren't all secretly the same idea.",
            "Multi-Asset Portfolio combines strategies across different instruments.",
            "Multi-Strategy Ensemble combines several strategies on the same instrument into one blended system.",
        ],
        "primary_buttons": [{"label": "Multi-Strategy Ensemble", "href": "/ensemble"}],
        "tools": [
            {"name": "Family Diversity", "href": "/family-diversity", "desc": "Confirms survivors aren't all the same underlying idea."},
            {"name": "Multi-Asset Portfolio", "href": "/portfolio", "desc": "Combine strategies across different instruments."},
            {"name": "Multi-Strategy Ensemble", "href": "/ensemble", "desc": "Blend several strategies into one combined system."},
        ],
    },
    "deployment": {
        "title": "Deployment", "accent_color": "#a3e635",
        "tagline": "Champion checks, then forward-testing and live monitoring once you're ready to actually trade a strategy.",
        "description": (
            "Deployment covers everything after a strategy is validated: ongoing champion checks (Overnight "
            "Autopilot, Compare, Strategy Health) that keep watching a strategy over time, and the live-markets "
            "path (Forward Test, Deploy Live, Monitor, Interactive Replay) for actually running it against a "
            "real or demo broker account. Deploy Live risks real money -- read its own page carefully before "
            "using it."
        ),
        "roadmap": [
            "Overnight Autopilot and Compare Strategies keep validating a champion over time as new data arrives.",
            "Add a broker/prop-firm account under Settings \u2192 API Keys \u2192 Brokers/Prop Firms.",
            "Forward Test runs a strategy against a demo/paper account first -- always do this before Deploy Live.",
            "Deploy Live trades real money -- read the warnings on that page first.",
            "Monitor (Live Market) and Interactive Replay let you watch what a running or historical session actually did.",
        ],
        "primary_buttons": [{"label": "Forward Test (MT5)", "href": "/forward-test"}],
        "tools": [
            {"name": "Overnight Autopilot", "href": "/overnight-autopilot", "desc": "Keeps re-checking a champion strategy unattended."},
            {"name": "Compare Strategies", "href": "/compare", "desc": "Side-by-side comparison of two or more strategies."},
            {"name": "Strategy Health / Auto Re-tune", "href": "/quant-lab/strategy-health", "desc": "Watches for a strategy's edge decaying over time."},
            {"name": "Forward Test (MT5)", "href": "/forward-test", "desc": "Demo/paper-account forward test -- do this before Deploy Live."},
            {"name": "Deploy Live", "href": "/deploy-live", "desc": "Real-money live trading -- read the warnings first."},
            {"name": "Monitor (Live Market)", "href": "/live-market", "desc": "Watch a running live/forward-test session."},
            {"name": "Interactive Replay", "href": "/replay", "desc": "Step bar-by-bar through a historical session."},
        ],
    },
    "graveyard": {
        "title": "Strategy Graveyard", "accent_color": "#c9d1d9",
        "tagline": "Every rejected strategy, and exactly why it failed -- so you don't re-test the same dead idea twice.",
        "description": (
            "Every strategy Forge, Search Lab, or Evolution Lab rejects gets recorded here with the specific "
            "reason it failed, per instrument -- not raw profit, but pass/payout probability and prop-firm "
            "survival. Worth checking before a new search: if a whole family keeps dying the same way on a "
            "given instrument, that's a pattern worth knowing before you spend more search budget on it."
        ),
        "roadmap": [
            "After a Forge/Search Lab/Evolution Lab run, check here for what got rejected and why.",
            "Look for repeated failure patterns within one family/instrument before re-running a similar search.",
        ],
        "primary_buttons": [{"label": "Open Strategy Graveyard", "href": "/graveyard"}],
        "tools": [
            {"name": "Strategy Graveyard", "href": "/graveyard", "desc": "Dead neighborhoods, and exactly why they failed."},
        ],
    },
    "quantlab": {
        "title": "Quant Lab", "accent_color": "#67e8f9",
        "tagline": "Standalone quant tools that sit outside the main Create \u2192 Deployment strategy pipeline.",
        "description": (
            "Quant Lab, Options Outlook, and Hedge Fund Manager are each self-contained -- a strategy-code "
            "translator, statistical-arbitrage and options tooling, an options calls/puts outlook, and a "
            "higher-level research \u2192 portfolio \u2192 execution \u2192 oversight workflow. None of them require "
            "having gone through Create/Test/Optimize/Validate first."
        ),
        "roadmap": [
            "Quant Lab covers strategy translation, statistical arbitrage, and other one-off quant tools.",
            "Options Outlook gives a calls/puts read on an instrument.",
            "Hedge Fund Manager is a separate, higher-level research/portfolio/execution/oversight workflow.",
        ],
        "primary_buttons": [{"label": "Open Quant Lab", "href": "/quant-lab"}],
        "tools": [
            {"name": "Quant Lab", "href": "/quant-lab", "desc": "Strategy translator, stat arb, and other quant tools."},
            {"name": "Options Outlook", "href": "/options-outlook", "desc": "Calls/puts outlook for an instrument."},
            {"name": "Hedge Fund Manager", "href": "/hedge-fund", "desc": "Research \u2192 portfolio \u2192 execution \u2192 oversight."},
        ],
    },
    "account": {
        "title": "Account", "accent_color": "#c9d1d9",
        "tagline": "Your profile, every integration's API keys in one place, notifications, and support.",
        "description": (
            "Account settings, one centralized place for every integration's credentials (AI, Trading, Data, "
            "and Broker/Prop-Firm accounts), where job-finished notifications go, and where to get help or "
            "report an issue -- all local to this app, no cloud account required."
        ),
        "roadmap": [
            "Set your name/email/company under Account Settings.",
            "Add every API key/broker account you'll use anywhere in the app under API Keys, once.",
            "Turn on email/Discord/Telegram notifications so a long-running job can tell you when it's done.",
            "Stuck on something? Support has the FAQ, Discord link, and an issue-report form.",
        ],
        "primary_buttons": [{"label": "API Keys", "href": "/settings/api-keys"}, {"label": "Account Settings", "href": "/settings/account"}],
        "tools": [
            {"name": "Account Settings", "href": "/settings/account", "desc": "Name, username, email, company, security."},
            {"name": "API Keys", "href": "/settings/api-keys", "desc": "Every integration's credentials in one place."},
            {"name": "Notification Settings", "href": "/settings/notifications", "desc": "Email, Discord, Telegram job-finished alerts."},
            {"name": "Support", "href": "/support", "desc": "FAQ, Discord, and issue reporting."},
        ],
    },
}


# web href -> desktop nav key (MainWindow._nav_items) of the equivalent tab
DESKTOP_NAV_FOR_HREF = {
    "/forge": "forge", "/generate-strategies": "genstrat", "/research-agent": "researchagent",
    "/research": "researchdirector", "/research-loop": "researchloop", "/speed-run": "speedrun",
    "/speed-run/multi-instrument": "speedrunmulti", "/library": "stratlibrary",
    "/": "run", "/payout-probability": "payout", "/prop-firm-recommender": "propfirmrec",
    "/family-diversity": "familydiversity", "/portfolio": "portfolio", "/ensemble": "ensemble",
    "/overnight-autopilot": "autopilot_pointer", "/compare": "compare",
    "/quant-lab/strategy-health": "strathealth_pointer", "/forward-test": "forwardtest",
    "/deploy-live": "deploylive", "/live-market": "livemarket", "/replay": "replay",
    "/graveyard": "graveyard", "/quant-lab": "quantlab", "/options-outlook": "optionsoutlook",
    "/hedge-fund": "hedgefund", "/settings/api-keys": "apikeys", "/settings/account": "account",
    "/settings/notifications": "account", "/support": "support",
}

# Sections that get a desktop Start Here page. (Optimize and Validate already
# have their own overview pages on both builds.)
DESKTOP_SECTIONS = ("create", "test", "champion", "deployment", "graveyard", "quantlab", "account")
