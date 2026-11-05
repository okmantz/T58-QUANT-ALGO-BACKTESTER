"""One place that states the bars a READY / HIGH-CONFIDENCE verdict must clear, read from the live pipeline defaults.

Templates, the metric popups and the docs quote these numbers; they come from ``FullPipelineConfig`` so a change to a default
changes every sentence that mentions it (2026-11 consistency pass -- the side panels used to hard-code stale values).
"""
from __future__ import annotations


def ready_bar() -> dict:
    from app.orchestration.full_pipeline import FullPipelineConfig
    from app.prop.presets import RULES_STAMP_MAX_AGE_DAYS

    c = FullPipelineConfig()
    out = {
        "min_trades": int(c.min_trades_for_ready),
        "preflight_trades": int(c.preflight_min_trades),
        "pass_pct": float(c.min_per_attempt_pass_pct),
        "breach_cap_pct": float(c.risk_of_ruin_cap),
        "funded_bust_cap_pct": float(c.funded_breach_cap_pct),
        "holdout_pct": round(float(c.holdout_frac) * 100),
        "rules_max_age_days": int(RULES_STAMP_MAX_AGE_DAYS),
    }
    out["line_ready"] = (f">= {out['pass_pct']:.0f}% per-attempt pass (lower bound), <= {out['breach_cap_pct']:.0f}% evaluation breach, "
                         f">= {out['min_trades']} trades")
    out["holdout_text"] = f"last {out['holdout_pct']}% locked (Advanced)"
    out["ready_short"] = f">= {out['pass_pct']:.0f}% pass, >= {out['min_trades']} trades"
    return out
