"""Plain-English summary for a report: what the verdict means, in a few sentences a trader can read in ten seconds.

Everything here is derived from numbers already in the report dict, never invented. Missing numbers are skipped, not guessed.
`summarize(report)` returns a dict with a headline, a short list of bullets ("what the numbers say"), the main concerns, and a
suggested next step; `summary_html(report)` renders it as one card for the top of the HTML report.
"""
from __future__ import annotations

import html
import math

_NEXT_STEP = {
    "FORWARD-TEST CANDIDATE": "Next step: run it on the T58 demo account (Deploy). A backtest can only retire risks, never prove an edge, so let it trade forward on money that is not real before anything else.",
    "EVALUATE": "Next step: forward test it, then take a prop-firm evaluation only if the forward results agree with this report.",
    "FUNDED": "Next step: keep the live results tracked against this report. Stand down if they drift well outside the simulated range.",
    "LEAD": "Next step: treat this as a lead, not a finished strategy. Run the Validate checks and look at the concerns below before spending more time on it.",
    "REJECT": "Next step: do not trade this version. Check the concerns below, and if you try a fix, treat it as a new idea (a new test) rather than a tweak to the same result.",
}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _money(v: float) -> str:
    return f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}"


def _stage(report: dict) -> str:
    try:
        from app.orchestration.verdict_labels import headline
        return headline(report.get("verdict"), (report.get("ladder") or {}).get("stage"), report.get("verdict_reasons"))
    except Exception:  # noqa: BLE001
        return str(report.get("verdict") or "UNTESTED").upper()


def _pass_sentence(p: float | None) -> str | None:
    if p is None:
        return None
    if p >= 70:
        word = "a strong rate"
    elif p >= 50:
        word = "a middling rate"
    else:
        word = "a weak rate"
    return f"In the Monte Carlo prop-firm simulation, {p:.0f} of every 100 account attempts passed the evaluation, {word}."


def summarize(report: dict) -> dict:
    strat = report.get("strategy") or {}
    stats = (report.get("historical_backtest") or {}).get("statistics") or {}
    mc = report.get("monte_carlo") or {}
    stage = _stage(report)
    bullets: list[str] = []
    concerns: list[str] = []

    trades, net, wr, pf, dd = (_num(stats.get(k)) for k in ("total_trades", "net_profit", "win_rate", "profit_factor", "max_drawdown_pct"))
    if trades is not None:
        bit = f"Over {strat.get('backtest_period_start', 'the test period')} to {strat.get('backtest_period_end', '')}".replace(" to ", " to ").strip()
        bit = f"{bit}, the strategy took {int(trades):,} trades"
        if net is not None:
            bit += f" and made {_money(net)} in total" if net >= 0 else f" and lost {_money(abs(net))} in total"
        bullets.append(bit + ".")
    if wr is not None and pf is not None:
        bullets.append(f"It won {wr:.0f}% of its trades, and its winners paid {pf:.2f}x what its losers cost (profit factor)." if pf >= 1
                       else f"It won {wr:.0f}% of its trades, but its winners paid only {pf:.2f}x what its losers cost (profit factor below 1 means it loses money).")
    if dd is not None:
        bullets.append(f"The worst peak-to-trough drop was {abs(dd):.1f}% of the account.")
    try:
        from app.reports.mc_headline import monte_carlo_headline
        mch = monte_carlo_headline(mc) if mc else {}
    except Exception:  # noqa: BLE001
        mch = {}
    s = _pass_sentence(_num(mch.get("eval_pass_probability")))
    if s:
        bullets.append(s)
    ror = _num(mc.get("risk_of_ruin_pct"))
    if ror is not None and ror >= 20:
        concerns.append(f"Risk of ruin is {ror:.0f}%: in that share of simulated paths the account eventually breaks its drawdown rule.")

    conc = report.get("concentration_check") or {}
    best_pct = _num(conc.get("best_trade_pct_of_gross_profit"))
    ex_best = _num(conc.get("net_profit_excluding_best_trade"))
    if ex_best is not None and ex_best <= 0:
        concerns.append("Take away the single best trade and the strategy no longer makes money. One lucky trade is carrying the result.")
    elif best_pct is not None and best_pct >= 25:
        concerns.append(f"The best single trade is {best_pct:.0f}% of all gross profit, which is a lot to rest on one trade.")
    ex_day = _num(conc.get("net_profit_excluding_best_day"))
    if ex_day is not None and ex_day <= 0:
        concerns.append("Take away the single best day and the strategy no longer makes money.")

    ho = report.get("holdout_comparison") or {}
    ho_net = _num((ho.get("holdout_statistics") or {}).get("net_profit"))
    in_net = _num((ho.get("in_sample_statistics") or {}).get("net_profit"))
    if ho_net is not None and in_net is not None and in_net > 0 and ho_net <= 0:
        concerns.append("It made money on the data it was tuned on but not on the final stretch it never saw. That is the classic sign of an overfit strategy.")

    warns = [w for w in (report.get("headline_warnings") or []) if w]
    if warns:
        concerns.append(f"The run raised {len(warns)} warning{'s' if len(warns) != 1 else ''} (see the amber boxes below); read those before trusting the numbers.")
    for r in (report.get("verdict_reasons") or [])[:2]:
        txt = str(r).strip()
        if txt and txt not in concerns:
            concerns.append(txt if txt.endswith((".", "!", "?")) else txt + ".")

    headline = {
        "FORWARD-TEST CANDIDATE": "This strategy cleared every check in this run.",
        "EVALUATE": "This strategy cleared every check and is ready for an evaluation.",
        "FUNDED": "This strategy is at the funded stage of the ladder.",
        "LEAD": "This is a promising lead, not yet a strategy you can rely on.",
        "REJECT": "This strategy failed a hard check and should not be traded as it stands.",
    }.get(stage, "Here is what this run found.")
    return {"stage": stage, "headline": headline, "bullets": bullets, "concerns": concerns[:5], "next_step": _NEXT_STEP.get(stage, "")}


def summary_html(report: dict) -> str:
    try:
        s = summarize(report)
    except Exception:  # noqa: BLE001 -- an explanatory card must never break a report
        return ""
    if not s["bullets"] and not s["concerns"]:
        return ""
    e = html.escape
    parts = [f'<section class="sec plain" id="plain"><h2>In plain English</h2><p class="plain-lead">{e(s["headline"])}</p>']
    if s["bullets"]:
        parts.append("<ul>" + "".join(f"<li>{e(b)}</li>" for b in s["bullets"]) + "</ul>")
    if s["concerns"]:
        parts.append('<h3>What to watch</h3><ul class="plain-concerns">' + "".join(f"<li>{e(c)}</li>" for c in s["concerns"]) + "</ul>")
    if s["next_step"]:
        parts.append(f'<p class="plain-next">{e(s["next_step"])}</p>')
    parts.append("</section>")
    return "".join(parts)
