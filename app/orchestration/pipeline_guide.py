"""
Shared "what to do next" guidance for the strategy lifecycle:

    create (Evolution Lab / Search Lab / Manual Builder / upload)
        -> promote to Strategy Library
        -> validate (Validation Lab: walk-forward, CPCV, sensitivity, ...)
        -> optimize (Quick Optimize / Full Pipeline's own GA step)
        -> Full Pipeline (the all-in-one re-validated verdict)
        -> promote to champion (save_to_library / manual promote)

Both the web app and the desktop app import from HERE rather than each
writing their own "next, go do X" text, so the two UIs can never drift
apart or disagree about what the recommended next step is at a given
stage -- see app.web.server's /evolution/promote, /search/job/.../promote,
Full Pipeline status routes, and app.ui.main_window's matching tabs for
the call sites.

Every function here is a pure string-builder: no I/O, no side effects,
safe to call from a web route's JSON response or a desktop Tkinter log
line equally.
"""
from __future__ import annotations


def after_evolution_stop(leaderboard_size: int) -> str:
    if leaderboard_size <= 0:
        return (
            "Next step: no candidate cleared the pre-filter yet. Let it run more generations, or try a "
            "different family selection -- there's nothing to promote until at least one candidate "
            "appears on the leaderboard."
        )
    return (
        f"Next step: pick your best candidate from the {leaderboard_size}-strategy leaderboard above and "
        "click PROMOTE. That saves it to the Strategy Library -- from there, run it through the "
        "Validation Lab (walk-forward + CPCV) or straight through Full Pipeline for a re-validated verdict."
    )


def after_promote_to_library(filename: str) -> str:
    return (
        f"Saved '{filename}' to the Strategy Library. Next step: run it through Full Pipeline (recommended -- "
        "baseline, walk-forward GA re-optimization, out-of-sample check, holdout check, and a final "
        "READY/MARGINAL/NOT READY verdict in one pass), or the Validation Lab first if you specifically "
        "want walk-forward/CPCV/sensitivity results on their own before optimizing."
    )


def after_search_complete(champion_candidate_id: str | None, leaderboard_size: int) -> str:
    if not champion_candidate_id:
        if leaderboard_size > 0:
            return (
                "Next step: no candidate passed every Stage 3 gate, but the leaderboard above has "
                f"{leaderboard_size} candidate(s) that made it partway. Promote the top one and run it "
                "through Full Pipeline anyway, or widen the search (more families / more candidates) and "
                "run again."
            )
        return (
            "Next step: nothing survived Stage 1 at all. Try a wider family selection, a higher "
            "max-candidates count, or looser Stage 1 filters (min trades / min profit factor) and search again."
        )
    return (
        f"Champion candidate found: {champion_candidate_id}. Next step: click PROMOTE next to it, then run "
        "it through Full Pipeline for a re-validated, walk-forward-optimized final verdict before "
        "considering it for a live prop-firm evaluation."
    )


def after_full_pipeline(verdict: str | None, saved_to_library: bool) -> str:
    verdict = (verdict or "").upper()
    library_note = " It was saved to the Strategy Library." if saved_to_library else ""
    if verdict == "READY":
        return (
            f"Verdict: READY.{library_note} Next step: this is your strongest evidence yet, but it's still "
            "a backtest -- consider a short forward-test (MT5 Forward Test tab, or paper trading) before "
            "risking real capital on a prop-firm evaluation."
        )
    if verdict == "MARGINAL":
        return (
            f"Verdict: MARGINAL.{library_note} Next step: it's not clearly ready or clearly dead. Try "
            "Quick Optimize on it, or adjust risk settings (position size, daily-loss limit) and re-run "
            "Full Pipeline -- don't take it live as-is."
        )
    if verdict == "NOT READY":
        return (
            "Verdict: NOT READY. Next step: this strategy doesn't hold up under re-validation. Go back to "
            "Evolution Lab or Search Lab for a different candidate rather than trying to rescue this one."
        )
    return "Next step: open the full report above for the detailed verdict and what drove it."


def after_full_pipeline_batch(outcomes: list[dict]) -> str:
    """outcomes: list of {"label", "ok", "verdict", "eval_pass_probability"} dicts,
    same shape as app.orchestration.full_pipeline.FullPipelineBatchOutcome."""
    ready = [o for o in outcomes if o.get("ok") and (o.get("verdict") or "").upper() == "READY"]
    if ready:
        best = max(ready, key=lambda o: o.get("eval_pass_probability") or 0)
        return (
            f"Next step: '{best['label']}' came back READY with the highest eval-pass probability "
            f"({best.get('eval_pass_probability', 0):.1f}%) of the batch -- open its report above, then "
            "consider a short forward-test before a live prop-firm evaluation."
        )
    marginal = [o for o in outcomes if o.get("ok") and (o.get("verdict") or "").upper() == "MARGINAL"]
    if marginal:
        return (
            "Next step: nothing in this batch came back READY, but "
            f"{len(marginal)} strateg{'y is' if len(marginal) == 1 else 'ies are'} MARGINAL -- open their "
            "reports, try Quick Optimize on the best of them, or adjust risk settings and re-run."
        )
    return (
        "Next step: nothing in this batch came back READY or MARGINAL. Go back to Evolution Lab or "
        "Search Lab for different candidates rather than optimizing any of these further."
    )
