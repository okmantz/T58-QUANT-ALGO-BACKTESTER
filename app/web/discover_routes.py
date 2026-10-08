"""'Your idea' screen: type an idea, see the compiled rule spec, the market x timeframe grid and the
break-it battery as they finish. Backed by app.discovery (idea compiler, hypothesis store, grid runner,
battery) and the shared job manager, so a run survives navigating away."""
from __future__ import annotations

import html
import threading

from flask import Blueprint, jsonify, redirect, render_template_string, request

discover_bp = Blueprint("discover", __name__)

_PAGE = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your idea - T58</title><link rel="stylesheet" href="/static/theme.css">
<style>.card{background:var(--panel-2);border:1px solid var(--border);border-radius:12px;padding:14px;margin:14px;max-width:960px}
textarea,input,select{width:100%;padding:10px;border-radius:8px;border:1px solid var(--border-light);background:var(--panel-3);color:var(--text)}
pre{white-space:pre-wrap;font-size:13px} table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid var(--border);padding:6px;text-align:left;font-size:13px}</style></head><body>
<div class="card"><h2>Your idea</h2>
<p>Describe a trading idea in plain words. It becomes a validated rule, runs across markets and timeframes (counted against every variant tried),
then the strongest cell is attacked: random-entry null, held-out data, 1.5x costs, volatility regimes, parameter neighbours, time folds, other markets.</p>
{% if error %}<p style="color:#F05B63">{{ error }}</p>{% endif %}
<form method="post" action="/discover/run">
<input type="hidden" name="_csrf_token" value="{{ csrf_token() }}">
<label>Idea</label><textarea name="idea" rows="3" placeholder="buy the first retrace into a fair value gap, stop beyond the far edge, target 2R" required></textarea>
<label>Datasets (hold Ctrl/Cmd to pick several markets)</label>
<select name="datasets" multiple size="10">{% for d in datasets %}<option value="{{ d }}">{{ d }}</option>{% endfor %}</select>
<label>Timeframes (comma separated; empty = the data's own)</label><input name="timeframes" placeholder="15min, 1h">
<label>Max bars per dataset (most recent)</label><input name="max_bars" type="number" value="60000">
<label>Random-entry null draws</label><input name="n_null" type="number" value="200">
<p><button type="submit">Run the idea</button></p></form></div>
<div class="card"><h2>Hypotheses on file</h2>
{% if mined %}<p>Mined {{ mined }} new hypothesis(es) from the research library.</p>{% endif %}
<p>Every idea tested here is stored as a hypothesis with its experiments attached, so "have we tested this, on which markets, and what happened?" has an answer. Papers become hypotheses too: mine the research library and the claims land here as <i>proposed</i>, ready to run.</p>
<form method="post" action="/discover/mine"><input type="hidden" name="_csrf_token" value="{{ csrf_token() }}"><p><button type="submit">Mine research library for hypotheses</button></p></form>
{% if hypotheses %}<table><tr><th>Status</th><th>Idea</th><th>Markets</th><th>Timeframes</th><th>Experiments</th><th>Source</th></tr>
{% for h in hypotheses %}<tr><td>{{ h.status }}</td><td>{{ h.idea }}</td><td>{{ h.markets }}</td><td>{{ h.timeframes }}</td><td>{{ h.n_experiments }}</td><td>{{ h.source }}</td></tr>{% endfor %}</table>
{% else %}<p>No hypotheses stored yet.</p>{% endif %}
</div></body></html>"""

_JOB = """<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your idea - result</title><link rel="stylesheet" href="/static/theme.css">
<style>.card{background:var(--panel-2);border:1px solid var(--border);border-radius:12px;padding:14px;margin:14px;max-width:960px}pre{white-space:pre-wrap;font-size:13px}
table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid var(--border);padding:6px;text-align:left;font-size:13px}</style></head><body>
<div class="card"><h2 id="t">Running...</h2><pre id="log"></pre><div id="res"></div></div>
<script>
const id="{{ job_id }}";
async function poll(){const r=await fetch("/discover/status/"+id);const j=await r.json();
 document.getElementById("log").textContent=(j.log||[]).join("\\n");
 if(j.done){document.getElementById("t").textContent=j.error?"Failed":"Finished";document.getElementById("res").innerHTML=j.error?("<p>"+j.error+"</p>"):(j.html||"");return;}
 setTimeout(poll,2500);} poll();
</script></body></html>"""


def _load_dataset(name: str, max_bars: int):
    from app.data.importer import import_csv
    from app.data.storage import resolve_stored_dataset
    path = resolve_stored_dataset(name)
    if path is None:
        raise ValueError(f"unknown dataset {name!r}")
    res = import_csv(str(path))
    if not res.is_valid:
        raise ValueError(f"could not read {name!r}")
    df = res.dataframe
    return df.iloc[-int(max_bars):].reset_index(drop=True) if max_bars and len(df) > max_bars else df


def _result_html(run) -> str:
    from app.reports.generator import battery_section_html
    h = run.hypothesis
    rows = "".join(
        f"<tr><td>{html.escape(str(c.market))}</td><td>{html.escape(str(c.timeframe))}</td><td>{c.n}</td><td>{c.mean_r:+.3f}</td>"
        f"<td>${c.net:,.0f}</td><td>{'n/a' if c.psr_deflated is None else f'{c.psr_deflated:.2f}'}</td></tr>" for c in run.cells)
    spec = html.escape(str(h.spec))
    return (f"<h3>Rule</h3><pre>{spec}</pre><p>{html.escape(h.mechanism)}</p>"
            + ("<p><b>Notes:</b> " + html.escape("; ".join(h.warnings)) + "</p>" if h.warnings else "")
            + f"<h3>Grid ({run.n_trials} trials counted for deflation)</h3><table><tr><th>Market</th><th>Timeframe</th><th>Trades</th><th>Mean R</th><th>Net</th><th>Deflated PSR</th></tr>{rows}</table>"
            + (battery_section_html(run.battery.to_dict()) if run.battery else "<p>No cell had enough trades to attack.</p>")
            + f"<p>Status: <b>{html.escape(h.status)}</b>. Every cell and attack is recorded against hypothesis <code>{h.id}</code>.</p>")


def _worker(job_id: str, idea: str, names: list, tfs: list, max_bars: int, n_null: int):
    from app.backtest.risk import RiskConfig
    from app.web.accuracy_form import harden_risk_config
    from app.data.instrument_specs import apply_any_instrument_spec as apply_instrument_spec, guess_any_instrument_symbol as guess_instrument_symbol
    from app.discovery.experiment_runner import run_hypothesis
    from app.discovery.hypothesis import HypothesisStore
    from app.discovery.idea_compiler import compile_idea
    from app.ai.llm_client import preferred_client
    from app.web.job_manager import JOB_MANAGER
    log = lambda m: JOB_MANAGER.log(job_id, m)  # noqa: E731
    try:
        log("Compiling the idea into a rule...")
        client = preferred_client()
        hyp = compile_idea(idea, llm=None if client.__class__.__name__ == "NullClient" else client, markets=names, timeframes=tfs)
        log(f"Rule ({hyp.source}): {hyp.spec}")
        datasets = {}
        for n in names:
            log(f"Loading {n}...")
            datasets[n] = _load_dataset(n, max_bars)
        sym = guess_instrument_symbol(names[0]) if names else None
        base = harden_risk_config(
            RiskConfig(initial_balance=50_000.0, risk_mode="fixed", risk_value=500.0, sizing_mode="fit_stop", max_trades_per_day=20),
            instrument=sym,
        )
        risk = apply_instrument_spec(base, sym) if sym else base
        log("Running the grid and the break-it battery (this can take several minutes)...")
        run = run_hypothesis(hyp, datasets, risk, store=HypothesisStore(), n_null=n_null)
        JOB_MANAGER.finish(job_id, html=_result_html(run))
    except Exception as exc:  # noqa: BLE001
        JOB_MANAGER.fail(job_id, str(exc))


@discover_bp.route("/discover")
def discover_form():
    from app.data.storage import list_stored_datasets
    hypotheses = []
    try:
        from app.discovery.hypothesis import HypothesisStore

        for h in HypothesisStore().all()[-25:][::-1]:
            hypotheses.append({
                "status": h.status, "idea": h.idea[:140],
                "markets": ", ".join(map(str, h.markets)), "timeframes": ", ".join(map(str, h.timeframes)),
                "n_experiments": len(h.experiments), "source": h.source,
            })
    except Exception:  # noqa: BLE001 -- the form must render even if the store is unreadable
        hypotheses = []
    return render_template_string(_PAGE, datasets=[d.name for d in list_stored_datasets()], error=request.args.get("error"),
                                  hypotheses=hypotheses, mined=request.args.get("mined"),
                                  csrf_token=lambda: __import__("flask").session.get("csrf_token", ""))


@discover_bp.route("/discover/mine", methods=["POST"])
def discover_mine():
    """Papers -> stored, citable hypotheses (app.discovery.paper_hypotheses).

    Before v9.5 this pipeline existed as a library with no user-reachable
    caller; the only way research became a testable claim was a human
    reading the PDF. Mining stores each claim as a 'proposed' hypothesis
    (testable ones carry a compiled rule spec; untestable ones say why),
    so the next step -- pick it in the idea box and run it -- is one
    click of work, not a research project."""
    try:
        from app.ai.llm_client import preferred_client
        from app.discovery.hypothesis import HypothesisStore
        from app.discovery.paper_hypotheses import extract_from_library

        client = preferred_client()
        found = extract_from_library(
            llm=None if client.__class__.__name__ == "NullClient" else client,
            store=HypothesisStore(),
        )
        return redirect(f"/discover?mined={len(found)}")
    except Exception as exc:  # noqa: BLE001
        return redirect(f"/discover?error=Mining+failed:+{exc.__class__.__name__}")


@discover_bp.route("/discover/run", methods=["POST"])
def discover_run():
    from app.web.job_manager import JOB_MANAGER
    idea = (request.form.get("idea") or "").strip()
    names = request.form.getlist("datasets")
    if not idea or not names:
        return redirect("/discover?error=Describe+an+idea+and+pick+at+least+one+dataset")
    tfs = [t.strip() for t in (request.form.get("timeframes") or "").split(",") if t.strip()]
    job_id = JOB_MANAGER.create(tool="Your idea", page_template="/discover/job/{job_id}")
    threading.Thread(target=_worker, daemon=True, args=(
        job_id, idea, names, tfs, int(request.form.get("max_bars") or 60000), int(request.form.get("n_null") or 200))).start()
    return redirect(f"/discover/job/{job_id}")


@discover_bp.route("/discover/job/<job_id>")
def discover_job(job_id: str):
    return render_template_string(_JOB, job_id=job_id)


@discover_bp.route("/discover/status/<job_id>")
def discover_status(job_id: str):
    from app.web.job_manager import JOB_MANAGER
    j = JOB_MANAGER.get(job_id)
    if j is None:
        return jsonify({"done": True, "error": "unknown job"}), 404
    return jsonify({"done": bool(j.get("done")), "error": j.get("error"), "log": j.get("log", [])[-200:], "html": j.get("html")})
