"""E2E: POST the Optimizer + Validation routes with a small real dataset and make sure each job FINISHES without error."""
import sys, os, time, json, warnings
warnings.filterwarnings("ignore")
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0,ROOT); os.chdir(ROOT); sys.path.insert(0,ROOT+"/scripts")
BASE = {"existing_dataset": "NQ1!/NQ_5min_20260120_20260415.csv", "strategy_mode": "manual", "sma_fast": "10", "sma_slow": "30",
        "sl_pips": "15", "tp_pips": "30", "initial_balance": "50000", "risk_mode": "percent", "risk_value": "1.0", "commission": "4.2",
        "pip_size": "1.0", "contract_size": "20", "account_size": "50000", "profit_target": "6", "daily_loss": "100", "max_dd": "4",
        "prop_preset": "topstep_50k", "timeframe": ""}
SMALL = {"n_folds": "2", "ga_population": "4", "ga_generations": "2", "ga_search_mc_sims": "50", "final_mc_sims": "300", "baseline_mc_sims": "200",
         "mc_sims": "200", "n_sims": "200", "n_simulations": "200", "max_evals": "12", "n_trials": "10", "population": "4", "generations": "2"}
ROUTES = [("/optimize-simple/start", {"engine_quick_optimize": "on"}),
          ("/quick-optimize/start", {}),
          ("/walk-forward-opt/start", {}), ("/walk-forward-ga/start", {}),
          ("/cpcv/start", {}), ("/pbo/start", {"n_perturbed_variants": "4"}), ("/sensitivity/start", {}), ("/parameter-robustness/start", {}),
          ("/regime-matrix/run", {}), ("/payout-probability/run", {})]
def main():
    import web_smoke_test as smoke
    from app.web import server
    restore = smoke._activate_smoke_license()
    only = sys.argv[1:]
    results = []
    try:
        c = server.app.test_client()
        for route, extra in ROUTES:
            if only and route not in only: continue
            form = dict(BASE); form.update(SMALL); form.update(extra)
            t0 = time.time()
            r = c.post(route, data=form, content_type="multipart/form-data")
            if r.status_code in (301, 302, 303):
                loc = r.headers["Location"]; sj = loc.rstrip("/") + "/status.json"
                status = None
                while time.time() - t0 < 900:
                    time.sleep(5)
                    sr = c.get(sj)
                    if sr.status_code != 200: status = f"status.json {sr.status_code}"; break
                    j = sr.get_json()
                    if j.get("done") or j.get("error"):
                        status = "ERROR: " + str(j.get("error"))[:300] if j.get("error") else "ok"; break
                else: status = "TIMEOUT"
            else:
                body = r.get_data(as_text=True)
                status = "ok(sync)" if r.status_code == 200 and "BLOCKED" not in body and "lc-error" not in body.split("<body")[-1][:4000] else f"HTTP {r.status_code}"
                if status.startswith("HTTP") or status == "ok(sync)" and False:
                    import re; m = re.search(r'class="lc-error[^"]*"[^>]*>(.*?)</div>', body, re.S); status += " " + (re.sub(r"<[^>]+>", "", m.group(1)).strip()[:300] if m else "")
            print(f"{route:34s} {status}  ({round(time.time()-t0)}s)", flush=True)
            results.append((route, status))
    finally:
        restore(); server._license_ok_cached = None
    bad = [r for r in results if not r[1].startswith("ok")]
    print("FAILED:", bad if bad else "none")
if __name__ == "__main__":
    main()
