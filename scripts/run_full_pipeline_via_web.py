"""Drive the REAL web route /full-pipeline/start (same code path the browser uses) and poll to completion."""
import sys, os, io, json, time, warnings
warnings.filterwarnings("ignore")
ROOT=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")); sys.path.insert(0,ROOT); os.chdir(ROOT)
sys.path.insert(0, ROOT+"/scripts")
def main():
    import web_smoke_test as smoke
    from app.web import server
    restore = smoke._activate_smoke_license()
    try:
        strat_path, dataset, contract, commission, out_json = sys.argv[1:6]
        extra = json.loads(sys.argv[6]) if len(sys.argv) > 6 else {}
        form = {
            "existing_dataset": dataset, "strategy_mode": "python",
            "initial_balance": "50000", "risk_mode": "percent", "risk_value": "1.0",
            "commission": commission, "pip_size": "1.0", "contract_size": contract,
            "account_size": "50000", "profit_target": "6", "daily_loss": "100", "max_dd": "4",
            "prop_preset": "topstep_50k", "timeframe": "",
        }
        form.update(extra)
        c = server.app.test_client()
        data = dict(form); data["strategy_file"] = (io.BytesIO(open(strat_path, "rb").read()), os.path.basename(strat_path))
        r = c.post("/full-pipeline/start", data=data, content_type="multipart/form-data", follow_redirects=False)
        print("POST", r.status_code, r.headers.get("Location"), flush=True)
        if r.status_code not in (301, 302, 303):
            print(r.get_data(as_text=True)[-3000:]); sys.exit(2)
        job_id = r.headers["Location"].rstrip("/").split("/")[-1]
        t0 = time.time(); last = 0
        while True:
            s = c.get(f"/full-pipeline/job/{job_id}/status.json").get_json()
            if len(s["log"]) > last:
                for line in s["log"][last:]: print(line, flush=True)
                last = len(s["log"])
            if s["done"] or s.get("error") or s.get("cancelled"): break
            time.sleep(15)
        print("ELAPSED", round(time.time() - t0), "ERROR", s.get("error"))
        json.dump(s, open(out_json, "w"), indent=1, default=str)
        if s.get("summary"): print(json.dumps({k: v for k, v in s["summary"].items() if k != "warnings"}, indent=1, default=str))
    finally:
        restore(); server._license_ok_cached = None


if __name__ == '__main__':
    main()
