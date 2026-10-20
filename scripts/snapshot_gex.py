"""Daily gamma-exposure snapshot + SqueezeMetrics history refresh.

    python -m scripts.snapshot_gex            # run once per trading day, after the close

Free sources have no deep GEX history, so we BUILD it: each run records today's live levels (app.data.gamma_data.record_snapshot) and
refreshes the SqueezeMetrics DIX/GEX daily file. A year of daily runs is a year of real gamma levels for the gex_* families.
"""
from app.data import gamma_data

if __name__ == "__main__":
    for inst in ("ES", "NQ"):
        try:
            print(inst, gamma_data.record_snapshot(gamma_data.live_gex(inst)))
        except Exception as exc:  # noqa: BLE001
            print(f"{inst}: {exc}")
    try:
        df = gamma_data.fetch_squeezemetrics_history()
        print("SqueezeMetrics rows:", len(df))
    except Exception as exc:  # noqa: BLE001
        print("SqueezeMetrics:", exc)
