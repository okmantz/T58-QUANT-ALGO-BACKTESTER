"""Freeze (or verify) the T58 market data pack: python scripts/freeze_data_pack.py [--verify]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data import data_pack  # noqa: E402

if __name__ == "__main__":
    if "--verify" in sys.argv:
        r = data_pack.verify()
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    print("wrote", data_pack.freeze())
