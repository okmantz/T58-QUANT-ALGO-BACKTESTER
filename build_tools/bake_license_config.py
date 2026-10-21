#!/usr/bin/env python3
"""Write app/licensing/build_config.py from the release secrets (any OS).

Reads T58_LICENSE_SERVER_URL and T58_MASTER_LICENSE_KEY_HASH from the
environment (both optional: absent values are written empty and the app then
reports it loudly at startup). Shared by the web and desktop workflows so the
same logic is not pasted into several YAML files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config = {
        "LICENSE_SERVER_URL": os.environ.get("T58_LICENSE_SERVER_URL", ""),
        "MASTER_LICENSE_KEY_HASH": os.environ.get("T58_MASTER_LICENSE_KEY_HASH", ""),
        # Serverless mode: raw URL of the public allowlist.txt (SHA-256 hashes of valid keys).
        "LICENSE_ALLOWLIST_URL": os.environ.get("T58_LICENSE_ALLOWLIST_URL", ""),
        # Google Sheet mode: the Apps Script web-app URL from license_sheet/SETUP.md.
        "LICENSE_SHEET_URL": os.environ.get("T58_LICENSE_SHEET_URL", ""),
    }
    header = ("# AUTO-GENERATED at build time by the release workflow from the "
              "T58_LICENSE_SERVER_URL / T58_MASTER_LICENSE_KEY_HASH repo secrets. "
              "Never edit by hand; never commit (see .gitignore).\n")
    out = ROOT / "app" / "licensing" / "build_config.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join([header] + [f"{k} = {json.dumps(v)}" for k, v in config.items()]) + "\n", encoding="ascii")
    print(f"Wrote {out.relative_to(ROOT)} -- URL set: {bool(config['LICENSE_SERVER_URL'])}, "
          f"master-key hash set: {bool(config['MASTER_LICENSE_KEY_HASH'])}")
    if config["LICENSE_SHEET_URL"]:
        print("Google Sheet mode: the app will check keys against the baked LICENSE_SHEET_URL (no server needed).")
    elif config["LICENSE_ALLOWLIST_URL"]:
        print("Allowlist mode: the app will check keys against the baked LICENSE_ALLOWLIST_URL (no server needed).")
    elif not config["LICENSE_SERVER_URL"]:
        print("WARNING: T58_LICENSE_SERVER_URL secret is not set -- this build will fail loudly at startup "
              "(no license server URL configured). See license_server/DEPLOY.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
