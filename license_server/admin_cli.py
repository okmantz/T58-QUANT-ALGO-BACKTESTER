#!/usr/bin/env python3
"""
T58 License Admin CLI -- create, revoke, suspend, reactivate, extend, and
list licenses from the command line, talking to a running license server
over HTTP (so this works identically whether the server is on your own
machine or deployed somewhere).

Usage:
    python admin_cli.py create alice@example.com --plan monthly --days 30
    python admin_cli.py list
    python admin_cli.py show T58-XXXX-XXXX-XXXX-XXXX
    python admin_cli.py revoke T58-XXXX-XXXX-XXXX-XXXX
    python admin_cli.py suspend T58-XXXX-XXXX-XXXX-XXXX
    python admin_cli.py reactivate T58-XXXX-XXXX-XXXX-XXXX
    python admin_cli.py extend T58-XXXX-XXXX-XXXX-XXXX --days 30

Configure once via environment variables (or pass --url/--token every time):
    export T58_LICENSE_SERVER_URL=https://your-license-server.example.com
    export T58_LICENSE_ADMIN_TOKEN=the-same-token-the-server-was-started-with
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def _request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-Admin-Token", token)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {"ok": False, "error": f"HTTP {exc.code}: {body}"}
    except urllib.error.URLError as exc:
        return {"ok": False, "error": f"Could not reach {url}: {exc.reason}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.environ.get("T58_LICENSE_SERVER_URL", "http://localhost:5055"),
                         help="Base URL of the license server (default: $T58_LICENSE_SERVER_URL or localhost:5055)")
    parser.add_argument("--token", default=os.environ.get("T58_LICENSE_ADMIN_TOKEN", ""),
                         help="Admin token (default: $T58_LICENSE_ADMIN_TOKEN)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Issue a new license")
    p_create.add_argument("email")
    p_create.add_argument("--plan", default="")
    p_create.add_argument("--days", type=int, default=None, help="Fixed expiry N days from now; omit for no fixed expiry (Whop webhook controls access instead)")
    p_create.add_argument("--whop-membership-id", default=None, help="Whop membership ID, so the webhook can auto-revoke this license on cancellation")

    sub.add_parser("list", help="List all licenses")

    p_show = sub.add_parser("show", help="Show one license")
    p_show.add_argument("license_key")

    for name in ("revoke", "suspend", "reactivate"):
        p = sub.add_parser(name, help=f"Set a license's status to {name}d" if name != "reactivate" else "Set status back to active")
        p.add_argument("license_key")

    p_extend = sub.add_parser("extend", help="Push a license's expiry out")
    p_extend.add_argument("license_key")
    p_extend.add_argument("--days", type=int, required=True)

    args = parser.parse_args()

    if not args.token:
        print("Error: no admin token. Pass --token or set $T58_LICENSE_ADMIN_TOKEN.", file=sys.stderr)
        return 1
    base = args.url.rstrip("/")

    if args.command == "create":
        result = _request("POST", f"{base}/admin/licenses", args.token, {
            "email": args.email, "plan": args.plan, "days": args.days,
            "whop_membership_id": args.whop_membership_id,
        })
    elif args.command == "list":
        result = _request("GET", f"{base}/admin/licenses", args.token)
    elif args.command == "show":
        result = _request("GET", f"{base}/admin/licenses/{args.license_key}", args.token)
    elif args.command in ("revoke", "suspend", "reactivate"):
        result = _request("POST", f"{base}/admin/licenses/{args.license_key}/{args.command}", args.token)
    elif args.command == "extend":
        result = _request("POST", f"{base}/admin/licenses/{args.license_key}/extend", args.token, {"days": args.days})
    else:
        parser.print_help()
        return 1

    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
