#!/usr/bin/env python3
"""Generate, assign and revoke license keys for serverless (allowlist) licensing.

Source of truth: licenses/keys_PRIVATE.csv  (plaintext keys -- keep it private,
it is git-ignored on purpose).
Published file:  licenses/allowlist.txt     (SHA-256 hashes of every key that is
NOT revoked -- copy this to the PUBLIC license repo and commit it).

    python build_tools/license_keys.py generate --count 500
    python build_tools/license_keys.py assign T58-ABCDE-... "jane@example.com"
    python build_tools/license_keys.py revoke T58-ABCDE-...
    python build_tools/license_keys.py list [--unassigned|--revoked]
    python build_tools/license_keys.py rebuild      # re-derive allowlist.txt

Every command that changes the CSV also rewrites allowlist.txt. Revoking =
marking the key revoked here, re-running (done automatically), then committing
the new allowlist.txt to the public repo.

Keys look like T58-XXXXX-XXXXX-XXXXX-XXXXX: 20 characters from a 31-symbol
alphabet (no 0/O/1/I/L), ~99 bits from the OS CSPRNG -- far too large to guess,
so publishing only their hashes leaks nothing. The hash is computed exactly the
way app/licensing/client.py checks it: sha256(key.strip().upper()).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import secrets
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "licenses"
CSV_NAME = "keys_PRIVATE.csv"
ALLOWLIST_NAME = "allowlist.txt"
ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # 31 symbols, no look-alikes
FIELDS = ["key", "issued_to", "issued_date", "revoked", "note"]


def make_key() -> str:
    chars = "".join(secrets.choice(ALPHABET) for _ in range(20))
    return "T58-" + "-".join(chars[i:i + 5] for i in range(0, 20, 5))


def key_hash(key: str) -> str:
    return hashlib.sha256(key.strip().upper().encode("utf-8")).hexdigest()


def read_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def write_rows(csv_path: Path, rows: list[dict]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({f: r.get(f, "") for f in FIELDS})


def write_allowlist(out_dir: Path, rows: list[dict]) -> int:
    live = sorted({key_hash(r["key"]) for r in rows if (r.get("revoked") or "").strip().lower() != "yes"})
    text = ("# T58 license allowlist -- SHA-256 hashes of valid keys, one per line.\n"
            "# Delete a line (and commit) to revoke that key.\n" + "\n".join(live) + ("\n" if live else ""))
    (out_dir / ALLOWLIST_NAME).write_text(text, encoding="ascii")
    return len(live)


def _find(rows: list[dict], key: str) -> dict:
    wanted = key.strip().upper()
    for r in rows:
        if r["key"].upper() == wanted:
            return r
    raise SystemExit(f"Key not found in {CSV_NAME}: {key}")


def cmd_generate(out_dir: Path, count: int) -> None:
    csv_path = out_dir / CSV_NAME
    rows = read_rows(csv_path)
    existing = {r["key"] for r in rows}
    added = 0
    while added < count:
        k = make_key()
        if k in existing:
            continue
        existing.add(k)
        rows.append({"key": k, "issued_to": "", "issued_date": "", "revoked": "", "note": ""})
        added += 1
    write_rows(csv_path, rows)
    live = write_allowlist(out_dir, rows)
    print(f"Added {added} keys ({len(rows)} total, {live} active).")
    print(f"  Keys (PRIVATE):   {csv_path}")
    print(f"  Public allowlist: {out_dir / ALLOWLIST_NAME}")


def cmd_assign(out_dir: Path, key: str, issued_to: str) -> None:
    csv_path = out_dir / CSV_NAME
    rows = read_rows(csv_path)
    row = _find(rows, key)
    row["issued_to"] = issued_to
    row["issued_date"] = date.today().isoformat()
    write_rows(csv_path, rows)
    print(f"Assigned {row['key']} to {issued_to}.")


def cmd_revoke(out_dir: Path, key: str) -> None:
    csv_path = out_dir / CSV_NAME
    rows = read_rows(csv_path)
    row = _find(rows, key)
    row["revoked"] = "yes"
    write_rows(csv_path, rows)
    live = write_allowlist(out_dir, rows)
    print(f"Revoked {row['key']} ({row.get('issued_to') or 'unassigned'}). {live} keys remain active.")
    print(f"Now commit {out_dir / ALLOWLIST_NAME} to your PUBLIC license repo for it to take effect.")


def cmd_list(out_dir: Path, unassigned: bool, revoked: bool) -> None:
    for r in read_rows(out_dir / CSV_NAME):
        is_revoked = (r.get("revoked") or "").lower() == "yes"
        if unassigned and (r.get("issued_to") or is_revoked):
            continue
        if revoked and not is_revoked:
            continue
        print(f"{r['key']}  {r.get('issued_to', '') or '-':30}  {r.get('issued_date', '') or '-':10}  "
              f"{'REVOKED' if is_revoked else ''}")


def cmd_rebuild(out_dir: Path) -> None:
    rows = read_rows(out_dir / CSV_NAME)
    print(f"Wrote {out_dir / ALLOWLIST_NAME} with {write_allowlist(out_dir, rows)} active keys.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="where the CSV and allowlist live (default: licenses/)")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="add N new keys")
    g.add_argument("--count", type=int, default=500)
    a = sub.add_parser("assign", help="record who a key was given to")
    a.add_argument("key")
    a.add_argument("issued_to")
    r = sub.add_parser("revoke", help="revoke a key and rewrite allowlist.txt")
    r.add_argument("key")
    ls = sub.add_parser("list", help="show keys")
    ls.add_argument("--unassigned", action="store_true")
    ls.add_argument("--revoked", action="store_true")
    sub.add_parser("rebuild", help="re-derive allowlist.txt from the CSV")
    args = p.parse_args(argv)

    if args.cmd == "generate":
        if args.count < 1:
            raise SystemExit("--count must be at least 1")
        cmd_generate(args.dir, args.count)
    elif args.cmd == "assign":
        cmd_assign(args.dir, args.key, args.issued_to)
    elif args.cmd == "revoke":
        cmd_revoke(args.dir, args.key)
    elif args.cmd == "list":
        cmd_list(args.dir, args.unassigned, args.revoked)
    elif args.cmd == "rebuild":
        cmd_rebuild(args.dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
