#!/usr/bin/env bash
# AUDIT 2026-10-12 (F-SECRETS): data/config/mt5_password.txt and mt5_settings.json
# are committed even though .gitignore lists them. Run from the repo root.
# 1) stop tracking (local files are kept); 2) commit; 3) ROTATE the MT5 password
# (it is in git history); 4) optionally scrub history with git-filter-repo.
set -euo pipefail
git rm --cached --ignore-unmatch data/config/mt5_password.txt data/config/mt5_settings.json
git commit -m "Stop tracking MT5 credentials (kept locally, already in .gitignore)"
echo "NEXT: change your MT5 account password now. To scrub history:"
echo "  pip install git-filter-repo && git filter-repo --path data/config/mt5_password.txt --path data/config/mt5_settings.json --invert-paths"
