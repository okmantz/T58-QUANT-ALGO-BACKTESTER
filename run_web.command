#!/usr/bin/env bash
# macOS: double-click this file in Finder to start the T58 web app.
# (First time only: right-click -> Open, or run `chmod +x run_web.command` in Terminal.)
cd "$(dirname "$0")"
bash ./run_web.sh
echo; read -r -p "Press Return to close this window..."
