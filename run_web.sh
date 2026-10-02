#!/usr/bin/env bash
# One-command launcher for the T58 web app on Linux and macOS.
#   bash run_web.sh        (or ./run_web.sh after: chmod +x run_web.sh)
# First run creates a private virtual environment (.venv) and installs the dependencies.
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.10+ is required but '$PY' was not found. Install it from https://www.python.org/downloads/"; exit 1
fi
if [ ! -d .venv ]; then
  echo "First run: creating a virtual environment (this takes a minute)..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
if [ ! -f .venv/.t58_deps_installed ] || [ config/requirements.txt -nt .venv/.t58_deps_installed ]; then
  echo "Installing dependencies..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r config/requirements.txt
  touch .venv/.t58_deps_installed
fi
exec python run_web.py
