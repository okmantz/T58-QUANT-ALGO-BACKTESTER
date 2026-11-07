#!/usr/bin/env bash
# T58 Quant Algo Backtester -- Chromebook (ChromeOS Linux / Crostini) installer.
# Run from inside the unzipped app folder:   bash install_chromebook.sh
# It makes the program runnable, adds a launcher-menu icon, and starts the app. Safe to re-run.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
EXE=""
for cand in "T58-Web-App" "T58-Quant-Algo-Backtester"; do
  if [ -f "$HERE/$cand" ]; then EXE="$cand"; break; fi
done
if [ -z "$EXE" ]; then
  echo "Could not find the T58 program next to this script ($HERE)."
  echo "Run this from inside the unzipped folder that holds the program and the _internal folder."; exit 1
fi
if [ ! -d "$HERE/_internal" ]; then
  echo "The _internal folder is missing next to the program. Re-extract the whole zip (do not move files out of it)."; exit 1
fi
if [ "$(uname -s)" != "Linux" ]; then
  echo "This installer is for ChromeOS's Linux environment. On Windows/macOS use the normal build."; exit 1
fi
ARCH="$(uname -m)"
echo "CPU: $ARCH"
chmod +x "$HERE/$EXE"

# Tk (only the desktop build needs it) -- install quietly if apt is usable, never fail the install over it.
if [ "$EXE" = "T58-Quant-Algo-Backtester" ] && ! python3 -c "import tkinter" >/dev/null 2>&1; then
  echo "Installing the Tk window library (asks for your Linux password only if sudo needs it)..."
  sudo apt-get update -y >/dev/null 2>&1 && sudo apt-get install -y python3-tk libtk8.6 >/dev/null 2>&1 || true
fi

# Launcher-menu entry: ChromeOS shows ~/.local/share/applications/*.desktop under "Linux apps".
APPS="$HOME/.local/share/applications"
mkdir -p "$APPS"
ICON="$(find "$HERE/_internal" -name icon-192.png 2>/dev/null | head -n 1)"
[ -n "$ICON" ] || ICON="utilities-terminal"
cat > "$APPS/t58-quant-algo-backtester.desktop" <<DESK
[Desktop Entry]
Type=Application
Name=T58 Quant Algo Backtester
Comment=Prop-firm strategy research, validation and simulation
Exec=$HERE/$EXE
Path=$HERE
Icon=$ICON
Terminal=false
Categories=Office;Finance;
DESK
echo "Added 'T58 Quant Algo Backtester' to your ChromeOS launcher (Linux apps folder)."

if [ ! -d "$HERE/data/raw" ]; then
  echo
  echo "No market data yet. Download T58-Market-Data.zip and run:  unzip T58-Market-Data.zip -d \"$HERE\""
fi
echo
echo "Starting the app. ChromeOS forwards Linux ports to Chrome, so the browser tab (web build) opens by itself;"
echo "if it does not, open  http://penguin.linux.test:5000  in Chrome."
cd "$HERE"
exec "./$EXE"
