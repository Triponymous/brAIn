#!/usr/bin/env bash
# Install brAIn as a LaunchAgent: the control server starts at login, survives
# reboots, brings the brain daemon up and restarts it if it ever dies.
#
#   ./scripts/install_launchd.sh              install (or reinstall) and load
#   ./scripts/install_launchd.sh --render-only  print the rendered plist, touch nothing
set -euo pipefail

PROJ_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON="$PROJ_DIR/.venv/bin/python"
LABEL=com.brain.control
TEMPLATE="$PROJ_DIR/scripts/$LABEL.plist"
DEST="$HOME/Library/LaunchAgents/$LABEL.plist"
OLD_DAEMON_PLIST="$HOME/Library/LaunchAgents/com.brain.braind.plist"

if [ ! -x "$PYTHON" ]; then
  echo "No venv at $PYTHON — run: uv venv && uv pip install -e \".[dev]\"" >&2
  exit 1
fi

render() {
  sed -e "s|__PROJ_DIR__|$PROJ_DIR|g" -e "s|__PYTHON__|$PYTHON|g" "$TEMPLATE"
}

if [ "${1:-}" = "--render-only" ]; then
  render
  exit 0
fi

mkdir -p "$PROJ_DIR/logs" "$HOME/Library/LaunchAgents"

# An older install pointed launchd at the daemon itself; that fights the Stop button.
if [ -f "$OLD_DAEMON_PLIST" ]; then
  launchctl unload "$OLD_DAEMON_PLIST" 2>/dev/null || true
  rm -f "$OLD_DAEMON_PLIST"
  echo "Removed old com.brain.braind agent."
fi

launchctl unload "$DEST" 2>/dev/null || true
render > "$DEST"
launchctl load "$DEST"

REAL_PYTHON=$("$PYTHON" -c 'import os, sys; print(os.path.realpath(sys.executable))')
cat <<MSG
Loaded $LABEL. Dashboard: http://127.0.0.1:8900   Logs: tail -f $PROJ_DIR/logs/brain.out.log

The daemon now runs under launchd, so macOS needs the permissions granted to the
Python binary itself, not to Terminal. In System Settings -> Privacy & Security add
  $REAL_PYTHON
to both  Input Monitoring  and  Microphone , then click Stop / Start in the dashboard
(or wait: the daemon checks and reports on every start in the log above).
MSG
