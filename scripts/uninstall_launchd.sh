#!/usr/bin/env bash
# Remove the brAIn LaunchAgent(s). Stops the control server; a running brain
# daemon keeps its pidfile and is stopped too, so nothing is left behind.
for LABEL in com.brain.control com.brain.braind; do
  PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
  if [ -f "$PLIST" ]; then
    launchctl unload "$PLIST" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Unloaded and removed $LABEL"
  fi
done
PIDFILE="$(cd "$(dirname "$0")/.." && pwd)/checkpoints/braind.pid"
if [ -f "$PIDFILE" ]; then
  kill -TERM "$(cat "$PIDFILE")" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "Stopped brain daemon"
fi
