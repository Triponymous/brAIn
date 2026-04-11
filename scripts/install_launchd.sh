#!/usr/bin/env bash
set -e
LABEL=com.brain.braind
PLIST_SRC=$(dirname "$0")/${LABEL}.plist
PLIST_DST=$HOME/Library/LaunchAgents/${LABEL}.plist

PROJ_DIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$PROJ_DIR/logs"
mkdir -p $HOME/Library/LaunchAgents

cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "Loaded $LABEL. Tail logs with: tail -f $PROJ_DIR/logs/braind.out.log"
