#!/usr/bin/env bash
set -e
LABEL=com.brAIntest.braind
PLIST_SRC=/Users/leonmatthies/brAIntest/scripts/${LABEL}.plist
PLIST_DST=$HOME/Library/LaunchAgents/${LABEL}.plist

mkdir -p /Users/leonmatthies/brAIntest/logs
mkdir -p $HOME/Library/LaunchAgents

cp "$PLIST_SRC" "$PLIST_DST"
launchctl unload "$PLIST_DST" 2>/dev/null || true
launchctl load "$PLIST_DST"
echo "Loaded $LABEL. Tail logs with: tail -f /Users/leonmatthies/brAIntest/logs/braind.out.log"
