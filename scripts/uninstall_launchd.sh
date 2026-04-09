#!/usr/bin/env bash
LABEL=com.brAIntest.braind
PLIST_DST=$HOME/Library/LaunchAgents/${LABEL}.plist
launchctl unload "$PLIST_DST" 2>/dev/null || true
rm -f "$PLIST_DST"
echo "Unloaded and removed $LABEL"
