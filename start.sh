#!/bin/bash
# ===============================================================
#  brAIn daemon launcher
# ===============================================================
#
#  IMPORTANT: run this from Terminal.app — not from an IDE terminal
#  (VS Code, Claude Code, ...) — or macOS won't attach the
#  Input-Monitoring / Microphone permission prompts to the right app.
#
#  Nothing is captured until you share a source in the dashboard.
#  macOS asks when a source first needs a permission:
#    - "Terminal would like to monitor your input"   -> Allow
#    - "Terminal would like to access the microphone" -> Allow
#
#  If the prompts never appear:
#    System Settings -> Privacy & Security ->
#      -> Input Monitoring -> enable Terminal.app
#      -> Microphone       -> enable Terminal.app
#
#  Prefer a UI? Run `python -m server.control` and open the dashboard
#  at http://127.0.0.1:8900: its Live page starts, stops and shares.
# ===============================================================

cd "$(dirname "$0")"

echo "brAIn daemon starting..."
# No permission probe here: it read input counters and recorded audio before
# any source was shared. The daemon checks only the sources you share.
echo ""
echo "API:       http://localhost:8000"
echo "Dashboard: http://127.0.0.1:8900  (run: python -m server.control)"
echo "Ctrl+C to stop"
echo ""

.venv/bin/python -m server.braind start
