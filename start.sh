#!/bin/bash
# ===============================================================
#  brAIn daemon launcher
# ===============================================================
#
#  IMPORTANT: run this from Terminal.app — not from an IDE terminal
#  (VS Code, Claude Code, ...) — or macOS won't attach the
#  Input-Monitoring / Microphone permission prompts to the right app.
#
#  On first launch macOS will ask:
#    - "Terminal would like to monitor your input"   -> Allow
#    - "Terminal would like to access the microphone" -> Allow
#
#  If the prompts never appear:
#    System Settings -> Privacy & Security ->
#      -> Input Monitoring -> enable Terminal.app
#      -> Microphone       -> enable Terminal.app
#
#  Prefer a UI? Run `python -m server.control` and start/stop the
#  daemon from the training console at http://127.0.0.1:8900.
# ===============================================================

cd "$(dirname "$0")"

echo "brAIn daemon starting..."
echo ""

# Quick permission pre-check (the daemon re-checks on startup too)
python3 -c "
import Quartz, time
idle = Quartz.CGEventSourceSecondsSinceLastEventType(
    Quartz.kCGEventSourceStateHIDSystemState, int(0xFFFFFFFF))
c1 = Quartz.CGEventSourceCounterForEventType(
    Quartz.kCGEventSourceStateHIDSystemState, 10)
time.sleep(1.5)
c2 = Quartz.CGEventSourceCounterForEventType(
    Quartz.kCGEventSourceStateHIDSystemState, 10)
idle2 = Quartz.CGEventSourceSecondsSinceLastEventType(
    Quartz.kCGEventSourceStateHIDSystemState, int(0xFFFFFFFF))
if idle2 > idle + 0.5 and c1 == c2:
    print('[WARN] Input Monitoring permission missing!')
    print('   -> System Settings -> Privacy & Security -> Input Monitoring')
    print('   -> enable Terminal.app, then restart this script.')
    print('')
else:
    print('[OK] Input Monitoring: OK')
" 2>/dev/null

# Microphone check
python3 -c "
import sounddevice as sd, numpy as np
try:
    audio = sd.rec(4000, samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 0.0001:
        print('[OK] Microphone: OK (RMS={:.4f})'.format(rms))
    else:
        print('[WARN] Microphone: RMS=0 - either very quiet or permission missing')
        print('   -> System Settings -> Privacy & Security -> Microphone -> Terminal.app')
except Exception as e:
    print('[WARN] Microphone: error - {}'.format(e))
" 2>/dev/null

echo ""
echo "API:     http://localhost:8000"
echo "Console: http://127.0.0.1:8900  (run: python -m server.control)"
echo "Ctrl+C to stop"
echo ""

.venv/bin/python -m server.braind start
