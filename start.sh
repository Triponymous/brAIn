#!/bin/bash
# ═══════════════════════════════════════════════════════════════
#  brAIn Daemon Starter
# ═══════════════════════════════════════════════════════════════
#
#  WICHTIG: Dieses Skript MUSS aus Terminal.app gestartet werden!
#  (Nicht aus Claude Code, nicht aus VS Code Terminal)
#
#  Beim ersten Start wird macOS fragen:
#    - "Terminal möchte Ihre Eingaben überwachen" → Erlauben
#    - "Terminal möchte auf das Mikrofon zugreifen" → Erlauben
#
#  Falls die Berechtigungen nicht kommen:
#    Systemeinstellungen → Datenschutz & Sicherheit →
#      → Eingabeüberwachung → Terminal.app aktivieren
#      → Mikrofon → Terminal.app aktivieren
#
# ═══════════════════════════════════════════════════════════════

cd "$(dirname "$0")"

echo "🧠 brAIn Daemon starting..."
echo ""

# Quick permission check
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
    print('⚠️  WARNUNG: Input Monitoring Permission fehlt!')
    print('   → Systemeinstellungen → Datenschutz & Sicherheit → Eingabeüberwachung')
    print('   → Terminal.app aktivieren, dann dieses Skript neu starten.')
    print('')
else:
    print('✅ Input Monitoring: OK')
" 2>/dev/null

# Check mic
python3 -c "
import sounddevice as sd, numpy as np
try:
    audio = sd.rec(4000, samplerate=16000, channels=1, dtype='float32')
    sd.wait()
    rms = float(np.sqrt(np.mean(audio**2)))
    if rms > 0.0001:
        print('✅ Mikrofon: OK (RMS={:.4f})'.format(rms))
    else:
        print('⚠️  Mikrofon: RMS=0 — entweder sehr leise oder Permission fehlt')
        print('   → Systemeinstellungen → Datenschutz & Sicherheit → Mikrofon → Terminal.app')
except Exception as e:
    print('⚠️  Mikrofon: Fehler — {}'.format(e))
" 2>/dev/null

echo ""
echo "Dashboard: http://localhost:5173"
echo "API:       http://localhost:8765"
echo "Strg+C zum Stoppen"
echo ""

.venv/bin/python -m server.braind start --port 8765
