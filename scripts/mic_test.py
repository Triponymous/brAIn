"""Mikrofon-Test: 5 Sekunden aufnehmen, RMS pro Sekunde anzeigen.

Starte das Script, dann REDE oder KLATSCH in den ersten 5 Sekunden.
"""
import sounddevice as sd
import numpy as np
import time

print("=" * 50)
print("  MIKROFON-TEST — 5 Sekunden")
print("  Rede, klatsch, mach Geraeusche!")
print("=" * 50)

for countdown in [3, 2, 1]:
    print(f"  Startet in {countdown}...")
    time.sleep(1)

print("\n  >>> AUFNAHME LAEUFT <<<\n")

rec = sd.rec(int(5 * 16000), samplerate=16000, channels=1, dtype='float32')
sd.wait()

print("  >>> FERTIG <<<\n")

# Analyse in 500ms Chunks
chunk_ms = 500
chunk_size = int(16000 * chunk_ms / 1000)
max_rms = 0

for i in range(0, len(rec), chunk_size):
    chunk = rec[i:i+chunk_size]
    rms = float(np.sqrt(np.mean(chunk**2)))
    max_rms = max(max_rms, rms)
    t = i / 16000
    bar = "█" * min(50, int(rms * 5000))
    status = "LAUT!" if rms > 0.01 else "leise" if rms > 0.001 else "still"
    print(f"  {t:4.1f}s  RMS={rms:.6f}  {status:6s}  {bar}")

print(f"\n  Peak RMS: {max_rms:.6f}")
if max_rms > 0.01:
    print("  MIKROFON FUNKTIONIERT!")
elif max_rms > 0.001:
    print("  Mikrofon funktioniert, aber sehr leise.")
elif max_rms > 0.0001:
    print("  Minimales Signal — Permission evtl. eingeschraenkt.")
else:
    print("  KEIN SIGNAL — Mikrofon-Permission fehlt fuer diesen Prozess.")
