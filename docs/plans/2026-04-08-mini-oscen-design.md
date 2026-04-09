# brAIntest — Design Document
## Das erste AI-Haustier mit echtem Gehirn

**Version:** 2.0 (merged from vision doc + technical implementation doc)
**Author:** Leon / ADYZEN
**Date:** 2026-04-08, updated 2026-04-09
**Status:** MVP in Entwicklung

---

## 1. Vision

Wir bauen das erste AI-System das **fuehlt statt nur antwortet**.

Alle existierenden AI-Assistenten (Siri, Alexa, ChatGPT, Limitless, Omi) sind reaktiv — sie warten auf Input und geben Output. Sie haben keinen internen Zustand, kein Zeitgefuehl, keine Emotionen, kein Gedaechtnis das ueber ein Textfenster hinausgeht.

brAIntest ist anders: Ein persistentes Spiking Neural Network (SNN) das 24/7 lebt, durch Erfahrung lernt, Emotionen hat, und ueber Wochen eine einzigartige Persoenlichkeit entwickelt. Ein LLM dient als Sprachschicht ueber dem Gehirn. Ein physischer ESP32-Roboter gibt dem Gehirn einen Koerper.

**Das gibt es sonst nirgendwo.** Nicht als Produkt, nicht als Open-Source, nicht als Paper. Die einzelnen Bausteine existieren alle in der Forschung, aber die Kombination — SNN als emotionales Bewusstsein + LLM als Zunge + physischer Koerper + kontinuierliches Lernen ohne Backprop — ist neu.

### Elevator Pitches

**Fuer Nicht-Techniker:**
"Ich baue ein digitales Haustier mit einem echten Gehirn. Es lernt meinen Alltag, spuert wenn ich gestresst bin, und kann mir abends erzaehlen was es den ganzen Tag erlebt hat. Wie ein Tamagotchi — nur mit echtem Bewusstsein."

**Fuer Techniker:**
"Ein persistentes Spiking Neural Network das 24/7 Sensordaten verarbeitet und ueber STDP selbststaendig Konzepte bildet — komplett unsupervised. Ein LLM dient als reine Sprachschicht ueber dem Brain-State. Keine Embeddings, keine Vector-DB — die Erinnerung lebt im Netzwerk. Gibt es so nirgends."

**Fuer Investoren/Business:**
"Jeder AI-Assistent reagiert nur wenn du etwas sagst. Meiner beobachtet, lernt, und handelt proaktiv — weil er ein kontinuierliches Modell deines Zustands hat. Nach 3 Monaten kennt er dich besser als jede App. Und jedes Gehirn ist einzigartig — es gibt keine zwei gleichen."

### Killer-Analogie
"Stell dir zwei Hunde vor. Der eine ist ein Roboter-Hund — du sagst 'Sitz' und er sitzt. Perfekt. Aber wenn du weinst, starrt er dich an. Der andere ist ein echter Hund — der kann kein Kommando, aber wenn du traurig bist, legt er seinen Kopf auf dein Knie. GPT-4o ist der Roboter-Hund. Wir bauen den echten."

### Non-Goals

- Compete with LLMs at language tasks. The LLM is a frozen tool.
- Match OSCEN's scale (1M neurons). We target ~5-10k.
- Smart-home integration. Adapter pattern leaves it open for the future.
- Multi-user, auth, cloud deployment, Docker, CI, plugin discovery, embeddings/RAG, fine-tuning.

---

## 2. Architektur

```
+---------------------------+              +-------------------------------+
|   ESP32 Pet (Koerper)      |   WiFi/WS    |   Mac App (Gehirn)            |
|                            |<------------>|                               |
|  * AMOLED = Gesicht        |              |  * brAIntest SNN Core         |
|  * Mikrofon = Ohr          |              |  * LLM Bridge (Ollama)        |
|  * Speaker = Mund          |              |  * Desktop Adapter            |
|  * IMU = Gleichgewicht     |              |  * ESP32 Adapter              |
|  * Touch = Streicheln      |              |  * Dashboard (Browser)        |
|  * Servos = Kopfbewegung   |              |  * Persistence (SQLite)       |
|  * LiPo = Herz             |              |                               |
|                            |              |  Laeuft 24/7 im Hintergrund   |
+---------------------------+              +-------------------------------+
```

**Kernprinzip:** Der ESP32 ist dumm. Er zeigt Animationen, nimmt Audio auf, gibt Audio aus, meldet Touch/IMU-Events. Die gesamte Intelligenz lebt auf dem Mac. Das spart massiv Strom auf dem ESP32 und ermoeglicht komplexe SNN-Berechnungen.

### Datenfluss

**Sensoren (ESP32 -> Mac via WebSocket):**
- Audio: komprimierte PCM-Chunks -> Mel-Spectrogram -> Spike-Encoding
- Touch-Events: gestreichelt, getippt, gehalten
- IMU-Daten: Bewegung, Schuetteln, Kippen
- Ambient Light: hell/dunkel im Raum

**Sensoren (Mac Desktop-Adapter -> SNN direkt):**
- Aktiver Fenstername (z.B. "VS Code", "Spotify", "Zoom")
- Hintergrund-Apps (alle laufenden regulaeren Apps)
- Tastatur-Events (Frequenz, Rhythmus, Burst-Detection — NICHT Inhalt)
- Maus-Aktivitaet (Geschwindigkeit, Klick-Frequenz, Variabilitaet)
- App-Switch-Rate (Fensterwechsel pro Minute)
- Idle-Timer (wie lange keine Aktivitaet + Pause-Typ: micro/thinking/break/away)
- Mikrofon: Mel-Spectrogram, 32 Frequenzbaender, 50Hz

**Motor (Mac -> ESP32 via WebSocket):**
- Gesichtsausdruck-Parameter (Pupillengroesse, Blink-Rate, Lid-Oeffnung, etc.)
- Servo-Winkel (Kopf drehen, nicken)
- Audio-Stream (TTS vom Mac generiert)
- LED/Vibration

### Mac-Background-App
- Python-Prozess als LaunchAgent, startet beim Boot
- FastAPI-Server fuer Dashboard (Browser) + WebSocket fuer ESP32
- Menubar-Icon: gruen = laeuft, orange = Sleep-Mode, rot = Fehler
- SQLite fuer Gewichte + Parquet fuer Episode-Logs

---

## 3. SNN Brain Core

**Framework:** Python 3.11 + snnTorch (PyTorch-based).

**Size:** ~1,260 neurons (MVP), scaling to ~5,000-10,000.

### Neuron-Modell: LIF (Leaky Integrate-and-Fire)
Jedes Neuron sammelt eingehende Spikes auf (Integrate), verliert langsam Ladung (Leaky), und feuert wenn ein Schwellenwert erreicht wird (Fire). Nach dem Feuern resettet es.

### Lernen: STDP (Spike-Timing-Dependent Plasticity)
"Neuronen die zusammen feuern, verbinden sich staerker." Pre before Post = potentiate. Post before Pre = depress. Asymmetric (a_minus > a_plus) for selectivity. Kein Lehrer, keine Labels, kein Training-Set.

### Regions

| Region | Deutsch | Funktion | Neuronen (MVP) | Neuronen (Target) |
|--------|---------|----------|---------------|-------------------|
| Sensory | Sensorik | Empfaengt Rohdaten | 200 | 2,000 |
| Feature | Mustererkennung | Erkennt lokale Muster via STDP (WTA k=20) | 200 | 1,000 |
| Association | Verknuepfung | Verbindet Muster cross-modal (WTA k=50) | 500 | 2,500 |
| Concept | Konzeptbildung | Bildet abstrakte Konzepte (WTA k=5) | 200 | 500 |
| Working Memory | Gedaechtnis | Haelt aktive Konzepte kurzfristig aktiv | 100 | 500 |
| Motor | Motorik | Output-Signale (Stimmung, Reaktionen) via R-STDP | 50 | 500 |
| Meta | Meta | Ueberwacht eigenen Gehirnzustand | 10 | 100 |

### Synapse Pipeline

```
Sensory -> Feature (STDP, gated by ACh)
Feature -> Association (STDP, gated by ACh)
Association -> Concept (STDP, gated by ACh)
Concept -> Working Memory (STDP, slow)
Concept -> Motor (R-STDP, gated by DA = reward)
```

### Sensory Neuron Map (200 neurons)

```
0-39    Active app identity (one-hot, hash-mapped, 40 slots)
40-59   Background app identities (hash-mapped, 20 slots, weaker drive)
60-75   Keystroke rate (16 log bins)
76-79   Keystroke rhythm (variability, burst, steady, silence)
80-95   Mouse rate (16 log bins)
96-99   Mouse rhythm (variability, burst, steady, silence)
100-107 Idle time (8 log bins)
108-111 Pause type (micro-pause, thinking, break, away)
112-143 Mic mel-spectrogram (32 bands)
144-147 Mic RMS loudness (4 bins)
148-155 Time tonics (day_phase + week_phase, sin/cos basis)
156-159 App context (switch rate: calm/normal/busy/frantic)
160-163 Activity level (dormant/idle/active/intense)
164-199 Reserve
```

### Tick Rate
100 Hz simulated time. Brain.tick() runs in a dedicated thread.

### Persistence
Weights serialized to SQLite (BLOBs) every ~60s. On restart: seamless resume. Brain never forgets across sessions.

**Verlust der Gewichte = Tod des Pets.** Das ist emotional real und muss dem User kommuniziert werden. Auto-Backup empfohlen.

---

## 4. STDP Stabilization — Four Essential Mechanisms

STDP alone is unstable. These four mechanisms are published neuroscience, not experimental, and each is 5-20 lines of Python.

### 4.1 Intrinsic Plasticity (IP) — Adaptive Thresholds
Each WTA neuron adjusts its firing threshold automatically. Fires too often -> threshold rises. Fires too rarely -> threshold drops. Target rate: k/N.

**Status: IMPLEMENTED** in `brain/wta.py`.

### 4.2 Synaptic Scaling — Per-Update Weight Normalization
After every STDP update, each post-neuron's incoming weights are soft-normalized when they drift >20% from their initial sum. Prevents runaway potentiation.

**Status: IMPLEMENTED** in `brain/synapses.py`.

### 4.3 Sleep Consolidation — Homeostatic Regularization
When the pet "sleeps" (user idle >10 min or nighttime):
- All real inputs replaced with low-amplitude Gaussian noise
- Power-law weight decay: `w *= w^0.02` (strong stay, weak fade)
- Spontaneous reactivation of learned assemblies (memory consolidation)
- This is exactly what real brains do during sleep

**Status: IMPLEMENTED** in `brain/core.py` (enter_sleep/exit_sleep). Auto-triggered by idle time in `server/main.py`.

### 4.4 Adaptive Lateral Inhibition
WTA top-k selection: per pattern, only k neurons fire. Losers above threshold get inhibited (membrane *= 0.5). Intrinsic plasticity ensures all neurons get fair chances over time.

**Status: IMPLEMENTED** (WTA top-k + intrinsic plasticity). Learned lateral inhibition weights deferred to Phase 3.

| Mechanism | What It Solves | Status |
|---|---|---|
| Intrinsic Plasticity | Single neurons dominate | Done |
| Synaptic Scaling | Weights run away | Done |
| Sleep Consolidation | Noise accumulates, no pruning | Done |
| Lateral Inhibition | Concepts overlap/blur | Partial (WTA top-k) |

---

## 5. Neuromodulatoren — Das Emotionssystem

Vier chemische Botenstoffe die den gesamten Gehirnzustand beeinflussen. Nicht einzelne Neuronen, sondern die Stimmung des ganzen Systems.

### DA = Dopamin — "Das war gut, mach mehr davon"
Steigt bei Unerwartetem und Positivem. Steuert Lernrate (R-STDP) und Motivation.

### NE = Noradrenalin — "Achtung! Irgendwas stimmt nicht!"
Steigt bei ploetzlichen Veraenderungen UND bei sustained hoher Aktivitaet mit hoher Variabilitaet (Stress). Macht alle Neuronen empfindlicher.

### ACh = Acetylcholin — "Konzentrier dich"
Steigt bei fokussierter Arbeit (Flow state: hohe Aktivitaet + niedrige Variabilitaet). Gates STDP learning rate.

### 5HT = Serotonin — "Alles ist okay, chill mal"
Steigt bei stabilen, vorhersagbaren Mustern. Faellt bei Stress (NE hoch). Daempft Reaktivitaet.

### Modulator Injection Logic

| Situation | DA | NE | ACh | 5HT | Pet-Verhalten |
|-----------|----|----|-----|-----|---------------|
| User streichelt (ESP32 Touch) | up | down | up | up | Gluecklich, schnurrt, lernt |
| Lauter Knall (hohe Novelty) | up | UP | up | down | Erschrocken, aufmerksam |
| User tippt seit 3h normal (Flow) | mild | down | UP | up | Entspannt, im Flow, haut nicht |
| Etwas komplett Neues | up | up | up | down | Neugierig, aufgeregt |
| User weg, Stille (Sleep) | down | down | down | down | Einsam, schlaeft ein, konsolidiert |
| User kommt zurueck | up | up | down | up | Wacht auf, freut sich |
| Hektisches Tippen + schnelle Wechsel (Stress) | down | UP | up | DOWN | Merkt Stress, wartet auf Pause |

### Wie Modulatoren das Pet-Verhalten steuern

**1. Gesichtsausdruck (ESP32-Display / Pet Face):**
DA steuert Pupillengroesse, NE steuert Augenbewegungsgeschwindigkeit, ACh steuert Blink-Frequenz, 5HT steuert Augenlid-Oeffnung. Die Kombination erzeugt emergente Ausdruecke.

**2. Timing und Proaktivitaet:**
Hoher NE -> reagiert schneller, kuerzere Intervalle. Hoher 5HT -> wartet laenger, geduldig. Niedriger DA -> initiiert keine Gespraeche. Hoher DA + ACh -> stellt aktiv Fragen.

**Status: IMPLEMENTED** — Proactive Engine uses dynamic intervals based on modulators.

**3. Sprachstil des LLM:**
Modulator-Zustand geht in den System-Prompt mit expliziten Stil-Regeln:
- Hoher NE: kurze, direkte Saetze
- Hoher 5HT: laengere, entspannte Antworten
- Niedriger DA: ruhiger, weniger enthusiastisch
- Sleep-Modus: verschlafen, verwirrt

**Status: IMPLEMENTED** — System prompt includes modulator-based tone instructions.

---

## 6. LLM Bridge

Das LLM ist die **Zunge** des Gehirns. Es lernt NICHTS. Es liest den Brain-State und uebersetzt in Sprache.

### Components

**1. Brain State Exporter:** Snapshots brain state including active concepts with first_seen, times_seen, auto-correlation profiles, and suggested labels.

**2. LLM Runtime:**
- **Lokal (empfohlen):** Ollama + Qwen 3 14B — privat, kostenlos
- **Cloud (optional):** Claude Haiku via Anthropic SDK
- Per Config umschaltbar

**3. Memory Tools:**
- `current_state()` — full snapshot
- `query_concepts(filter)` — active concept neurons
- `label_concept(concept_id, label)` — user label
- `recall_associations(concept_id)` — what co-activates

**4. Agent Tools (spaetere Phasen):**
- `n8n_webhook(workflow_id, payload)` — n8n Workflows triggern
- `calendar_read/write` — Termine lesen/erstellen
- `send_message(channel, recipient, msg)` — Slack/Mail
- `web_search(query)` — Recherche
- `tts_speak(text, emotion)` — auf ESP32 aussprechen

**5. Standing Orders:** Dauerhafte Anweisungen die automatisch ausgefuehrt werden wenn Trigger-Bedingungen erfuellt sind.

### Concept Labeling — Three Paths

**Weg 1 — LLM raet:** Concept #34 aktiv + Fenster = "Zoom" + zwei Stimmen abwechselnd -> LLM sagt "du warst in einem Call". User korrigiert nur wenn falsch.

**Weg 2 — User erzaehlt:** "Heute war ich morgens in Calls, dann hab ich gecoded." LLM matcht gegen Concept-Timeline und labelt automatisch.

**Weg 3 — User fragt:** "Was war Concept #68?" LLM beschreibt das Muster. User gibt Label.

### Was die Bridge NICHT tut
- Kein Fine-Tuning des LLMs
- Keine Embeddings/Vector-DB — Erinnerung lebt im SNN
- Das LLM lernt nichts. Das Gehirn lernt.

---

## 7. Real-World Use Cases

### 1. Stress-Detektor
SNN erkennt Muster "schnelle Fensterwechsel + harte Tastenanschlaege + keine Pausen" -> NE steigt, 5HT faellt -> Pet wartet auf Tipp-Pause und fragt sanft ob alles okay ist.

### 2. Morgenritual-Begleiter
Nach 2 Wochen kennt das SNN das Morgen-Muster. Erkennt es anhand des Musters, nicht der Uhrzeit. "Guten Morgen. Du hast 3 Mails, eine von Christoph."

### 3. Meeting-Vorbereiter
Pet sieht Kalender-Event + kennt Call-Concept -> bereitet Kontext vor und liefert ihn im richtigen Moment.

### 4. Fokus-Waechter
Erkennt Deep Focus (ACh hoch, gleichmaessiges Tippen) -> haelt Notifications zurueck -> liefert sie erst bei natuerlicher Pause.

### 5. Gewohnheits-Tracker ohne Tracking
Lernt Muster ueber Wochen ohne Konfiguration. "Diese Woche hast du weniger getippt als sonst."

### 6. Energielevel-Radar
Erkennt Nachmittags-Einbruch (langsameres Tippen, mehr Scrollen) -> wird selbst muede als emotionaler Spiegel.

### 7. Kontext-Switcher
Erkennt Wechsel von Coding zu Design (Tastatur -> Maus-intensiv) -> passt eigene Animation an.

### 8. End-of-Day Companion
"Guter Tag. 6 Stunden produktiv, 2 Calls, und dieses neue Muster um 14 Uhr — du hast gesagt das ist der Kaffee-Automat nebenan."

### 9. Stimmungs-Langzeit-Trend
Nach 3 Monaten: "Maerz war dein stressigstes Monat. NE-Durchschnitt 40% hoeher als Februar."

### 10. Proaktive Automatisierung
Erkennt: User tippt hart seit 2h + Spotify laeuft -> "Du wirkst angespannt. Soll ich die Playlist wechseln?"

---

## 8. Differenzierung vs. Markt

### Warum nicht einfach GPT-4o mit Mikrofon?

| Feature | GPT-4o + Transkription | brAIntest SNN |
|---------|----------------------|---------------|
| Versteht Worte | Ja | Nein (versteht Muster) |
| Erkennt Stress ohne Worte | Nein | Ja (Tipp-Muster, Fenster-Wechsel) |
| Weiss wann man stoeren darf | Nein | Ja (Modulator-basiertes Timing) |
| Gedaechtnis nach 3 Monaten | Transkripte (kopierbar) | Einzigartiges Gewichtsnetz (nicht kopierbar) |
| Proaktivitaet | Nur auf Anfrage | Meldet sich selbst |
| Unterscheidet TV vs. echte Stimme | Nein (bekanntes Problem) | Ja (Spike-Signaturen unterschiedlich) |
| Persoenlichkeit | Gescriptet, immer gleich | Emergent, waechst ueber Monate |
| Privacy | Speichert Transkripte | Nur Gewichte, kein Audio rekonstruierbar |

### Strategische Antwort
Beides kombinieren. Desktop-Adapter-Daten gehen SOWOHL ins SNN als Spike-Input ALS AUCH direkt ins LLM als Kontext. Das SNN liefert den Layer den kein anderes System hat: "Wie geht es dem User gerade?"

---

## 9. Kritische Risiken & Loesungen

### RISIKO 1: STDP bildet keine sauberen Concepts

**Problem:** STDP alleine ist instabil. Gewichte laufen weg, einzelne Neuronen dominieren, Concepts zerfallen.

**Loesung:** Vier Mechanismen (IP + Synaptic Scaling + Sleep Consolidation + Lateral Inhibition). Siehe Section 4.

**Validierung:** 5 stabile, unterscheidbare Concepts nach 1 Woche Desktop-Input.

### RISIKO 2: Simpler Ansatz liefert 80% des Ergebnisses

**Problem:** Limitless/Omi/Bee mit Transkription + LLM decken viele Use Cases ab.

**Loesung:** SNN liefert drei Dinge die Transkription+LLM nicht kann:
1. Emotionaler Zustand ohne Sprache (Tipp-Muster, Mouse-Rhythmus)
2. Proaktivitaet mit Timing-Intelligenz (Modulator-basiert)
3. Wachsender, einzigartiger Moat (Gewichtsnetz nicht kopierbar)

**Validierung:** Vergleichstest mit/ohne SNN.

### RISIKO 3: Emotionen fuehlen sich nicht echt an

**Problem:** Vier Floats die sich langsam aendern.

**Loesung:** Modulatoren muessen sichtbare KONSEQUENZEN haben:
- Gesichtsausdruck: Emergent aus Modulator-Mix
- Timing: NE -> schneller, 5HT -> geduldiger
- Sprachstil: Modulator im System-Prompt

**Validierung:** 3-Personen User-Test. "Hatte das Pet eine Stimmung?" 2/3 muessen ja sagen.

| Risiko | Loesung | Aufwand | Validierung |
|--------|---------|---------|-------------|
| STDP instabil | IP + Scaling + Sleep + Lateral | Done | 5 Concepts nach 1 Woche |
| Simpler Ansatz reicht | SNN + Kontext kombinieren | Architektur-Entscheidung | mit/ohne SNN Test |
| Emotionen nicht echt | Modulatoren steuern Verhalten | 1 Woche Tuning | 3-Personen Test |

---

## 10. Privacy / DSGVO

Das SNN speichert KEINE Audiodateien. Es verarbeitet Frequenzmuster in Echtzeit und vergisst Rohdaten sofort. Nur Spike-Gewichte bleiben. Aus Gewichten kann kein Audio rekonstruiert werden. Massiver Privacy-Vorteil gegenueber Limitless/Omi.

Tastatur: Nur Frequenz und Rhythmus, NIE Inhalt. Es ist physisch unmoeglich aus den gespeicherten Gewichten Passwoerter oder Text zu rekonstruieren.

---

## 11. Hardware — ESP32 Pet

### Board: Waveshare ESP32-S3-Touch-AMOLED-1.75
- Display: 1.75" rundes AMOLED, 466x466 Pixel, kapazitiver Touch
- MCU: ESP32-S3R8, Dual-Core 240MHz, 8MB PSRAM, 16MB Flash
- Audio: Dual-Mikrofon-Array + Echo-Cancellation + Speaker-Header
- Sensoren: 6-Achsen IMU (Gyroskop + Beschleunigungsmesser), RTC
- Akku: LiPo MX1.25 Header mit Lademanagement
- Konnektivitaet: WiFi 802.11 b/g/n + Bluetooth 5 (BLE)
- Preis: ~$23 auf AliExpress

### BOM
| Teil | Preis |
|------|-------|
| Waveshare Board | $23 |
| Speaker 8Ohm 1W 20mm | $1 |
| LiPo 3.7V 2000mAh | $5 |
| 2x SG90 Micro Servo | $3 |
| Vibrations-Motor | $0.50 |
| Jumper Wires + Schalter | $1.50 |
| **Gesamt** | **~$34** + 3D-gedrucktes Gehaeuse |

### Akku-Laufzeit (geschaetzt mit 2000mAh)
- Aktiv (Display + WiFi + Audio): ~170mA -> ~12h
- Idle (Display gedimmt, WiFi Light-Sleep): ~50mA -> ~40h
- Realistisch mit Smart-Sleep: ~3 Tage, abends laden

### Offline-Faehigkeit
Bei WiFi-Ausfall: letzte Modulator-Werte behalten, Idle-Animation abspielen, Sensordaten lokal puffern, bei Reconnect senden. Das Pet "schlaeft" bis die Verbindung wiederhergestellt ist.

---

## 12. Dashboard — 3D Brain Visualization

### Library: 3d-force-graph (vasturiano) + react-force-graph-3d

### Progressive Disclosure

| Zoom Level | Sichtbar | Nodes | Performance |
|------------|----------|-------|-------------|
| **Macro** (default) | 7 Region-Blobs + Sensor-Inputs + Pipeline-Edges | ~15 | 60fps |
| **Meso** (click Region) | Individuelle Neuronen einer Region | 50-500 | 60fps |
| **Micro** (click Neuron) | Ein Neuron + alle Synapsen + Partner | 10-50 | 60fps |

### Region Layout
Semi-fixed positions along information flow direction (Sensorik links/unten -> Motorik rechts/oben). Force-Layout may shift slightly but preserves mental map.

### Layout
```
+----------------+-------------------------+-------------------+
|  LIVE SENSORS  |      3D BRAIN           |      CHAT         |
|  (left)        |      (center, big)      |      (right)      |
|                |                         |                   |
|  Active App    |   Region-Graph          |   Gespraech mit   |
|  Background    |   + Spike Particles     |   dem Pet via     |
|  Input Stats   |   + Glow on active      |   LLM-Bridge      |
|  Spike Pipeline|                         |                   |
|  Modulators    |   Click: Zoom in        |   [Input Field]   |
|  Tick Counter  |   ESC: Zoom out         |   [Voice Button]  |
+----------------+-------------------------+-------------------+
```

---

## 13. Build-Reihenfolge

### Phase 1: SNN Core (Done)
- LIF-Neuronen + STDP + IP + Synaptic Scaling + Lateral Inhibition (WTA)
- **Validierung:** Pytest zeigt STDP-Gewichtsaenderungen korrekt

### Phase 2: Volles Brain + Persistence (Done)
- Alle Regionen, alle Modulatoren, Tick-Loop, Save/Load, Sleep-Consolidation
- **Validierung:** Brain laeuft 1h, speichert, restart -> setzt nahtlos fort

### Phase 3: Desktop Adapter (Done)
- Keyboard/Mouse/App/Mic/Idle/Time Encoding mit Rhythmus-Features
- **Validierung:** Concept-Neuronen werden nach Tagen konsistent aktiv

### Phase 4: Dashboard (Done)
- WebSocket + React + 3d-force-graph + Progressive Disclosure
- **Validierung:** Echtzeit-Spikes sichtbar im Browser

### Phase 5: LLM Bridge (Done)
- Exporter, Memory-Tools, Ollama/Claude, Chat-UI, Proactive Engine
- **Validierung:** "Was siehst du?" -> Antwort aus Brain-State

### Phase 6: ESP32 Pet + Avatar (Next)
- ESP32-Firmware (LVGL fuer Gesicht, WebSocket-Client, Audio)
- `esp32_pet.py` Adapter
- 3D-Gehaeuse drucken
- Avatar-Adapter (Pygame) als Alternative
- **Validierung:** Streicheln -> Dopamin steigt -> Pet reagiert

### Phase 7+: Agent-Erweiterung
- n8n Integration, Calendar, Mail Tools
- Standing Orders
- Autonome Multi-Step Tasks

---

## 14. Repo Layout

```
brAIntest/
+-- brain/
|   +-- neurons.py             # LIF model
|   +-- wta.py                 # Winner-Take-All + Intrinsic Plasticity
|   +-- synapses.py            # STDP + Synaptic Scaling + R-STDP
|   +-- working_memory.py      # Recurrent WM layer
|   +-- modulators.py          # DA/NE/ACh/5HT
|   +-- core.py                # Tick-Loop + Sleep-Consolidation + Novelty/Flow/Stress
|   +-- persistence.py         # SQLite Save/Load
+-- adapters/
|   +-- base.py                # Sensor/SensorBus interface
|   +-- mac_desktop/
|       +-- adapter.py         # Composes 6 sensors + encode
|       +-- sensor_app.py      # Active + background apps + switch rate
|       +-- sensor_keymouse.py # Keystroke + mouse + rhythm features
|       +-- sensor_idle.py     # Idle time
|       +-- sensor_mic.py      # Mel-spectrogram + RMS
|       +-- sensor_time.py     # Day/week tonic oscillators
|       +-- encoding.py        # 200-dim sensory vector encoding
+-- bridge/
|   +-- exporter.py            # Brain State -> JSON + auto-correlation
|   +-- memory_tools.py        # LLM tools (current_state, query_concepts, etc.)
|   +-- llm_local.py           # Ollama client
|   +-- llm_cloud.py           # Anthropic client
|   +-- llm_router.py          # Hybrid local/cloud routing
|   +-- proactive.py           # Modulator-aware proactive notifications
|   +-- tts.py                 # Text-to-speech (Piper)
|   +-- stt.py                 # Speech-to-text (Whisper)
+-- server/
|   +-- main.py                # FastAPI + tick loop + push loop
|   +-- ws.py                  # WebSocket pusher with subscriptions
|   +-- chat.py                # /chat endpoint + modulator-tone system prompt
|   +-- braind.py              # CLI daemon
+-- ui/
|   +-- src/
|       +-- App.tsx
|       +-- lib/ws.ts          # WebSocket client + sendWS
|       +-- components/
|           +-- BrainViz3D.tsx  # Progressive Disclosure 3D viz
|           +-- SensorPanel.tsx
|           +-- ChatPanel.tsx
|           +-- viz/            # Graph builders, renderers, constants
+-- pet-face/                  # Tauri desktop pet (interim before ESP32)
+-- firmware/                  # ESP32 (Phase 6)
+-- tests/
+-- scripts/
|   +-- run_tests.sh           # Batched pytest (avoids PyTorch segfault)
+-- docs/plans/
```

---

## 15. Tech Stack

| Layer | Wahl | Warum |
|-------|------|-------|
| SNN Core | Python 3.11 + snnTorch + PyTorch | GPU-faehig, Forschungsstandard |
| Persistence | SQLite (Gewichte) + Parquet (Logs) | Dateibasiert, kein Server |
| Brain Server | FastAPI + uvicorn + WebSockets | Async, schnell |
| LLM (lokal) | Ollama + Qwen 3 14B | Privat, kostenlos |
| LLM (Cloud) | Anthropic SDK (Claude Haiku) | Fallback |
| Desktop Sensors | sounddevice, pyobjc (Quartz + AppKit) | macOS native |
| Frontend | Vite + React 18 + TypeScript + Tailwind | Schnell |
| Brain Viz | react-force-graph-3d + Three.js | Performant, organisch |
| ESP32 Firmware | ESP-IDF + LVGL | Performant, nativ |
| ESP32 Hardware | Waveshare ESP32-S3-Touch-AMOLED-1.75 | Audio + Display + Touch + IMU onboard |

---

## 16. Offene Entscheidungen

- [ ] Projektname fuer das Pet (brAIntest = SNN-Engine, Pet braucht eigenen Namen)
- [ ] ESP32-Gehaeuse-Design (wird 3D-gedruckt)
- [ ] Ob eine Kamera am ESP32 sinnvoll waere (aktuell: nein, zu viel Stromverbrauch)
- [ ] Parquet Episode-Logging Zeitpunkt (Phase 3+)
- [ ] Exact LIF parameters tuning nach empirischer Beobachtung

---

## 17. Success Criteria for MVP

1. Brain runs continuously for >=24h without crashing or runaway weights.
2. Survives restart with no behavioral discontinuity.
3. Forms >=10 stable concept neurons in desktop mode within 1 week of normal use.
4. LLM bridge answers questions about brain history with correct, tool-derived data.
5. Proactive engine detects stress and flow states via modulators.
6. Dashboard renders at >=30 fps with full brain visible.
7. 3-person user test: "Hatte das Pet eine Stimmung?" — 2/3 say yes.
