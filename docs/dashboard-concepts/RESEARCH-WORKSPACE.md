# Observatory — Research workspace 03

Stand: 14. September 2026. Lokal umgesetzt; kein Commit, Push oder Deployment.

**Fortführung:** Die fünf hier beschriebenen Ansichten bleiben synthetisch. Die zusätzliche, getrennte [Live session](http://127.0.0.1:4178/observatory.html#live) und ihre opt-in Telemetrie sind in [LIVE-DATA.md](LIVE-DATA.md) dokumentiert.

## Ergebnis

Der bestätigte Observatory-Entwurf ist zu einer zusammenhängenden Forschungsoberfläche ausgebaut. Das echte interaktive 3D-Gehirn bleibt erhalten. Der Schwerpunkt liegt jetzt auf lesbarer Hierarchie, überprüfbarer Herkunft, konsistenten Zeitreihen und nachvollziehbaren Annotationen.

**Vorschau:** [Observatory](http://127.0.0.1:4178/observatory.html#overview) · [Methods](http://127.0.0.1:4178/observatory.html#methods)

Harvard ist eine Qualitätsambition, keine behauptete Zugehörigkeit. Die Oberfläche erzeugt weder wissenschaftliche Validierung noch Verkaufsreife. Keine Behauptung über Interesse eines möglichen Käufers.

## Fünf Ansichten, fünf Aufgaben

| Ansicht | Zweck | Umgesetzt |
|---|---|---|
| Observatory | Kontext und Interpretation verstehen | Beispielkontext, benannter oder unbekannter Zustand, Ähnlichkeit als Dezimalwert statt vermeintlicher Konfidenz, sechs Eingangsbeispiele, vier erklärte Modulatoren. |
| Neural explorer | Mechanismus untersuchen | Erhaltenes 3D-Modell, Circuit, selektierbare Einheiten, direkte Nachbarn, Ereigniszähler, zeitgebundener Inspector, Trace und Raster. |
| Vocabulary | Bedeutung kontrollieren | Suche mit echtem Leerzustand, sicher dargestellte lange Labels/Notizen, Beispiel- und lokale Annotationen getrennt ausgewiesen. |
| Learning journal | Änderungen zurückverfolgen | Unveränderte Moment-Snapshots, Szenario, Frame, Einheit, Notiz, vorherige und neue Referenz, Öffnen des exakten Snapshots. |
| Methods & provenance | Evidenz von Darstellung unterscheiden | Quellrevision, Generator, Seeds, Zeitbasis, Architektur, Annahmen, Lizenzherkunft und ausdrücklich fehlende Live-Daten/Evaluation. |

## Gestalterische Entscheidungen

- Der freigegebene dunkle Observatory-Charakter bleibt: flache Tintenflächen, feine Trennlinien, zurückhaltendes Mint und die vorhandenen Regionsfarben.
- System-Sans für Inhalt, Monospace für IDs, Zeit und Werte; wichtige Texte sind größer. Kein externer Font- oder CDN-Aufruf.
- Gemeinsame Kontextleiste statt erfundener laufender Session. Ein neutraler/bernsteinfarbener Datenstatus behauptet keine aktive Verbindung.
- Klarere Überschriften statt Aussagen über Gedanken oder Bewusstsein. Ein unbekanntes Muster bleibt sichtbar unbekannt.
- Bestehende Korrekturen an einer Referenz werden ausdrücklich angekündigt; die vorherige Referenz bleibt erhalten.
- Touch-, Tastatur-, Fokus-, Such- und Fehlerzustände sind Teil der Oberfläche. Methods bleibt auch bei schmaler Navigation beschriftet erreichbar.
- 219 durch spätere identische Selektoren überstimmte CSS-Deklarationen wurden entfernt. Keine zusätzliche UI-Bibliothek.

Der Design-Audit beeinflusste insbesondere Typografie, Zustandsklarheit, zugängliche Controls und die Behandlung von langen/fehlenden Inhalten. UI-Kontrollmaßstab: [Web Interface Guidelines](https://raw.githubusercontent.com/vercel-labs/web-interface-guidelines/main/command.md).

## Ein Dataset für alle Instrumente

**OBS-DEMO-03 / procedural-lif-v1 / Seed 4271.**

- Vier Kontext-Fixtures × 1.000 Einheiten × 200 Samples = 800.000 geprüfte Samples.
- Zeitbasis: 100 Samples/s, Samples bei 0.00 bis 1.99 s, dargestelltes Fenster [0, 2) s.
- Kleine prozedurale Integrate-and-fire-Spuren für spikende Beispielregionen. Expansion liefert binäre Projektions-Fixtures.
- Der Generator ist **kein snnTorch-Lauf**, implementiert **keine** vollständige WTA-/STDP-/Feedback-Dynamik und ist **keine** physiologische Simulation.
- Brain und Circuit verwenden dieselbe Aktivitätsserie wie Inspector und Raster.
- Goldene Punkte zeigen Ausgaben des aktuellen oder der vier vorhergehenden Samples. Der Zähler „outputs active at cursor“ zählt nur den aktuellen Sample.
- Ereigniszahlen im Inspector zählen bis einschließlich Cursor. Die Trace zeigt das ganze Beispiel-Fenster, der Cursor seine aktuelle Position.
- Wandernde Lichter auf den Fasern illustrieren Richtung. Sie behaupten keine gemessene kausale Weiterleitung eines konkreten Spikes.
- Modulatorwerte, Kontextbeschreibungen und Ähnlichkeit sind redaktionell gesetzte Fixture-Werte. Sie bleiben im jeweiligen Kontext konstant und werden nicht als daraus berechnete Erkennung ausgegeben.

Replay startet nicht automatisch, läuft standardmäßig mit 0.1× und unterstützt 0.5× / 1×. Es stoppt bei Frame 199, bei Dialogen, beim Verlassen der Netzwerkansichten oder einem versteckten Dokument. Reduced Motion unterbindet Wiedergabe; der Slider bleibt manuell bedienbar.

## Annotationen und Reproduktion

Ein Label ist an **Szenario + Frame** gebunden, nicht pauschal an alle Zeitpunkte eines Szenarios. Die angezeigte Einheit wird als zusätzlicher Inspektionskontext festgehalten; das Label wird dadurch nicht zur Bedeutung dieses einzelnen Neurons.

Journal-Einträge enthalten eigene unveränderliche Kopien von Snapshot und Referenz. Spätere Änderungen an Kontext, Frame oder Vokabular verändern alte Einträge nicht. Beispielereignisse und echte Interaktionen mit dieser lokalen Demo sind voneinander getrennt.

**Speicherung:** nur Tab-Speicher. Bei unexportierten Annotationen wird eine Browser-Warnung vor Verlassen/Neuladen angefordert. Es gibt kein heimliches Local Storage und keine Übertragung an einen Server.

**Export:** JSON-Schema `brain.observatory.snapshot.v3`, inklusive:

- Generatorversion, Seeds, kompletter Beispielkontexte und tatsächlichem SHA-256 dieser Dataset-Beschreibung.
- Referenz-Commit des Python-Modells und ausdrücklich als Defaults markierter Architektur.
- Selektierter Ansicht, Region, Einheit, Frame, Replay-Speed und Sichtbarkeitseinstellungen.
- Kompletter 200-Sample-Serie der selektierten Einheit.
- Vokabular, lokalen Annotationen, vorherigen/nachfolgenden Referenzen und synthetischer Ausgangshistorie.
- Grenzen der Aussage und separater Anatomie-Attribution.

Der Dataset-Hash bestätigt nicht die Echtheit eines Experiments und ist keine Signatur der gesamten Exportdatei. Der Downloadstatus bedeutet „angefordert“, nicht „auf einem verifizierten Datenträger gesichert“.

**Copy view:** reproduziert die öffentliche Fixture-Auswahl in einem lokalen URL. Die URL enthält weder Labels noch Notizen. Sie ist kein gehosteter Share-Link und übernimmt keinen frei gedrehten Kamerawinkel.

## Bezug zum Repository

Geprüfte Basis: `f8ba635a5b2307934c5b0adba0ca8b0e7be952b7`.

| Aussage | Quelle / Abgrenzung |
|---|---|
| 200 / 500 / 200 / 100 Einheiten | Konstruktor-Defaults in `brain/core.py`, nicht das abweichende historische Modul-Docstring oder eine angenommene laufende Konfiguration. |
| 140.000 plastische Parameter | Expansion→Concept: 100.000; Concept→Memory und Memory→Concept: je 20.000. Feste Sensory→Expansion-Matrix zusätzlich. |
| 100 Hz | Default in `server/braind.py`. Die Demo verwendet separat dieselbe nominelle Zeitbasis. |
| Expansion besitzt keine LIF-Membran | Feste schwellenwertbasierte Projektion im Core. |
| `concept_membrane` ist kein echtes Membranpotenzial | `server/main.py` liefert dort `brain.concept_spike_accum`. Eine zukünftige echte Anbindung muss das Feld korrekt behandeln. |
| Anatomie | Z-Anatomy / BodyParts3D, separat CC BY-SA 4.0; räumliche SNN-Platzierung bleibt illustrativ. |

Die 678 räumlichen Pfade stammen aus einem Katalog von 3.386 deterministischen Beispielkanten (Seed 92712). Weder Katalog noch Positionen sind aus einem Checkpoint ausgelesen.

## Technischer Aufbau

| Datei | Zuständigkeit |
|---|---|
| `observatory.html` | Semantische Struktur, fünf Ansichten, Dialoge und gemeinsame Netzwerkfläche. |
| `observatory.css` | Observatory-UI und responsive Zustände. |
| `observatory-data.js` | Kontext-Fixtures, Architektur, Einheiten, Kanten und Ausgangshistorie. |
| `demo-telemetry.js` | Versionierter deterministischer Generator; begrenzter Cache für vier Kontexte. |
| `observatory-session.js` | Snapshot, Annotationen, URL-Grenze, Export und Verlustwarnung. |
| `observatory.js` | UI-Controller, verknüpfte Ansichten und Replay. |
| `brain-view.js` / `brain-3d/` | Vorhandener lazy geladener 3D-Renderer; Aktivität an gemeinsames Dataset angepasst. |
| `verify-observatory.mjs` | Ohne Browser oder zusätzliche Dependencies laufende Vertragstests. |

Kein Framework-Wechsel, keine neuen Runtime-Abhängigkeiten, keine Sensorabfragen, keine Backend- oder Betriebssystemänderungen.

## Prüfbelege

`rtk proxy node --test docs/dashboard-concepts/verify-observatory.mjs` — **7 Tests bestanden**:

1. Script-Syntax, eindeutige DOM-IDs und wesentliche semantische Beschriftungen.
2. Architekturabgleich mit Python-Konstruktor und eindeutiger Verbindungskatalog.
3. 800.000 endliche, begrenzte Samples; binäre Ausgaben und exakte kumulative Zähler.
4. Determinismus, unterscheidbare Kontexte und übereinstimmende Aktivitätswerte.
5. Validierte URL-Felder; ungültige Werte verworfen, keine fremden Notizparameter übernommen.
6. Eingefrorene Snapshots und unveränderte Vorher-/Nachher-Referenzen.
7. JSON-Vertrag, passende Einheit/Zeitserie, Annotationen und nachgerechneter SHA-256.

`rtk proxy node 3d/verify-brain.mjs` — **PASS**: 1.000 Einheiten, 3.386 Kanten, 678 Pfade, echte räumliche Tiefe, korrekte Kurvenendpunkte und lokale Assets.

Im Browser geprüft:

- Desktop bei 1.440 px; mobile Darstellung und Sonderzeichen/Leerzustand bei 390 px.
- Alle fünf Seiten bei 320 px ohne horizontales Seiten-Overflow; Methods-Tabelle intern horizontal scrollbar.
- Frame 043: 3D, Inspector, Trace und URL synchron; C-042 zeigt 0.919 a.u. und 1 Ereignis bis Cursor.
- E-499: binärer Output; Brain → Circuit mit 1.000 Einheiten → Brain hält Auswahl und Frame.
- 1×-Replay stoppt bei 199; genau ein Canvas, fünf Draw Calls und fünf Geometrien im geprüften Lauf.
- Annotation mit langem Label und HTML-Sonderzeichen an unbekanntem Kontext/Frame 199; bei Frame 198 wieder unbekannt.
- Journal öffnet exakt Frame 199 und E-499 samt sicher dargestellter Notiz.
- Leere Eingabe meldet einen lokalen Fehler und fokussiert das Feld; Vokabular-Leerzustand versteckt eine unpassende Detailkarte.
- JSON-Download über UI angefordert; vollständiger JSON-Inhalt/Hash zusätzlich im Vertragstest geprüft.
- Frisch geladener Direktlink stellt Rest/E-499/Expansion/Frame 117/Circuit/0.5× sowie ausgeschaltete Konturen/Verbindungen wieder her.
- Expliziter `WEBGL_OFF=1`-Fallback bezeichnet das statische Bild korrekt, deaktiviert 3D-Replay und erhält einen funktionierenden Circuit.
- Keine neuen Browser-Warnungen oder Konsolenfehler im abschließenden geprüften Lauf.

Nicht als geprüft behauptet: reale Mobil-GPU-Performance, Screenreader-End-to-End, erzwungener Graphics-Context-Loss, OS-seitig aktivierter Reduced-Motion-Modus und verlässliches Speichern einer Browser-Download-Datei. Reduced-Motion- und Verlustwarnungspfade wurden im Code geprüft; die komplette Studiogate-Abnahme des 3D-Skills bleibt offen.

## Was für tatsächliche Forschungs- und Produktreife noch fehlt

1. **Ein echter read-only Telemetrievertrag:** Session-/Checkpoint-ID, Konfiguration, monotone Sample-Zeit, Spike-Ereignisse, echte Membranwerte oder klar markiertes Fehlen, Projektionsausgabe, Versionierung und Stale-/Disconnect-Zustände.
2. **Eine überprüfbare Fragestellung:** konkrete Hypothese zum Vorteil gelernter Desktopmuster; dokumentiertes Evaluationsprotokoll und Trennung von Trainings-/Anpassungsdaten und späteren Testbeobachtungen.
3. **Vergleiche und Fehleranalyse:** geeignete einfache Baseline, Ablationen für Expansion/Feedback/Modulation, unbekannte Muster, Fehlalarme und Nutzen für die tatsächliche Arbeit. Keine Ergebnisse vorwegnehmen.
4. **Durable Experimente:** versionierte Konfiguration/Seeds, Checkpointbezug, wiederladbare Annotationen, nachvollziehbare Ergebnisse und klarer Daten-Lebenszyklus.
5. **Kontrollierte Produktaktionen:** tatsächlich benötigte Eingänge, sichtbare Berechtigungen, eng begrenzte Freigaben, Vorschau, Widerruf/Rücknahme und Aktionshistorie.

Diese Punkte sind eine nächste Forschungsstrecke, keine stillschweigend implementierten Fähigkeiten. Ein schöner Workspace macht das Projekt besser vermittelbar; belastbare Evidenz muss aus dem Projekt selbst kommen.
