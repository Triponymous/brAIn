# brAIn — Living Desktop Manager

Drei eigenständige Dashboard-Konzepte · 14. September 2026

**Gewählte Richtung: Observatory, mit der Verständlichkeit von Symbiosis.** Der ausgearbeitete, eigenständige Entwurf liegt in [observatory.html](observatory.html); Konzept, Bedienabläufe und technische Grenzen stehen in [OBSERVATORY.md](OBSERVATORY.md). Die drei ursprünglichen Richtungen bleiben unten als Vergleich erhalten.

**Empfehlung für das Forschungsprojekt: Observatory.** Es macht die besondere Leistung von brAIn sofort sichtbar: ein laufendes, lernendes Netzwerk, dessen Verhalten man untersuchen und durch eigene Begriffe mitgestalten kann. **Symbiosis** ist die stärkste Alternative, wenn der tägliche Nutzen als Living Desktop Manager den Schwerpunkt bekommt. **Fieldnotes** priorisiert wissenschaftliche Nachvollziehbarkeit.

## Die Entwürfe öffnen

Die Datei [index.html](index.html) ist vollständig eigenständig: kein Build, keine Installation, keine externen Fonts, keine Bibliotheken und keine Verbindung zum Brain-Daemon. Direkt im Browser öffnen oder über den lokalen Preview-Server aufrufen:

- [01 — Observatory](http://127.0.0.1:4178/index.html#observatory)
- [02 — Fieldnotes](http://127.0.0.1:4178/index.html#fieldnotes)
- [03 — Symbiosis](http://127.0.0.1:4178/index.html#symbiosis)

Die obere Leiste wechselt zwischen den drei Richtungen. „Konzept & Idee“ erklärt jeweils Design, Nutzen und Abwägungen auf Deutsch. Die Produktoberflächen sind für das internationale Open-Source-Projekt auf Englisch.

**Umfang:** Designkonzepte mit lokalen Beispielinteraktionen. Alle Aktivitäten, Zustände, Graphpositionen, ausgewählten Verbindungen, Zeitreihen und Einzelneuronwerte sind synthetisch. Die reale Anwendung, Sensoren, Checkpoints und Berechtigungen wurden nicht verändert.

## Gemeinsame Produktidee

Das Dashboard beantwortet fünf konkrete Fragen:

1. **Was nimmt brAIn wahr?** Sensorart, Einheit, Zeitpunkt und sichtbarer Verbindungsstatus.
2. **Was passiert im Netzwerk?** Region, Aktivität, Kontext und bei Bedarf das einzelne Neuron.
3. **Was wurde gelernt?** Wiederkehrende Muster, Nutzerbegriffe und Veränderungen über Zeit.
4. **Wie belastbar ist die Interpretation?** Beobachtung, Modellähnlichkeit und menschliches Label getrennt zeigen.
5. **Was kann ich damit tun?** Erkunden, einen Begriff bestätigen oder korrigieren, später eine erklärte Desktop-Empfehlung annehmen.

„Lebendig“ entsteht durch nachvollziehbare Reaktionen und fortlaufendes Lernen. Das visuelle Niveau einer guten Forschungsarbeit entsteht durch sorgfältige Typografie, Datenherkunft, lesbare Abbildungen und zurückhaltende Gestaltung. Es wird keine Verbindung zu Harvard oder einer anderen Institution behauptet.

## 01 — Observatory

**Leitidee:** Ein präzises Beobachtungsinstrument für ein lebendes System.

| Dimension | Entscheidung |
| --- | --- |
| Hauptfrage | Was passiert gerade im Netzwerk — und warum ist es interessant? |
| Hauptfläche | Große Neuronenlandschaft mit vier Regionen und angrenzendem Inspector |
| Einstieg | Aktiver Zustand, Default-Architektur, gelerntes Vokabular |
| Vertiefung | Neuron auswählen → Region verstehen → Beispielaktivität und Membran/Projektion untersuchen |
| Menschlicher Loop | Zustandslabel bestätigen oder einen eigenen Begriff beibringen |
| Besonders geeignet | Technische Demos, Forschung, Contributor-Onboarding, snnTorch-Community |
| Bewusste Abwägung | Höhere Informationsdichte; gute Erklärungen sind ein Pflichtbestandteil |

**Design-DNA:** Tintenfarbener Hintergrund `#101416`, Datenflächen `#151b1e`, helles Mint `#b9e4c9`; mineralisches Blau, Lavendel und Ocker markieren die Regionen. Neutrale Grotesk für das Interface, Monospace für Messwerte. Dünne Konturen, 14-px-Panels, ruhige Hierarchie. Keine 3D-Kamera als Voraussetzung zum Verständnis.

**Erlebnis:** Alle 1.000 Beispielneuronen sind sichtbar. Auswahl hebt Verbindungen hervor; der Inspector erklärt die Rolle der Region. Zoom, pausierbare synthetische Aktivität und ein Frame-Regler machen das Netzwerk erforschbar. Ein Dropdown plus Zahlenfeld bietet einen Tastaturzugang.

**Im späteren Produkt:** Standardmäßig nur relevante beziehungsweise gefilterte Kanten anzeigen; 140.000 plastische Verbindungen nicht als dauerhaftes Linienknäuel rendern. Region → Neuron → Verbindung stufenweise erschließen. Layout stabil halten, nur Aktivität ändern. Jede hervorgehobene Kante benötigt eine tatsächliche Quelle.

## 02 — Fieldnotes

**Leitidee:** Ein digitales Laborjournal, das Beobachtung und Erklärung zusammenbringt.

| Dimension | Entscheidung |
| --- | --- |
| Hauptfrage | Was wurde beobachtet, und welche Schlussfolgerung ist dadurch gedeckt? |
| Hauptfläche | Nummerierte Netzwerkabbildung, Spike-Raster und Prediction-error-Plot |
| Einstieg | Forschungsfrage, Modellparameter, Session und Datenherkunft |
| Vertiefung | Abbildung auswählen → Messung prüfen → Sessions vergleichen |
| Menschlicher Loop | Nutzerkorrektur als dokumentierter Teil des Experiments |
| Besonders geeignet | Forschungskommunikation, wiederholbare Experimente, spätere Paper-Abbildungen |
| Bewusste Abwägung | Weniger Begleitergefühl; belastbare Vergleiche erfordern zusätzliche Messinfrastruktur |

**Design-DNA:** Warmes Papier `#f6f4ef`, dunkle Druckfarbe `#272d29`, Karmin `#8f343e`. Serifenschrift für Überschriften, Grotesk für Erläuterungen, Monospace für technische Angaben. Horizontale Regeln statt Kartenstapel; großzügige Abstände, nummerierte Abbildungen, klar beschriftete Achsen.

**Erlebnis:** Die Architektur ist eine erklärte Abbildung. Das Raster zeigt einzelne synthetische Spike-Ereignisse; „Compare“ ergänzt eine zweite synthetische Fehlerkurve. Rechts steht eine Beobachtung mit Ursprung, Label und Modellähnlichkeit. „Methods“ erläutert, welche Angaben echte Experimente ausweisen müssten.

**Im späteren Produkt:** Session-ID, Commit, Seed, Konfiguration, Sensor-Modus, Messfenster und Metrikdefinition exportieren. Vergleichswerte benötigen identische Bedingungen oder sichtbar benannte Unterschiede. Unsicherheit aus tatsächlichen Wiederholungen ableiten. Die fallenden Beispielkurven im Entwurf belegen keinerlei Lernerfolg.

## 03 — Symbiosis

**Leitidee:** Ein verständlicher, ruhiger Begleiter, der im Alltag nützlich wird.

| Dimension | Entscheidung |
| --- | --- |
| Hauptfrage | Was erkennt brAIn gerade an meinem Arbeitsrhythmus — und passt das? |
| Hauptfläche | Ein benannter Zustand, wenige erklärende Signale und unmittelbares Feedback |
| Einstieg | „A little more in sync.“ — persönliche Kontinuität und ein verständlicher Moment |
| Vertiefung | Wort bestätigen/korrigieren → eigenes Vokabular → optional ins Netzwerk wechseln |
| Menschlicher Loop | Die Wortwahl des Nutzers bildet den Kern der Oberfläche |
| Besonders geeignet | Tägliche Nutzung und die Weiterentwicklung zum Living Desktop Manager |
| Bewusste Abwägung | Die warme Darstellung braucht präzise Sprache über die Grenzen der Interpretation |

**Design-DNA:** Waldgrüne Präsenzfläche `#19352f`, warmes Elfenbein `#eeeee5`, Salbei `#e0e6cf`, Tinte `#20362f`. Große, ruhige Serifentitel und weiche, sparsam eingesetzte Panels. Eine organische Darstellung der 1.000 Beispielneuronen dient als wiedererkennbare Präsenz; sie ist keine anatomische Rekonstruktion.

**Erlebnis:** Das Modell schlägt das gelernte Wort „Flow“ vor und nennt die beobachtbaren Muster. Eine Korrektur verändert das Beispielvokabular in allen drei Varianten. Der Tagesstreifen lässt exemplarische Momente auswählen. „The science“ führt zum Neuronen-Inspector.

**Im späteren Produkt:** Von einer erkannten Signatur zu einer konkreten, erklärten Empfehlung gelangen: etwa einen ruhigeren Workspace anbieten. Dafür braucht es echte Integrationen, überprüfbare Auslöser und explizite Nutzerfreigaben. Tagesrückblick und Replay benötigen ein bewusst gewähltes Aufzeichnungs- und Aufbewahrungskonzept.

## Anschluss an den vorhandenen Code

Grundlage ist der lokale Checkout `f8ba635`, nicht ein gestarteter Brain-Prozess. Die öffentliche Projektgeschichte kann andere Ausbaustufen beschreiben.

| Bereich | Im Checkout belegt | Konsequenz für das Dashboard |
| --- | --- | --- |
| Default-Architektur | `brain/core.py`, `Brain.__init__`: 200 Sensory, 500 Expansion, 200 Concept, 100 Working Memory | Die Konzeptzählung von 1.000 Einheiten beruht auf den Konstruktor-Defaults. |
| Daemon-Erzeugung | `server/braind.py` erzeugt `Brain()` | `config.json` enthält abweichende Zahlen; ein echtes Dashboard muss die tatsächlich laufende Instanz beschreiben. |
| Plastische Verbindungen | Expansion→Concept 500×200; Concept→WM 200×100; WM→Concept 100×200 | 140.000 plastische Gewichte. Die feste Sensory→Expansion-Projektion ist zusätzlich vorhanden. |
| Basis-Telemetrie | `server/main.py`, `push_loop`: Tick, Sleep-Modus, Modulatoren, Sensoren, Konzeptcluster, regionale Spike-Anzahlen | Gute Grundlage für Übersicht, Zustands-Feedback und Sensoranzeigen. |
| Konzeptaktivität | `concept_membrane` enthält `brain.concept_spike_accum` | Der Name darf im UI nicht als echte Konzept-Membranspannung interpretiert werden. |
| Membran / Expansion | `wm_membrane` wird geliefert; Expansion ist eine feste Schwellenprojektion, kein LIF-Layer mit eigener Membran | Einzelneuron-Inspector braucht nach Region unterschiedliche Metriken. |
| Detail-Abonnements | `server/ws.py`: macro, meso, micro; `server/main.py` ergänzt `region_spikes` für verfügbare Regionen | Das ist noch keine vollständige Topologie-/Synapsen-API. Expansion und echte individuelle Gewichte fehlen in diesem Datenpfad. |
| Gelerntes Vokabular | `bridge/felt_state.py`, `server/feel.py`, `train-ui/index.html` | Bestehender Ansatzpunkt für „That fits“ und „Teach a word“. |
| Konzeptprofile | `bridge/exporter.py`: Sensor-Kookkurrenzen und vorgeschlagene Labels | Eine Assoziation ist keine kausale Erklärung und kein Beweis für einen psychologischen Zustand. |
| Episoden / Benchmarks | `bridge/episode_log.py`, `benchmark/` | Ansatzpunkte für weitere Forschung; kein Beleg, dass neuronengenauer Session-Replay bereits vorhanden ist. |

Für einen umgesetzten Neuronen-Explorer müssen reale Spike-Ereignisse und ihre Zeitbasis klar definiert sein. Eine niedrige Membran allein ist kein belastbarer Beleg für einen Spike. Eine Rate benötigt ein Messfenster; Modellzustand, Visualisierungsakkumulator und Ereignis dürfen nicht vermischt werden.

Fachlicher Bezug für LIF und Spike-Darstellungen: [offizielles snnTorch-LIF-Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html) und [Spike-Encoding-Tutorial](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html). Diese Quellen erklären die Darstellungen; sie validieren nicht die Ergebnisse von brAIn.

## Umsetzung der Konzeptdatei

- Drei unterschiedliche Layouts und visuelle Identitäten in einer portablen Datei.
- 1.000 deterministisch erzeugte Beispielneuronen pro dargestelltem Graphen, synthetische Kanten entlang der Regionstopologie.
- Keine Frameworks, externen Assets, Netzwerkaufrufe, Mikrofonzugriffe oder Betriebssystemaktionen.
- Reduzierte Bewegung wird respektiert; Wiedergabe ist pausierbar und stoppt beim Wechsel der Variante beziehungsweise beim Verlassen des Tabs.
- Dialoge, Label-Eingabe, Tastaturauswahl eines Neurons, Timeline-Auswahl und synthetischer JSON-Export.
- Der Export ist ausdrücklich kein Brain-Checkpoint. Änderungen im Beispielvokabular verschwinden beim Neuladen.

Die Konzepte entstanden mit der Struktur aus `design-dna`: messbare Gestaltungstokens, klar unterschiedliche visuelle Charaktere und jeweils eine passende Darstellungsform für das Netzwerk. `motion-principles` führte zu kontrollierbarer Bewegung und einem statischen Einstieg.

## Prüfung

Im Browser geprüft:

- Alle drei Desktop-Kompositionen visuell kontrolliert; Layouts bei 320, 390, 768 und 1.440 px geprüft, ohne horizontalen Dokumentüberlauf.
- Pro Variante 1.000 dargestellte Beispielneuronen verifiziert.
- Auswahl eines Neurons per Graph-Klick und per Tastaturformular; unterschiedliche Metrik für Expansion gegenüber LIF-Regionen.
- Zoom, Reset, Start und Pause der synthetischen Aktivität.
- Eigenen Begriff eingeben: Beispielzustand und Vokabular aktualisieren sich variantenübergreifend; der Dialog schließt.
- Session-Vergleich blendet eine zweite Kurve ein; Auswahl eines Tagesabschnitts aktualisiert die Erklärung.
- Konzeptdialog und Rückkehr zum Entwurf.
- Keine Browser-Warnungen oder JavaScript-Fehler in der abschließenden Kontrolle; eingebettetes JavaScript syntaktisch gültig.

Die repräsentativen sekundären Textfarben erreichen mindestens 4,5:1 Kontrast: Observatory 7,07:1, Fieldnotes 4,61:1, Symbiosis 4,77:1. Die Prüfung ist kein vollständiges Accessibility-Audit. Export und Reduced Motion sind implementiert; Download beziehungsweise Betriebssystem-Präferenz wurden nicht eigens umgestellt oder durch einen automatisierten End-to-End-Test geprüft. Der Kern des Forschungsprojekts wurde für diese Designaufgabe nicht gestartet oder verändert.
