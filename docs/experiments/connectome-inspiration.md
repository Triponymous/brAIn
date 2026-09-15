# Was brAIn vom Fliegengehirn übernehmen sollte

Stand: 14.09.2026. Entscheidung: **Verschaltungsprinzipien testen, kein komplettes
Fliegengehirn als Desktop-Modell übernehmen.** Keine Änderung am laufenden SNN.

## Was die neue Karte tatsächlich liefert

Der verlinkte Beitrag erschien am 03.09.2026. Das MaleCNS-Projekt umfasst das
Gehirn **und den ventralen Nervenstrang**: 166.691 Neuronen laut Autorenabstract,
rund 125 Millionen synaptische Kontakte laut Google Research. Das ist ein
struktureller Forschungsdatensatz, kein fertig trainiertes universelles KI-Modell.
Die Größenangabe bezeichnet nicht nur das Gehirn; einzelne Kontakte sind auch
nicht dasselbe wie die Anzahl verschiedener verbundener Neuronenpaare.
[Autorenabstract auf der Projektübersicht](https://sites.research.google/gr/neural-mapping/),
[Google Research](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/).

Eine Verbindungskarte legt nicht sämtliche neuronalen Zeitkonstanten,
Synapsenwirksamkeiten, Aktivitätszustände oder Lernregeln fest. Ein passendes
Beispiel ist die connectomgestützte Modellierung des visuellen Fliegensystems:
Die Forscher ergänzten die gemessene Struktur durch auf eine konkrete Aufgabe
optimierte Parameter. Struktur und Aufgabe wirkten zusammen; das Netz verstand
nicht durch das Einlesen der Karte beliebige neue Eingaben.
[Lappalainen et al., Nature 2024](https://www.nature.com/articles/s41586-024-07939-3).

Meine technische Schlussfolgerung: Ein auf Fliegensinne und Bewegungsverhalten
zugeschnittenes Gesamtconnectom besitzt keine automatisch passende Zuordnung
zu Tastaturaktivität, Meetings oder Unterbrechbarkeit. Diese Zuordnung und ihr
Nutzen wären zusätzliche Forschungsprobleme. Mehr Neuronen wären dabei kein
Erfolgsnachweis für brAIn.

## Der stärkste Ansatzpunkt ist bereits vorhanden

Der aktuelle [Core](../../brain/core.py) verwendet standardmäßig 200 sensorische,
500 Expansions-, 200 Konzept- und 100 Arbeitsgedächtniseinheiten. Die Expansion
ist eine feste Zufallsprojektion mit ungefähr 10 % Verbindungen; Konzeptneuronen
konkurrieren über [WTA](../../brain/wta.py), und
[Arbeitsgedächtnis](../../brain/working_memory.py) liefert Rückkopplung.
Das ist keine biologische Rekonstruktion, bietet aber einen passenden Ansatzpunkt.

Im Mushroom Body der Fliege helfen Kenyon-Zellen und hemmende Rückkopplung über
APL, ähnliche Geruchsmuster unterschiedlich zu repräsentieren. Eingriffe in
diesen Kreislauf verschlechterten in einer experimentellen Studie die gelernte
Unterscheidung ähnlicher Gerüche. Das ist spezifische biologische Evidenz –
**noch kein Nachweis für Desktop- oder Audioklassifikation**.
[Lin et al., Nature Neuroscience 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4000970/).

| Prinzip | Bezug zu brAIn | Konkrete, noch unbewiesene Übertragung |
| --- | --- | --- |
| Sparse Aktivität durch hemmende Rückkopplung | Expansion vorhanden; ihre Ausgabe besitzt noch kein aktivitätsgeregeltes Sparsitätsziel | Erproben, ob eine sparsamere Expansion ähnliche Eingaben besser trennt |
| Lernsignale in spezifischen Schaltkreisen | Expansion→Konzept erhält aktuell eine gemeinsame DA/ACh-Modulation | Später lokale, durch tatsächliches Feedback begründete Lernsignale testen |
| Wiederkehrende Verbindungsmuster und getrennte Teilnetze | Sensorik, Konzepte und Gedächtnis sind schon getrennt | Kleine ausgewählte Motive gegen passende Zufallsnetze vergleichen |
| Zelltypen und nachvollziehbare Verbindungen | Observatory zeigt Regionen und Neuronen | Tatsächliche Modellverbindungen nach Funktion, Richtung und Veränderung untersuchbar machen |

Die zweite Idee stützt sich auf getrennte modulierte Lernkreise und deren
Rückkopplung im **larvalen** Mushroom Body; sie ist nicht als neue Entdeckung
des männlichen Adult-Connectoms gemeint.
[Eschbach et al., Nature Neuroscience 2020](https://www.nature.com/articles/s41593-020-0607-9).
Netzwerkmotive und regionale Organisation wurden auch im **weiblichen**
Adult-Connectom untersucht; die Übertragung auf brAIn bleibt unsere Hypothese.
[Lin et al., Nature 2024](https://www.nature.com/articles/s41586-024-07968-y).

DA, NE, ACh und 5HT in brAIn sind Modellvariablen. Eine Fliegenarchitektur würde
daraus keine Messung menschlicher Neurotransmitter, Gefühle oder Aufmerksamkeit
machen. Dafür liefern die hier zitierten Arbeiten keinen Nachweis.

## Kleiner Code-Check: wenige Verbindungen bedeuten nicht wenig Aktivität

Ein isolierter Diagnoseaufruf mit frischem `Brain()`, Seed 4271, aktivierte nur
die ersten 5, 10 beziehungsweise 20 sensorischen Eingänge der Projektionsmatrix.
Kein Mikrofon, keine Desktop-Daten, kein Lernen und kein laufendes Modell wurden
verwendet. Die folgenden Werte sind **synthetische Strukturdiagnostik**:

| Gleichzeitig aktive sensorische Einheiten | Aktive Expansion bei Schwelle 1 | Bei Schwelle 1,25 |
| --- | --- | --- |
| 5 von 200 | 39,8 % | 7,2 % |
| 10 von 200 | 65,4 % | 24,6 % |
| 20 von 200 | 88,8 % | 57,8 % |

Gemessene Verbindungsdichte: 9,997 %. Die beiden Schwellen entsprechen der
Projektionsregel ohne beziehungsweise mit 5HT-Modellwert 0,05. Das sind ausgewählte
Zustände, keine Messung ihrer Häufigkeit im Desktopbetrieb. Der Check zeigt:
Eine dünn besetzte Matrix kann viele Ausgabeeinheiten gleichzeitig aktivieren.
Das rechtfertigt eine Hypothese zur Aktivitätsregelung, noch keinen Umbau.
Codebezug: `Brain.__init__`, `_expansion_weights`, Schritt 4 in `Brain.tick`.

## Mein nächstes Experiment – nicht bereits implementiert

1. Die vorhandene Architektur und ihren neuronalen Umfang beibehalten.
2. Die aktuelle Expansion mit einer APL-inspirierten hemmenden Rückkopplung
   vergleichen. Eine einfache Top-k-Variante mit vergleichbarer Aktivitätsrate
   als Kontrolle hinzufügen: So prüfen wir, ob das Motiv mehr bringt als nur
   weniger Aktivität. Die Variante ist eine technische Abstraktion, keine
   anatomisch getreue APL-Simulation.
3. Gleiche Eingaben, Initialisierungsseeds, Lernexposition und Tuning-Budgets;
   dieselben unveränderten STDP-, WTA- und Gedächtniskomponenten. Trainings-/Test-
   Trennung nach unabhängigen Aufnahmen beziehungsweise zeitlichen Episoden.
4. Musterüberlappung und Aktivitätsdichte als Diagnose messen; entscheidend sind
   zusätzlich Klassifikationsgüte auf unbekannten Episoden, Lernstabilität und
   Laufzeit. Auch ein stilles Netz hat wenig Überlappung, aber keinen Nutzen.
5. Erst nach erfolgreicher Replikation in den Live-Pfad übernehmen. VAD und
   neue Architektur zunächst getrennt verändern, um ihren Beitrag unterscheiden
   zu können. Kein Connectom-Download und keine neue Sensorfreigabe nötig.

Wenn wir später echte Verbindungsmotive importieren, benötigen wir insbesondere
Kontrollnetze mit vergleichbaren Ein-/Ausgangsgraden, Vorzeichen,
Gewichtsverteilungen und Initialisierung. Ein aktueller **Preprint, kein hier
verifizierter begutachteter Befund**, zeigt in einem konkreten Flyvis-Vergleich,
wie vermeintliche Topologievorteile bei strengeren Kontrollen verschwinden
können. Das widerlegt nicht jede biologische Inspiration, begründet aber faire
Kontrollen statt eines pauschalen „biologisch ist besser“.
[Dhiman, arXiv 2026](https://arxiv.org/abs/2604.04033).

## Konsequenz fürs Dashboard und die Audio-Arbeit

Fürs Observatory wären ein- und ausgehende Modellverbindungen, hemmende versus
erregende Beiträge, zeitliche Aktivität und Änderungen nach Feedback wertvoller
als eine zusätzliche biologische Gehirnform. Eine anatomische Referenzansicht
sollte klar von unserem Modell und seinen Messdaten getrennt sein; ein
sichtbarer Pfad belegt allein noch keine kausale Erklärung einer Entscheidung.
Das ist ein Gestaltungsvorschlag, keine bereits umgesetzte Funktion.

Wichtig für den späteren SNN-Audiotest: `Brain.tick` maskiert aktuell die
Mel-Eingänge im separaten `ConceptTracker` (`tracker_spikes[112:145] = 0`).
Die SNN-Konzeptschicht und diese zusätzliche Cluster-Auswertung sind daher nicht
dieselbe Messung. Ein neues Audiosignal muss am tatsächlich verwendeten
Auswertungspfad geprüft werden; VAD allein löst diese Zuordnungsfrage nicht.

Der [Offline-Audiovergleich](audio-vad.md) ist separat vorbereitet und inklusive
echtem Silero-ONNX-Modell auf künstlichen Audiosignalen getestet: **49 Tests
bestanden** zusammen mit Capture-/Telemetrie-Regressionen. Für eine Aussage über
Meeting-/Musikerkennung fehlt weiterhin ein rechtmäßig nutzbarer, beschrifteter
Korpus. Es wurde niemand kontaktiert und kein Mikrofon aktiviert.

Quellenumfang: Autoren-/Projektseiten zur neuen MaleCNS-Veröffentlichung und die
oben verlinkten Originalarbeiten. Der Cell-Volltext der neuen Veröffentlichung
war über den Recherchezugang nicht abrufbar; hier wird keine vollständige
Begutachtung seiner Methoden behauptet. Es wurden keine MaleCNS-Daten importiert.
