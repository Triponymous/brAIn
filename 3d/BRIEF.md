# Design-Brief — brAIn / Observatory: echtes 3D-Fasernetz

Phase 0 · 14. September 2026 · **Vorschlag zur Freigabe, noch keine Umsetzung**

Ziel: Im bestehenden Observatory-Dashboard soll ein echtes, räumlich erkundbares SNN erscheinen. Die aktuelle Bildfläche mit darübergelegten SVG-Punkten erfüllt den neuen Wunsch nach echtem 3D nicht. Eine Umsetzung muss deshalb echte räumliche Geometrie, Perspektive, Tiefenstaffelung und Auswahl im Raum haben.

## §tier

**Vorgeschlagen: 2 — eine begrenzte 3D-Szene im vorhandenen Dashboard.** Nur die Netzwerkansicht wird räumlich; Navigation, Zustandsanzeige, Erklärungen, Vokabular und Journal bleiben erhalten.

Kein Neubau einer Engine-Website. Die Tier-Einstufung beschreibt den technischen Umfang, keine zugesagte Lieferfrist. Modellbeschaffung, gewünschte Dichte und die tatsächliche Zielhardware bestimmen den Aufwand. Die Freigabe von Tier und Signature-Liste ist noch offen.

## §signature

1. **Das räumliche Fasergehirn.** Eine erkennbare Gehirnform aus geschwungenen, unterschiedlich tief liegenden Fasern; eine zurückhaltende blauweiße Kontur gibt Orientierung, ohne das Innere zu verdecken. Ausgangsansicht seitlich-schräg wie im zuletzt beigefügten Bild, mit erkennbarer Nähe und Ferne.
2. **Direktes Erkunden.** Ziehen dreht das gesamte Gehirn wirklich im Raum; Mausrad bzw. Pinch verändert den Abstand. Front-, Seiten- und Draufsicht sowie Reset machen die Orientierung wiederherstellbar. Keine bloße Bildneigung.
3. **Ein Neuron verstehen.** Hover identifiziert eine Einheit, Klick fixiert die Auswahl und zeigt ihre Daten im bestehenden Inspector. Ein- und ausgehende Beispielverbindungen werden hervorgehoben; andere Fasern treten zurück.
4. **Den Signalweg verfolgen.** Beim Abspielen wandern klar erkennbare Impulse entlang ausgewählter Verbindungen. Feine blauweiße Fasern tragen die ruhige Grundstruktur, wenige goldene Impulse markieren Aktivität. Pause und Einzel-Frame-Steuerung halten den Zustand an.
5. **Vom Ganzen zum Detail.** Einzelne Modellregionen isolieren, die Hülle ausblenden und zwischen räumlichem Fasernetz und technischer Schaltung wechseln. Auswahl und Zeitposition bleiben erhalten.

## §non-goals

- Keine Live-Anbindung an den Brain-Daemon, keine neuen Sensoren und keine OS-Aktionen in dieser Designausarbeitung.
- Kein medizinisch validiertes Gehirn, kein Patientenscan, keine Behauptung biologischer Lokalisation der Modellregionen.
- Keine Veröffentlichung oder Verwendung der beigefügten Stockbilder als Runtime-Texturen; insbesondere keine Entfernung von Wasserzeichen.
- Keine automatische Kamerafahrt beim Scrollen, kein Scroll-Locking und kein kompletter Dashboard-Neubau.
- Kein Versprechen, gleichzeitig alle 140.000 plastischen Gewichte als einzelne sichtbare Fasern darzustellen.
- Kein Trainieren oder Verändern des echten SNN durch dekorative Klick-Impulse.

## Referenzen — was jeweils übernommen wird

| Referenz | Visuelle Beobachtung | Rolle im Zielbild |
|---|---|---|
| Bild 1 · 480 × 720 | Türkisfarbene Gehirnkontur, durchsichtige Struktur, wenige warme Lichtpunkte, geschwungene Bahnen. | Kontur, Farbdisziplin und Impulse; nicht die spiegelgleiche Rasterkomposition als Kameraersatz. |
| Bild 2 · 1.000 × 519 | Räumlich gestaffelte Knoten und dünne Verbindungen, erkennbare Perspektive und Tiefenunterschiede. | Greifbare Knoten und räumliche Erkundbarkeit; die flächendeckende Linienmenge wird im Dashboard reduziert. |
| Bild 3 · 612 × 408 | Organische, gebündelte Faserzüge in einer transparenten Gehirnform, unterschiedlich tiefe Schichten. | Ergänzende Faserstruktur. Getty-/Urhebermarkierung bleibt auf der privaten Referenz erhalten. |
| Bild 4 · zuletzt einzeln beigefügt | Seitlich-schräge, fast körperlose Gehirnform aus sehr feinen blauweißen Fasern; viele kleine goldene Punkte und wenige gebündelte Signalwege. | **Neue Hauptreferenz:** Silhouette, Ausgangskamera, feine Faserdichte und zurückhaltendes Goldlicht. |

**Aktualisierte Gewichtung:** Das zuletzt separat gesendete Bild 4 wird als Hauptreferenz vorgeschlagen. Bild 3 ergänzt die Faserführung, Bild 1 die klar lesbaren Impulse, Bild 2 die auswählbaren Knoten im Raum. Das ist eine Gestaltungsentscheidung zur Freigabe, keine Behauptung über die technische Entstehung der Referenzbilder.

Bewegung kann aus den vier Standbildern nicht abgelesen werden. Orbit, Auswahl und Signalfluss sind deshalb ausdrücklich vorgeschlagene Interaktionen.

## Assets — vor jedem Modellieren beantworten

**Ja, Bildreferenzen existieren; nein, ein verwendbares 3D-Modell wurde im aktuellen Projekt noch nicht gefunden.** Vorhanden sind die vier beigefügten PNGs und die bisherige KI-generierte Gehirnillustration. Diese Bilder liefern Look-Referenzen, aber keine von allen Seiten korrekte 3D-Geometrie.

Vor dem Modellieren wird nach einer verwendbaren, lizenzgeklärten Gehirnoberfläche gesucht. Eine vorhandene geeignete Geometrie wird bevorzugt wiederverwendet. Die Fasern müssen wegen Auswahl, Filter und Signalfluss zur Laufzeit separat steuerbar sein. Anatomische Hülle, dekorative Fasern und tatsächlich dem Modell zugeordnete Kanten dürfen nicht verwechselt werden.

Die lokalen Referenzkopien liegen unter `3d/reference-sources/` und sind von Git ausgeschlossen. Keine Lizenz zur öffentlichen Verwendung der angehängten Bilder wird vorausgesetzt. Das bisherige generierte Bild darf als ausdrücklich benannter WebGL-Fallback bleiben, nicht als primärer 3D-Inhalt.

## §palette

Die folgenden Werte sind konkrete **Vorschläge**, visuell aus den Referenzen abgeleitet und mit Observatory kombiniert; noch kein gemessener Referenzkorridor.

| Rolle | Hex | Wofür |
|---|---|---|
| Szenengrund | `#091417` | Dunkle Bühne innerhalb des bestehenden Panels |
| Primärtext | `#eff3ef` | Lesbare UI und ausgewählte IDs |
| Blauweiß | `#a1cbd4` | Feine Kontur und nahe aktive Fasern |
| Tiefenblau | `#3b789f` | Räumlich entfernte, ruhige Faserstruktur |
| Goldlicht | `#f5d78b` | Kleine wandernde Impulse und Auswahl-Details |
| Observatory-Mint | `#b9e8cb` | Bestehende primäre Bedienelemente |

**Aktualisiert durch Nutzerwunsch am 14.09.2026:** Neuronenpunkte übernehmen exakt die vier Circuit-CSS-Farben, auch bei Aktivität, Auswahl und Regionsfiltern. Aktivität verändert Größe und Deckkraft; keine goldene oder weiße Überfärbung der Modellrolle. Gold bleibt ausschließlich für die bereits illustrierten wandernden Faserimpulse. Anatomische Atlas-Labels werden getrennt von Modellfarben geführt; keine erfundene biologische Zuordnung der vier SNN-Schichten.

## §type

- **Überschriften:** lokale System-Sans, Weight 500, `clamp(24px, 2.5vw, 37px)`, letter-spacing `-0.04em`; bestehende Observatory-Hierarchie.
- **Messwerte und Neuron-IDs:** SFMono-Regular / Consolas / monospace, Weight 400, `clamp(11px, 0.85vw, 13px)`, letter-spacing `0`; IDs nur bei Hover oder Auswahl im Raum.
- **Bedienung und Erklärungen:** lokale System-Sans, Weight 400, `clamp(12px, 1vw, 14px)`, line-height 1.55, letter-spacing `0`.

Die Referenzen enthalten keine geeignete UI-Typografie. Diese Rollen werden aus dem gewählten Dashboard übernommen, nicht aus den Stockbildern vermeintlich extrahiert.

## §materials

1. **Orientierende Hülle:** überwiegend transparent; die Silhouette und wenige Faltenkämme werden im Streiflicht sichtbar. Sie darf die Tiefe des Fasernetzes nicht als milchige Wand verdecken.
2. **Fasern:** dünne, räumlich geschwungene Bahnen; nahe Abschnitte klarer, entfernte dunkler. Keine flachen, zufälligen Linien vor einer Textur. Faserbündel sollen nachvollziehbare Gruppen bilden.
3. **Knoten und Impulse:** kompakte helle Kerne mit begrenztem Halo. Wärme nur an wenigen aktiven Stellen, kein dauerhaft überbelichtetes Gesamtgehirn und kein ungerichtetes Partikelkonfetti.

## §camera

- Perspektivkamera, geplanter Brennweitenbereich **40–65 mm**; Startwert **50 mm** als zu prüfender Entwurfsparameter.
- Startpose: leicht erhöhte seitliche Dreiviertelansicht nach der zuletzt gesendeten Hauptreferenz; etwa 20° gegenüber einer strengen Seitenansicht und 15° Elevation als Ausgangspunkt. Im Blockout anhand echter Geometrie überprüfen.
- Manueller Orbit: 360° horizontal; vertikal begrenzt, sodass die Orientierung nicht versehentlich kippt. Dolly relativ zur automatisch berechneten Fit-Distanz ungefähr 60–150 %.
- Keine automatische Hero-Fahrt. Direkte Maus-/Touch-Steuerung und kurze, begrenzte Wechsel zu Standardansichten; Reset stellt Pose, Zielpunkt und Distanz wieder her.
- Auf kleinen Bildschirmen dieselben Kernfunktionen, aber weniger sekundäre Fasern und keine permanent schwebenden Labels. Seitenscroll bleibt benutzbar.

## §measurable

Vier unveränderte Nutzerreferenzen sind mit `measure-anchors.py` vermessen; Rohwerte in `anchors/corridor.json`. Bild 4 bleibt Hauptanker. Die Werte gelten für die Referenzbilder, nicht als behauptete Messwerte des Browser-Komposits.

| Kennzahl | Zielkorridor / Status | Quelle |
|---|---|---|
| `dynamik` (P95−P5) | Referenzen 124,2–181,0; Hauptanker 124,2 | Vier Referenzmessungen |
| `clippedPct` | ≤ 0,5 % als Prüfgrenze; noch nicht gemessen | Skill-Grenze |
| `farbEntropieBits` | Referenzen 4,62–7,34; Hauptanker 4,62 | Vier Referenzmessungen |
| `dominantAnteil` | Pro Anker in `anchors/corridor.json`, volle Bildfläche, 5-Bit-Farbbins | Referenzmessungen |
| `kantendichte` | Referenzen 0,3364–0,4746; Hauptanker 0,3364 | Sobel-Schwelle 60, volle Bildfläche |
| Tone Mapping | NeutralToneMapping für Kontur/Fasern; kategorische Neuronenpunkte mit `toneMapped: false`, normaler Alpha-Mischung und sRGB-Ausgabe für identische Circuit-Farbwerte | Browser-Sichtprüfung + Farbvertragstest |

**Funktionale Abnahmekriterien:** Beim Orbit ändern sich Überdeckung und Perspektive wirklich; Auswahl entspricht einer stabilen Modell-ID; Verbindungen im Inspector und im Raum stimmen überein; Pause hält Impulse an; Reset funktioniert; Filter und Auswahl überleben den Ansichtswechsel; keine horizontale Seitenüberbreite bei 390 px. Reduzierte Bewegung und WebGL-Ausfall erhalten einen klar benannten Standbild-/Schaltungsweg.

**Vorläufiges Performance-Ziel:** Interaktion auf dem tatsächlichen Desktop flüssig; konkrete Framezeiten, Draw-Calls, Ladezeit und Speicherverhalten werden gemessen, nicht zugesagt. Ein persistentes Canvas, keine neuen Render-Schleifen pro Ansicht und keine dauernde Wiedergabe im unsichtbaren Bereich.

## §tuning

Noch keine skalaren Rückmeldungen oder Messungen; keine geratenen Einträge.

| param | tooHigh | tooLow | current | verifiedAt | shotPath |
|---|---|---|---|---|---|

## §LOCKED

- `[LOCKED 2026-09-14]` Nutzerwunsch: „wirklich 3d“ — die Hauptansicht darf nicht erneut ein PNG mit 2D-Overlay sein.
- `[LOCKED 2026-09-14]` Nutzerwunsch: „mehr interaktv“ — echte räumliche Erkundung und sinnvolle Auswahl gehören zum Kernumfang.
- `[LOCKED 2026-09-14]` Nutzerentscheidung aus dem bisherigen Verlauf: Observatory bleibt die Designrichtung, mit der Verständlichkeit von Symbiosis.
- `[LOCKED 2026-09-14]` Die vier neuen Referenzen bestimmen die visuelle Richtung; der vorherige massive graue Gehirn-Look ist nicht der Zielmaßstab.

## §rejections

| Datum | Wortlaut | Abgeleitete Constraint |
|---|---|---|
| 2026-09-14 | „die darstellung des snn muss besser weden es muss wie ein echtes gehirn sein“ | Eine generische regionale Punktwolke allein erfüllt die gewünschte Gehirnform nicht. |
| 2026-09-14 | „muss nich besser werden mehr interaktv und wirklich 3d und mehr wie audf den bildern“ | Die neue Primäransicht braucht echte räumliche Geometrie und die transparente, leuchtende Netz-/Fasersprache der beigefügten Referenzen. |

## Freigabe

**Freigegeben am 14.09.2026:** „ok bau das gehirn so, aber als interaktives 3d modell“. Der lokale interaktive Prototyp ist umgesetzt; Ergebnis und Tests in `docs/dashboard-concepts/BRAIN-3D.md`. Offen bleibt die menschliche visuelle Endabnahme, nicht mehr die Erlaubnis zur Umsetzung. Die vollständige neunphasige Studio-Abnahme wird nicht als bestanden behauptet.
