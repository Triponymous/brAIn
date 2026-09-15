# Circuit-Farben und anatomische Orientierung

## Plan vor der Änderung · 14. September 2026

Begrenzte Korrektur der freigegebenen Fasergehirn-Richtung, kein neues 3D-Design.
Der web3d-director-Skill führt zur Wiederverwendung der bestehenden Atlas-Geometrie
und zur getrennten Prüfung von Farbtreue, Raumlage und Interaktion.

1. Circuit und 3D lesen dieselben vier CSS-Farbwerte. Aktivität verändert Größe
   und Deckkraft, nicht die Bedeutung der Farbe. Auch Auswahl und Filter färben
   einen Input-Punkt nicht in einen Memory-Punkt um.
2. Die Punktkerne erhalten eine direkte Linear-sRGB → sRGB-Ausgabe ohne
   Tone-Mapping, additive Farbverschiebung oder weiße Überstrahlung.
3. Der vorhandene Z-Anatomy/BodyParts3D-Bake behält die benannten Quellstrukturen
   und ihre Oberflächenkoordinaten. Kein radiales Zusammenschieben in unbekanntes
   Gewebe; jede Modell-ID erhält einen stabilen Punkt auf einer Atlasfläche.
4. Frontal-, Parietal-, Temporal- und Occipitalregion sowie Kleinhirn und Hirnstamm
   werden durch separat schaltbare, aus den Originalstrukturen abgeleitete
   Orientierungspunkte beschriftet. Linke/rechte Hemisphäre und Kameraachsen
   werden gegen die Namen im Originalmodell geprüft.
5. Tests vergleichen Farben mit Circuit, Oberflächenpunkte mit dem Bake,
   Richtungen, stabile IDs, Auswahl, Filter, Replay und Live/Demo-Wechsel.

## Wissenschaftliche Grenze

Input, Expansion, Concepts und Working memory sind **Modellschichten**, keine
anatomischen Gehirnlappen. Das Projekt liefert keine biologische Lokalisation
dieser 1.000 Recheneinheiten. Eine fixe Zuordnung „Memory = Hippocampus“ oder
„Expansion = bestimmter menschlicher Hirnlappen“ wäre unbelegt. Die Modellrollen
bleiben deshalb über die Atlasoberfläche verteilt; anatomische Namen beschreiben
die Vorlage, nicht den biologischen Ort einer berechneten Funktion.

Die beschrifteten Punkte sind Orientierung an benannten Strukturen, keine
vollständige Parzellierung, individuelle Bildgebung oder klinische Registrierung.
Die Fasern bleiben illustrative Beispielpfade; in Live bleiben sie ausgeblendet.

## Quellen

- [NINDS: Know Your Brain](https://www.ninds.nih.gov/sites/default/files/2025-05/know-your-brain-brian-basics.pdf):
  räumliche Lage der Hirnlappen; Text über den Suchindex zugänglich, PDF-Abruf 403.
- [Kamiński et al., Nature Neuroscience 2017](https://www.nature.com/articles/nn.4509):
  Humanstudie zu verteilten Aktivitätsrepräsentationen im Arbeitsgedächtnis;
  Abstract über den Suchindex zugänglich, Volltextabruf nicht verfügbar.
- [Three.js Color Management](https://threejs.org/manual/en/color-management.html):
  sRGB-/Linear-sRGB-Konvertierung; zusätzlich Prüfung der lokal verwendeten
  r180-Implementierung. Keine Aktualisierung der vendorten Bibliothek.
- [Atlas-Lizenz und Herkunft](https://github.com/DrMuratAltun/beyin-simulatoru/blob/main/LICENSE-DATA.md),
  lokale Quell-Hash- und Credit-Angaben in [ASSETS.md](ASSETS.md).

## Prüfergebnis

Umgesetzt und lokal geprüft am 14. September 2026.

| Prüfung | Ergebnis |
|---|---|
| 3D-Verifikation | PASS, einschließlich aller 1.000 eindeutigen Oberflächenpositionen |
| Gemeinsame Farbquelle | `--s #9dc7d7`, `--e #bcb0d9`, `--c #b9e8cb`, `--w #e0bc87` |
| Farbtreue | Alle Punkte ergeben nach Linear-/sRGB-Rückumwandlung exakt den Circuit-Hexwert |
| Zustandswechsel | 5 Filter × 4 Auswahlen × 3 Aktivitätszustände ohne Änderung der Grundfarben |
| Auswahl | Vergrößerter Marker; eine inaktive Einheit wird durch Auswahl nicht aktiv gezeichnet |
| Anatomie | 12 Quellanker, 6 Regionen, geprüfte Links-/Rechts- und anterior/posterior-Achsen |
| Projektionslayout | 20 Kombinationen aus Kamera und Größe; keine gleichseitigen Labelkollisionen |
| Dashboard-Regression | 33 bestehende Tour-, Demo-, Live- und Capture-Tests bestanden |
| Browser | Chrome, sichtbares Dokument und echte WebGL-Render-Aufrufe |
| Ansichten | Side, Front, Top, Perspective; sechs sichtbare Atlas-Anker |
| Interaktion | Atlas ein/aus; Kamera lässt Inspector unverändert; Memory-Filter und Frame 42→43 |
| Circuit-Roundtrip | Identische Auswahl und Frame, dieselben vier berechneten CSS-Farben |
| Mobile | 390 × 844: sechs Labels, keine Überdeckung durch Atlas-Schalter, kein horizontaler Überlauf |
| Live | Tatsächliche Modellbeobachtung mit gleicher Palette, einem Canvas, ohne Beispielkanten |
| Renderumfang | Demo 5 Draw-Calls; Live ohne Kanten 3 Draw-Calls |
| Datenschutz | Erfassungsrevision 4 unverändert; Test-Ansicht danach getrennt, Runner nicht verändert |
| Konsole | Keine Warnungen oder Fehler im abschließenden Browsercheck |

Zusätzlich behoben: Die globale Regions-Klickbehandlung hatte auch das diagnostische
`data-region` des 3D-Containers getroffen. Kamera-/Atlas-Klicks wechselten dadurch
ungewollt den Inspector. Sie reagiert jetzt ausschließlich auf Regionsbuttons;
auch die freiwillige Tour-Übung verwendet diesen engeren Selektor. Circuit-Nachbarn
behalten bei Auswahl ebenfalls die Farbe ihrer eigenen Modellrolle.

Eine erste Layoutprüfung deckte einen zu engen Sichtbarkeitsbereich auf; korrigiert
und erneut PASS. Die mobile Browserprüfung fand einen vom Atlas-Schalter verdeckten
Parietal-Text; Abstand ergänzt, Regressionstest erweitert, Browser-Nachprüfung ohne
Überdeckung. Versionierte lokale Asset-URLs vermeiden veraltete dynamische Module
im Browsercache. Der vorhandene Atlas wird weiterverwendet, nicht neu modelliert.

Farbgleichheit bezeichnet die Grundfarben und denselben Opazitätszustand wie in
Circuit. Transparente Halos, Antialiasing, Hintergrund und Pixeldichte können
einzelne Bildschirmrandpixel verändern; keine pixelidentischen Screenshots behauptet.
Keine vollständige Studio-, medizinische oder neuroanatomische Abnahme behauptet.
Kein GPU-/FPS-Benchmark, kein klinischer Atlas und keine biologische Funktionskarte.

```sh
rtk proxy node 3d/verify-brain.mjs
rtk proxy node --test docs/dashboard-concepts/verify-onboarding.mjs docs/dashboard-concepts/verify-observatory.mjs docs/dashboard-concepts/verify-live.mjs docs/dashboard-concepts/verify-capture.mjs
```
