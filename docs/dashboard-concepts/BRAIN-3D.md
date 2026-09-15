# Observatory — interaktives Fasergehirn

**Fortführung:** Das 3D-Modell wurde in Research workspace 03 übernommen. Punktaktivität,
Circuit, Inspector und Raster verwenden jetzt ein gemeinsames versioniertes Fixture-Dataset.
Die folgenden Belege beschreiben den ursprünglichen 3D-Ausbau; aktuelle Dashboard-Tests,
Zeitbasis und Grenzen stehen in [RESEARCH-WORKSPACE.md](RESEARCH-WORKSPACE.md).

14. September 2026 · [Vorschau](http://127.0.0.1:4178/observatory.html#network)

## Umsetzung

Echtes WebGL mit räumlicher Anatomiegeometrie statt Bildrotation: 89 Strukturen aus
Z-Anatomy/BodyParts3D bilden zwei Hemisphären, Windungen, Cerebellum und Stammanteile.
68.251 feine Kontursegmente tragen die blauweiße Form; Gold markiert synthetische
Aktivität. Die letzte Nutzerreferenz bestimmt diese Farb- und Fasersprache.

1.000 stabile SNN-Einheiten, 678 sichtbare Beispielpfade aus dem vorhandenen Katalog
mit 3.386 Kanten. Die Auswahl ergänzt sämtliche angrenzenden Beispielverbindungen,
auch wenn sie nicht in der sichtbaren Gesamtstichprobe liegen. Kein Punkt ist eine
Behauptung über die biologische Lokalisation einer SNN-Region.

## Bedienung

- Ziehen: Kamera um das Modell drehen; Scrollen/Pinch: zoomen.
- Perspective / Side / Front / Top: reproduzierbare Ansichten. Reset setzt die Kamera zurück.
- Punkt anklicken: Modell-ID, Werte und dieselben Nachbarverbindungen im Inspector.
- Tastatur: Canvas fokussieren, Pfeile drehen, +/− zoomen, Home setzt zurück.
- Präzise Auswahl: Single neuron → Region / Index → Find.
- Contours und Connections schalten Orientierung und Verbindungen getrennt.
- Activity spielt goldene Impulse entlang der Kurven; Pause und Frame-Slider halten den Zustand fest.
- Brain / Circuit teilen Auswahl, Region, Szenario und Aktivitätsframe.

## Geprüft

| Prüfung | Ergebnis |
|---|---|
| Maus-Orbit | Kameraposition änderte sich von `2.114,1.149,6.892` zu `5.518,2.986,3.746`; tatsächliche Überdeckung und Silhouette ändern sich. |
| Neuron anklicken | Sichtbarer 3D-Punkt wählte `S-136`; Inspector zeigte dieselbe ID. |
| Keyboard / Zoom / Reset | Pfeiltaste änderte die Kamera; Zoom veränderte Entfernung; Reset stellte Perspektive wieder her. |
| Regionsfilter | Expansion wählte `E-042` und zeigte korrekt Projektionsoutput statt LIF-Membran. |
| Brain → Circuit → Brain | `E-042` und Filter erhalten; Circuit enthält alle 1.000 Einheiten. |
| Konturen / Kanten aus | Nur Punkte sichtbar, Draw-Calls von 5 auf 1. |
| Playback | Frames und Subframe-Zeit verändern sich; Pause hält an. Über 3.600 Render-Aufrufe blieben 5 GPU-Geometrien konstant. |
| Desktop / schmal | Visuell bei 1440, 677 und 390 CSS-Pixeln geprüft; bei 390 keine horizontale Überbreite. Das ersetzt keinen Test auf einem physischen Smartphone. |
| WebGL-Fallback | `?WEBGL_OFF=1#network` zeigt ausdrücklich eine statische Illustration; Circuit bleibt mit 1.000 Einheiten bedienbar. |
| Statische Tests | `rtk proxy node 3d/verify-brain.mjs`: Syntax, lokale Assets, 1.000 IDs, deterministische 3D-Positionen, alle Kantenendpunkte und Farbfilter bestanden. |
| Browserkonsole | Keine Fehler oder Warnungen im normalen 3D-Pfad. |

5 Draw-Calls im vollständigen Bild. Beobachtet: ca. 396 ms initiale Viewer-Vorbereitung
und 39–53 ms warm, Renderer-CPU-Submission-P95 0,50 ms in einem 300-Sample-Fenster.
Das sind lokale Entwicklungsbeobachtungen, **keine** GPU-Zeit, FPS-Garantie oder
reproduzierbare Ladezeit-Benchmark. Shader-Look wurde anschließend von AgX auf
Neutral umgestellt, damit Blau/Gold klarer getrennt bleiben.

## Lebenszyklus und Grenzen

Ein persistentes Canvas über beide Netzwerkseiten und Circuit-Wechsel. Ein vorhandener
RAF-Besitzer für Playback; sonst Rendering nur bei Eingaben oder Zustandsänderungen.
Keine GPU-Zeichnung außerhalb des sichtbaren Graphbereichs. DPR gedeckelt,
Resize entprellt, Geometrieaustausch mit Dispose. Context-Loss hat einen sichtbaren
Fehlerzustand; Reduced Motion blockiert Playback und erhält manuelles Scrubbing.
Diese beiden OS-/GPU-Ereignisse wurden im Quellcode geprüft, nicht künstlich im Browser ausgelöst.

Alle SNN-Werte und Faserverbindungen bleiben synthetisch. Kein Patientenscan,
keine gemessene Traktografie, keine Live-Anbindung und keine Änderung am Backend.
Kein Hosting, Commit oder Push. Die menschliche visuelle Abnahme ist offen.

## Dateien und Quellen

`brain-view.js`: Adapter zum Dashboard. `brain-3d/viewer.js`: Kamera, Rendering,
Raycasting und Lebenszyklus. `brain-3d/network.js`: Kurven, Punkte und Aktivität.
`brain-view.css`: integrierte Bedienoberfläche. `assets/brain-contours.glb`: Anatomie.
Die alte 2D-Koordinatenliste wurde entfernt; das frühere PNG bleibt als Fallback.

[Anatomiequelle und Lizenz](assets/brain-LICENSE.md), [Three.js MIT](vendor/three/LICENSE.txt).
Reproduzierbare Aufbereitung: `3d/bake-brain.mjs`; Quellen und Grenzen: `3d/ASSETS.md`, `3d/BAKE.md`.
Die Vorgaben aus `web3d-director` und `motion-principles` beeinflussen Quellenwahl,
echte Geometrie, begrenzte Bewegung und Prüfung. Die vollständige neunphasige
Studio-/Produktionsabnahme wurde nicht behauptet; LOD, Quantisierung und echtes
Mobilgeräte-Profiling bleiben für eine spätere Produktionsintegration offen.
