# Geometrieaufbereitung

`rtk proxy node 3d/bake-brain.mjs`

1. Standard-GLB lesen; alle Elterntransformationen auf ausgewählte Anatomieteile anwenden.
2. 89 Strukturen auswählen: Cortex, Cerebellum und Stammanteile. Anatomische Form nicht neu modelliert.
3. Zentrieren, skalieren und seitliche Präsentationsachse wählen.
4. Dreiecksoberflächen mit schrägen Ebenen schneiden: 68.251 feine Kontursegmente.
5. 4.968 deterministische Oberflächenpunkte als Ausgangspunkt der SNN-Anordnung samplen.
6. Benannte Sample-Bereiche und zwölf Orientierungspunkte aus sechs tatsächlichen
   Quellstrukturen pro Hemisphäre in glTF-Extras erhalten. Die Anker sind jeweils
   ein Originalvertex nahe dem Schwerpunkt der benannten Teilstruktur.
7. Als Standard-glTF-2.0-GLB mit LINES- und POINTS-Primitiven exportieren.

Präsentationsachsen, gegen die benannten Originalteile geprüft: −X anterior,
+X posterior, +Y superior, +Z linke und −Z rechte Hemisphäre. SNN-Punkte werden
ohne radiales Schrumpfen aus den 4.968 Cortex-Samples ausgewählt. Die Farben
bezeichnen Rechenrollen, nicht diese anatomischen Regionen.

Keine Material-, Licht- oder Kamera-Bakes: Das Rendering ist eine unbeleuchtete
Datenzeichnung, keine reflektierende Materialsimulation. Die Kamera reagiert direkt
auf Orbit-Eingaben; Verbindungen, Auswahl und Impulse müssen zur Laufzeit auf den
synthetischen SNN-Zustand reagieren. Es gibt keine choreografierte Kamerafahrt.

Die Konturen sind Anatomie, **nicht** gemessene Nervenbahnen. Die gebündelten
Verbindungskurven entstehen aus dem synthetischen Verbindungskatalog. Das glTF-Asset
und die daraus abgeleitete Anatomie-Anordnung unterliegen dem Attribution-/ShareAlike-Hinweis.

Bewusster Umfang des Prototyps: Float32-GLB, eine Detailstufe, kein Draco/Transcoder.
Library-Payload ca. 0,90 MB plus Anatomie ca. 1,70 MB (unkomprimiert). Damit ist die
strenge 2-MB-Gesamtvorgabe des Studio-Skills nicht erfüllt; kein formales Phase-7-Pass.
LOD/Quantisierung und echtes Mobilgeräte-Profiling sind vor einer Produktionsfreigabe offen.
