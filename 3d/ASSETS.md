# Anatomie und Bibliotheken

## Anatomisches Ausgangsmodell

Quelle: [DrMuratAltun/beyin-simulatoru · brain.glb](https://github.com/DrMuratAltun/beyin-simulatoru/blob/main/brain.glb).
Lizenz des Modells: [CC BY-SA 4.0](https://github.com/DrMuratAltun/beyin-simulatoru/blob/main/LICENSE-DATA.md).
Attribution: BodyParts3D © DBCLS (CC BY-SA 2.1 JP), Z-Anatomy contributors (CC BY-SA 4.0),
GLB-Export Dr. Murat Altun. [Originale Quellenangaben](https://github.com/DrMuratAltun/beyin-simulatoru/blob/main/ATTRIBUTION.md).

SHA-256 des verwendeten Downloads:
`16e7d8c61a1094059c5084ff42af1c7bfaf762c99f3c3c1720fb3df0eb0ac6d3`.

Runtime-Derivat: `docs/dashboard-concepts/assets/brain-contours.glb`, 1.705.804 Bytes,
CC BY-SA 4.0. Derivat-Hinweis und vollständige Quelle in `assets/brain-LICENSE.md`;
sichtbarer Credit direkt unter dem 3D-Graph. Das Rohmodell liegt ignoriert unter
`3d/sources/`; keine Fremdlogos oder Texturen übernommen. Die Referenzbilder sind
nicht Teil des Runtime-Assets und werden nicht veröffentlicht.

Ergänzung 14.09.2026: Benannte Quellstrukturen je Sample-Bereich und zwölf
anatomische Orientierungspunkte (sechs Regionen, beide Hemisphären) liegen als
glTF-Extras im selben Asset. Keine neue Modellquelle; Konturgeometrie unverändert.

## Three.js

Three.js **0.180.0**, MIT, lokal unter `vendor/three/` und `vendor/utils/`.
Originaldateien von `cdn.jsdelivr.net/npm/three@0.180.0/`: Core, WebGL-Modul,
OrbitControls, GLTFLoader, BufferGeometryUtils. Lizenztext lokal `vendor/three/LICENSE.txt`.
Keine Änderungen an vendortem Code. Keine CDN-Abfragen zur Laufzeit.

API-Grundlage: [offizielle Three.js-Dokumentation](https://threejs.org/docs/llms.txt)
und die tatsächlich verwendeten r180-Modulquellen. Import Map, WebGLRenderer,
BufferGeometry, Points, Raycaster und OrbitControls; kein zusätzlicher Postprocessing-Stack.

## Fallback

Das bereits in dieser Aufgabe generierte `brain-observatory.png` bleibt ausschließlich
als klar bezeichnete statische Illustration bei fehlendem WebGL. Herkunft und Prompt
sind im historischen `docs/dashboard-concepts/BRAIN-VISUAL.md` dokumentiert.
