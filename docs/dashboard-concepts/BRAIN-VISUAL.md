# Observatory — archivierter 2D-Vorentwurf

**Abgelöst durch [BRAIN-3D.md](BRAIN-3D.md).** Die folgenden Angaben dokumentieren den früheren Bild-/SVG-Ansatz. Das Bild bleibt ausschließlich als statischer WebGL-Fallback erhalten; das heutige Brain ist echte 3D-Geometrie. `brain-points.js` wurde als ungenutzte 2D-Koordinatenliste entfernt.

14. September 2026 · Änderung auf Leons Wunsch: „es muss wie ein echtes gehirn sein“

## Ergebnis

Die Hauptansicht des SNN zeigt jetzt eine plastisch wirkende Gehirnform mit zwei Hemisphären, zentraler Furche und vielen kortikalen Windungen. Die Illustration trägt eine separate, bedienbare Ebene mit den 1.000 synthetischen SNN-Einheiten.

- **Brain:** neue Hauptansicht; Neuronen per Zeiger oder Inspector auswählen.
- **Surface:** Illustration zurückblenden, damit die Neuronenebene hervortritt.
- **Circuit:** bisherige Darstellung der vier Modellregionen als technische Alternative.
- Regionenauswahl, ausgewählte Einheit und Aktivitätsframe bleiben beim Ansichtswechsel erhalten.
- Verbindungen stammen aus demselben synthetischen Katalog wie die Nachbar-Schaltflächen.
- Aktivität direkt am Graph starten oder pausieren; die Steuerung ist mit dem Footer synchron.
- Bestehende Zustandsanzeige, Lernfunktion, Vokabular und Journal bleiben unverändert.

## Herkunft und Grenzen

Das Gehirnbild ist **KI-generierte, anatomisch inspirierte Illustration**, kein Patientenscan und kein validierter anatomischer Atlas. Es ist eine zweidimensionale Grafik mit räumlicher Wirkung, kein frei drehbares 3D-Modell.

Die Koordinaten der 1.000 auswählbaren Einheiten wurden aus sichtbaren Bereichen der Illustration abgeleitet. Sie sind weder gemessene Positionen biologischer Neuronen noch Positionen eines tatsächlichen anatomischen SNN-Modells. Die Farben kennzeichnen die vier rechnerischen Modellregionen, nicht biologische Gehirnlappen. Telemetrie und Aktivität bleiben synthetische Demo-Daten.

## Dateien

- [observatory.html](observatory.html): bestehendes Dashboard, neue Hauptansicht integriert.
- [brain-view.js](brain-view.js): Vektor-Overlay, Zeigerauswahl, Verbindungen und Ansichtswechsel.
- [brain-view.css](brain-view.css): Layout und responsive Darstellung.
- `brain-points.js` (inzwischen entfernt): 1.000 aus dem Bild abgeleitete Illustration-Koordinaten im damaligen Vorentwurf.
- [assets/brain-observatory.png](assets/brain-observatory.png): unveränderte Kopie des generierten Bildes, 1.254 × 1.254 px, Alpha-Kanal, ca. 1,8 MB.

Keine neuen Bibliotheken, kein Buildschritt, keine Sensoren, keine Backend-Änderungen und keine externen Requests im Dashboard. Zum Übertragen den gesamten Ordner verwenden, nicht nur die HTML-Datei.

## Bildgenerierung

**Modus:** eingebautes Bildgenerierungswerkzeug (`imagegen`), kein API-/CLI-Fallback.
**Projektpfad:** `docs/dashboard-concepts/assets/brain-observatory.png`.
Die ursprüngliche generierte Datei bleibt unverändert am Ausgabeort erhalten. Die Pixel wurden ausschließlich gelesen, um Koordinaten für die darüberliegende Neuronenebene zu bestimmen; das Bild wurde nicht nachträglich bearbeitet.

### Verwendeter Prompt

```text
Use case: stylized-concept
Asset type: anatomically inspired brain illustration used as the central substrate of an interactive spiking-neural-network research dashboard. Generate ONLY the isolated brain asset, not a dashboard.
Primary request: an immediately recognizable, extraordinarily refined human brain, with realistic complex cortical gyri and deep sulci, two distinct cerebral hemispheres and the longitudinal fissure. A sophisticated scientific imaging aesthetic, not a cartoon brain or a generic cloud of dots.
Composition: square image, orthographic superior-dorsal view with a very slight anterior tilt, both hemispheres clearly visible symmetrically around a nearly vertical central fissure. Whole brain centered, about 78% of image width and 86% of image height, comfortable 7% blank border. No floating cut sections. No spinal cord or extraneous anatomy. Preserve realistic mildly asymmetric intricate winding folds.
Material and lighting: translucent smoky graphite cortical tissue with fine silvery mineral-sage highlights that trace the crests of the gyri; subtly luminous milky MRI-like volumetric surface, rich dark valleys, gentle depth and exquisite fine anatomical detail. Soft cool side illumination and muted mint rim light. Keep it dark enough for an overlay of bright, clickable neuron nodes, but clearly recognizable. Sophisticated, calm, premium research instrument aesthetic. Not wet tissue, not metallic chrome, not a wire sphere.
Background: perfectly uniform solid very dark blue-green #111b1e right up to the edges, no floor, no horizon, no cast shadow, no texture, no framing UI. The brain itself must be visibly brighter than the background and the edges clean.
No text, no labels, no lettering, no logo, no watermark, no color legend. Do not draw neural nodes, connecting lines or charts: those will be real interactive vector elements added later. This is illustrative design art, not an actual medical scan.
```

## Prüfung

- JavaScript-Syntax des Dashboards und beider neuen Skripte geprüft.
- Brain und Circuit rendern jeweils 1.000 auswählbare Einheiten.
- Direkte Auswahl von C-101 geprüft; ID bleibt beim Wechsel Brain → Circuit → Brain erhalten.
- Surface-Umschaltung, Regionsfilter, Expansion-Metrik, Zoom, Verbindungen und beide Aktivitätssteuerungen geprüft.
- Große Desktop-Aufteilung bei 1.440 px und mobile Ansicht bei 390 px visuell geprüft, ohne horizontales Seiten-Overflow.
- Herkunftserklärung auf schmalem Bildschirm geprüft.
