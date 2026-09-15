# 3D-Projektzustand — brAIn / Observatory

PHASE 0 APPROVED 2026-09-14T11:43:35Z

Leon: „ok bau das gehirn so, aber als interaktives 3d modell“.
Tier 2 und die letzte Fasergehirn-Richtung sind freigegeben.

## Stand 2026-09-14

Interaktiver lokaler 3D-Prototyp umgesetzt und im Browser geprüft.
Vorschau: http://127.0.0.1:4178/observatory.html#network
Ergebnis und Testbelege: docs/dashboard-concepts/BRAIN-3D.md.

Echte Z-Anatomy/BodyParts3D-Geometrie, Orbit, Zoom, Kamera-Presets, 1.000 per
Raycasting auswählbare Einheiten, 678 gebündelte Beispielpfade, goldene Impulse.
SNN-Daten bleiben synthetisch; kein Backend, Hosting, Commit oder Push.

## Freigaben und Grenzen

Die ausdrückliche Bauanweisung wurde direkt umgesetzt, ohne erneut Richtungsvarianten
zur Auswahl zu stellen. Nur die Phase-0-Freigabe ist menschlich belegt.
Die neunphasige Studio-Abnahme ist NICHT als bestanden markiert. Finale menschliche
Look-/Motion-Abnahme offen; Nutzer kann jetzt das Modell selbst drehen und erkunden.

Vier Referenzen mit read-only Bildanalyse vermessen (anchors/corridor.json).
Eigene Browserbilder wurden visuell, nicht mit einem vollständigen Komposit-Metrik-Gate geprüft.
Skill-Browser-Skripte wurden nicht verwendet: UI-Tests ausschließlich über CUA.

Statische Tests: rtk proxy node 3d/verify-brain.mjs — PASS.
Browser: echter Maus-Orbit, S-136-Ray-Pick, Zoom, Keyboard, Regionsfilter,
Circuit-Roundtrip, Wiedergabe, 1440/677/390px und expliziter WebGL-Fallback geprüft.
Fünf Draw-Calls und fünf Geometrien stabil während über 3.600 Render-Aufrufen.

Produktionsschritte offen: Asset-Quantisierung/LOD (aktueller unkomprimierter
Runtime-Payload rund 2,6 MB), reales Mobilgeräte-/GPU-Profiling, erzwingbarer
Context-Loss-Test. Keine fiktiven GPU-, FPS- oder Abnahme-Ergebnisse.

Keine Memory-Dateien geschrieben. Frühere 2D-Ablehnungen stehen im BRIEF.md.

## Dashboard-Fortführung / Research workspace 03

Leon bewertete die 3D-Richtung mit „ok nice“ und beauftragte die professionelle
Ausarbeitung des gesamten Dashboards. Anatomie, Kamera und Renderer bleiben erhalten;
die Punktaktivität verwendet jetzt dieselben deterministischen Fixture-Serien wie
Inspector, Circuit und Raster. Das ist keine neue Behauptung einer Studiogate-Abnahme.

Dataset, Zeitbasis, Grenzen, JSON-Snapshot und aktuelle Tests:
docs/dashboard-concepts/RESEARCH-WORKSPACE.md. verify-brain.mjs weiterhin PASS.
Keine Memory-Dateien, Backend-Änderungen, Veröffentlichung oder Deployment.

## Fortführung: nur lesende Modellbeobachtung

Die zusätzliche Live-Ansicht verwendet denselben Renderer und dasselbe Canvas.
Anatomie, Kamera und Materialien bleiben bestehen; der Datenadapter übernimmt
echte binäre Ausgaben des Python-Cores. Im Live-Modus sind alle Beispielkanten
ausgeblendet. Die Post-Step-Membran wird separat numerisch/als Zeitreihe gezeigt.

`node 3d/verify-brain.mjs`: PASS. Browser: ein Canvas, identischer selektierter
Tick/Einheit in Live-Inspector und 3D, Demo-/Live-Rückkehr geprüft. Die Testeingänge
waren ausdrücklich künstlich; noch keine freigegebene persönliche Datenerfassung.
Detailbelege und Grenzen: `docs/dashboard-concepts/LIVE-DATA.md`.
Keine neue visuelle Richtung und keine zusätzliche Studiogate-Freigabe behauptet.

Nach der ausdrücklichen Nutzerfreigabe am 14. September 2026 wurde der lokale
Metadaten-Runner aktiviert. Echte App-Kategorie speist den Core; Tastatur, Maus und
Leerlauf sind wegen fehlender macOS-Eingabeüberwachung noch nicht verfügbar.
Follow, identischer Tick in Inspector/3D und genau ein Canvas erneut geprüft.
Keine Änderungen an der 3D-Szene, keine Gesundheitsdaten und keine Persistenz.

## Fortführung: Circuit-Farben und Atlas-Anker

Nutzerwunsch: identische Farben der Neuronen in Brain und Circuit, anatomisch
korrekte räumliche Orientierung. Begrenzte Korrektur der freigegebenen Richtung;
keine neue Studio-Phasenfreigabe und keine neue Modell-/Sensorarchitektur.

Die Punktfarben lesen jetzt dieselben vier CSS-Variablen wie Circuit. Aktivität
verändert Größe/Deckkraft, nicht den Farbton; Auswahl allein erzeugt kein Leuchten.
Punktmaterial ohne Tone-Mapping oder additive Farbverschiebung, nach den Fasern
gerendert. Die Originalkonturen bleiben; 1.000 eindeutige Sample-Positionen werden
nicht mehr radial in unbekannte innere Strukturen verschoben.

Zwölf aus benannten Atlas-Meshes abgeleitete Anker beschriften sechs anatomische
Regionen und die jeweilige Hemisphäre. −X anterior, +X posterior, +Y superior,
+Z links, −Z rechts gegen die Quelldaten geprüft. SNN-Rollen bleiben ausdrücklich
ohne behauptete biologische Lokalisation; Labels sind keine vollständigen Lappengrenzen.

Verify: `node 3d/verify-brain.mjs` PASS; 33 Dashboard-/Onboarding-/Live-/Capture-Tests
PASS. Früherer Fehler der Label-Sichtbarkeit behoben und erneut geprüft.
Mobile Überdeckung des Parietal-Labels behoben; finale 390-px-Prüfung ohne Überdeckung.
Chrome: echte WebGL-Aufrufe, alle Presets, Atlas ein/aus, Circuit-Roundtrip und
Live-Rendering ohne Beispielkanten geprüft. Ein Canvas; 5 Demo-/3 Live-Draw-Calls.
Erfassungsrevision unverändert; kein Backend-Edit, Commit, Push oder Memory-Write.

Plan, Quellen, Testmatrix und Grenzen: `3d/ANATOMY-COLOR.md`.
Keine vollständigen Screenshot-Korridor-/GPU-Performance-Gates und keine klinische
oder vollständige Studio-Abnahme behauptet; die bisherigen Produktionsgrenzen bleiben.
