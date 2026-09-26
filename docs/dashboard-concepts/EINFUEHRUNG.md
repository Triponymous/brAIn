# Geführte Einführung in brAIn Observatory

## Ziel und Plan

Zweisprachiger Rundgang (Englisch und Deutsch) direkt im vorhandenen Dashboard, kein separates Mockup.
Englisch ist der Standard passend zum Dashboard. Die Sprachwahl in Begrüßung und
Schrittkarte wechselt jederzeit zwischen English und Deutsch, ohne den aktuellen
Schritt, eine bereits bestätigte Übung oder die Sensorfreigaben zu verändern.
Nur eine ausdrücklich gewählte Sprache wird zusätzlich zum bisherigen
„Einführung gesehen“-Hinweis im Browser gespeichert. Ohne verfügbaren Speicher
funktioniert die Auswahl für den aktuellen Besuch; beim nächsten Laden gilt Englisch.
Die Sprache des restlichen Dashboards bleibt unverändert.
Die erste Nutzung bietet die Einführung an. Danach ist sie über einen dauerhaften
„Einführung“-Button und gezielt nach Kapitel erreichbar. Überspringen und Beenden
bleiben immer möglich; Übungen sind freiwillig.

1. Tatsächliche Oberfläche und Datenquellen prüfen.
2. Kapiteltexte an vorhandene Bedienelemente binden.
3. Begrüßung, Markierung, Schrittkarte und Tastaturbedienung ergänzen.
4. Erstbesuch, Neustart, Kapitelwechsel, Layout und Datenschutzgrenzen testen.

Die Graphify-Abfrage lieferte keinen vorhandenen Index. Navigation und Texte wurden
deshalb direkt anhand der HTML-/JS-Dateien und der laufenden Oberfläche geprüft.
Es wird kein neuer Repository-Graph als Nebenprodukt erzeugt.

## Was die Bereiche bedeuten

### Live session: die tatsächliche lokale Sitzung

- Oben wählst du das Modell: **Persistent brain** (`server.braind`,
  `127.0.0.1:8000`) lernt über Tage und speichert seinen Checkpoint; **Session
  runner** (`server.observe`, `127.0.0.1:8001`) beginnt jedes Mal frisch.
  **Connect local model** verbindet die Ansicht mit dem gewählten Dienst und
  schaltet keine Erfassungsquelle ein. **Disconnect view** trennt nur diese Ansicht.
  **Start daemon** und **Stop daemon** laufen über `server.control` auf Port 8900;
  die Tour startet und stoppt nichts.
- **Data & privacy:** Tastatur- und Mausaktivität sind Ereigniszähler, keine
  Inhalte oder Koordinaten. Idle time ist Zeit seit einer Eingabe; die aktive App
  zeigt die Ansicht nur als grobe Kategorie. Jeder Eingang ist getrennt schaltbar.
  Das persistente Gehirn kennt zusätzlich Mikrofon-Merkmale und speichert die
  Wahl, ohne sie nach einem Neustart je zu erweitern; der Runner hat vier Quellen
  nur für diese Sitzung. Für Tastatur, Maus und Idle ist zusätzlich eine
  macOS-Berechtigung nötig, für das Mikrofon die Mikrofon-Berechtigung.
- **Stop all capture** (Runner) und **Stop all sources** (persistentes Gehirn)
  stoppen die Erfassung und neue Modellschritte des jeweiligen Dienstes. Sie
  löschen keine vorhandenen Werte, beenden keine anderen Anwendungen und
  widerrufen keine macOS-Berechtigung. Browser schließen genügt nicht. Löschen ist
  ein eigener Schritt: `server.braind erase`.
- **Name a moment** (nur persistentes Gehirn) lehrt ein eigenes Wort für den
  aktuellen Zustand oder beantwortet eine offene Frage des Gehirns. **Recognized
  now** nennt das nächste gelernte Wort mit einer Ähnlichkeit, keiner kalibrierten
  Wahrscheinlichkeit; pausiert gibt es keinen aktuellen Zustand.
- **Captured model tick** ist der Modellschritt. **Active outputs** zählt aktive
  Ausgänge am ausgewählten Schritt, nicht Hertz. **Retained samples** ist ein
  begrenztes Fenster, keine vollständige Aufzeichnung.
- Das **Live-Gehirn** zeigt vorhandene Modellwerte in einer anschaulichen Form.
  Farben entsprechen Circuit: Blau/Input, Violett/Expansion, Mint/Concepts,
  Bernstein/Memory. Hellere Punkte markieren Ausgänge; die ausgewählte Einheit
  ist zusätzlich vergrößert, ohne dadurch Aktivität zu erhalten. Atlas-Beschriftungen
  benennen die anatomische Vorlage, nicht die Lokalisation der Modellfunktionen; ohne exportierte
  Gewichte werden keine echten Verbindungen behauptet. Ohne Daten bleibt es leer.
- **Follow incoming samples / Freeze view** steuert nur den betrachteten Zeitpunkt.
  Der Schieberegler untersucht ältere aufbewahrte Schritte. Lernen und Erfassung
  können unabhängig davon weiterlaufen.
- Im **Captured unit inspector** wählst du Modellregion und Einheit. Membrane ist
  ein Modellwert nach dem Schritt; Expansion besitzt kein solches Membranpotential.
  DA, NE, ACh und 5-HT sind interne Modellvariablen, keine gemessenen menschlichen
  Neurotransmitter oder Gefühle.
- **Only what was observed** zeigt Eingaben am ausgewählten historischen Tick,
  nicht automatisch den aktuellen Erfassungszustand. Fehlend ist nicht null.
- **Export observed window** speichert das begrenzte beobachtete Fenster als JSON,
  nicht das ganze Experiment oder alle Modellgewichte. Der Dienst hält bis zu
  512 Schritte im RAM, die Ansicht bis zu 200. Das persistente Gehirn speichert
  seinen Checkpoint selbst; der Runner speichert nichts, und bei ihm sind Mikrofon
  und Wearables nicht angeschlossen.

### Observatory: die synthetische Übersicht

Die Seiten außerhalb von Live session sind die Forschungsdemo. **Example context**
wechselt zwischen vier vorgegebenen Situationen. „Flow“ und ähnliche Wörter sind
Beispiel- bzw. Nutzerlabels, keine psychologischen Feststellungen. **Similarity**
ist ein Beispielwert, keine kalibrierte Sicherheit. **Confirm label** und **Teach
a word** erzeugen hier nur lokale Demo-Annotationen, kein Training des Live-Modells.

Die Modulatorkarten erklären die vorgesehenen Rollen der Modellvariablen. Die
Input-context-Zeilen sind ebenfalls synthetisch; sie zeigen nicht, dass sechs
Sensoren gerade aktiv sind. Die Annotation trail führt zum Learning journal.
Vorgeschlagene Desktop-Hilfe ist eine Vorschau, keine tatsächliche OS-Aktion.

### Neural explorer: Modellzusammenhänge untersuchen

Das 3D-Gehirn lässt sich drehen und zoomen. Ein Punkt wählt eine Einheit. **Circuit**
zeigt dieselben Demo-Daten schematisch. **Contours** betrifft die Form;
**Atlas labels** zeigt schaltbare anatomische Orientierungspunkte an den
Quellstrukturen, einschließlich der linken/rechten Hemisphäre.
**Connections** die illustrierten Beispielverbindungen. Die vier Regionen bedeuten
Input/Sensory (200), Expansion (500), Concepts (200), Working memory (100).
Das sind Rechenrollen, keine anatomische Lokalisation dieser Funktionen.

**In plain language** erklärt den Kontext, **Single neuron** einzelne Werte und
Nachbarn. Das Suchformular ist eine Alternative zum Anklicken winziger Punkte.
**Replay** und der Frame-Regler bewegen sich durch 200 synthetische Samples in
einem Zwei-Sekunden-Fenster. Das Raster darunter zeigt 40 Konzept-Einheiten.
Replay ist keine Erfassung. Vorwärts wandernde Lichter sind kein Kausalitätsbeweis.

### Vocabulary: Worten eine nachvollziehbare Referenz geben

Vier Beispiellabels dienen als Startpunkt. Suche und Karten zeigen Herkunft und
Referenzsignatur. **Teach a new word** hält den ausgewählten Demo-Moment fest und
öffnet das Formular. Eigene Annotationen bleiben im Tab-Speicher; ein Neuladen
verliert sie, wenn sie nicht zuvor exportiert wurden. Der Rundgang fügt keine hinzu.

### Learning journal: den richtigen Moment wiederfinden

Beispielhistorie und eigene Demo-Korrekturen sind gekennzeichnet. „Your corrections“
enthält auch Beispielkorrekturen; „SYNTHETIC“ und „LOCAL ANNOTATION“ unterscheiden
die Herkunft. Filter grenzen die Einträge ein. Ein Eintrag zeigt den Moment; Wiedergeben öffnet
den gespeicherten Demo-Kontext im Explorer. Es ist keine lückenlose Live-Chronik.

### Methods & provenance: Belege und Grenzen

Diese Seite dokumentiert die synthetische Demo: Generator, Seed, Referenzrevision,
Architektur, Grenzen und Asset-Lizenzen. Sie ist keine aktuelle Live-Konfiguration
und kein Wirksamkeitsnachweis. **Copy view** kopiert die Demo-Auswahl als lokale URL,
ohne eigene Labels/Notizen und ohne Kameraposition. **Export** enthält dagegen
Demo-Annotationen; vor einer Weitergabe den Inhalt prüfen. Live hat seinen eigenen
Export im Bereich Live session. Der Rundgang startet keine Exporte.

## Gestaltung und Sicherheit

- Observatory-Farben, Schrift und bestehende 3D-Darstellung bleiben erhalten.
  Aus den UI-/Motion-Skills werden flache, klare Karten, große Klickflächen,
  sichtbarer Fokus und reduzierte Bewegung übernommen.
- Die Begrüßung ist ein modaler Dialog. Die laufende Tour ist ausdrücklich nicht
  modal: Das markierte Originalelement bleibt bedienbar. Keine falsche
  `aria-modal`-Kennzeichnung für weiterhin interaktive Seiteninhalte.
- Escape beendet die Tour; bei einem darüber geöffneten Erklärdialog schließt
  Escape zuerst diesen. Markierungen, Fokus-Verweise und Layout-Zusätze werden
  beim Beenden entfernt. Der vorherige Bereich wird wiederhergestellt.
- Der Browser speichert, dass die Einführung angeboten wurde, und die ausdrücklich
  gewählte Tour-Sprache. Kein Nutzerprofil,
  keine Annotation und kein Capture-Zustand wird dadurch gespeichert oder verändert.
- Keine automatische Verbindung, Sensorfreigabe, Annotation, Download oder OS-Aktion
  durch Tour-Start, Weiter, Zurück oder Kapitelwechsel.

Referenzen für das Interaktionsmodell:
[WAI-ARIA: Dialog und Modalität](https://www.w3.org/WAI/ARIA/apg/patterns/dialog-modal/),
[MDN: dialog](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/dialog).

## Abnahme

Quellwahl geprüft am 26. September 2026: Die Texte für Verbindung, Datenschutz
und Export beschreiben beide Modelle; der Datenschutz-Schritt markiert beide
Schalterbereiche (`.live-privacy`). 48 automatisierte Tests bestanden. Im Browser
ohne laufenden Daemon die Schritte 1–7 auf Englisch und Deutsch bei 1360, 390 und
320 px Breite: Ziel jeweils vorhanden und sichtbar, kein horizontaler Überlauf,
keine Seitenfehler und keine Anfrage außer GET. Die Tour startet also weder den
Daemon noch eine Erfassung.

Spracherweiterung geprüft am 15. September 2026, Branch `codex/english-onboarding`:

- **37 automatisierte Tests bestanden:** 15 Tour-Tests und 22 bestehende Dashboard-
  Tests. Beide Sprachen enthalten alle 18 Schritte einschließlich Übungen,
  Rückmeldungen, Warnungen und Bedienlabels. Gesperrter Speicher und Sprachwechsel
  ohne Fortschrittsverlust sind abgedeckt.
- **Browser:** englische Begrüßung auf Desktop und bei 320 × 568 geprüft;
  Deutschwahl über Neuladen erhalten. Sprachwechsel während der Tour behält den
  Schritt. Alle 18 englischen Karten bei 390 × 844 und alle 18 deutschen Karten
  bei 320 × 568 ohne horizontalen Kartenüberlauf durchlaufen. Bei Platzmangel
  bleibt der Karteninhalt intern scrollbar. Viewport anschließend zurückgesetzt.
- **Bestehendes Design beibehalten:** Sprachwahl mit 44-px-Mindesthöhe und
  sichtbarem Tastaturfokus gemäß Dashboard-Skill ergänzt.
- Keine Sensorfreigaben geändert, keine Verbindung oder Exporte ausgelöst;
  Erfassungsrevision weiterhin 4. Browserkonsole ohne Warnungen oder Fehler.

Historische Abnahme der ursprünglichen deutschen Einführung:

Geprüft am 14. September 2026:

- **33 automatisierte Tests bestanden:** 11 neue Tour-Tests sowie 22 bestehende
  Prüfungen für Demo, Live-Daten und Erfassungssteuerung. Darunter der bestehende
  End-to-End-Vertrag von tatsächlichem Python-Modelloutput zur Browser-Datenstruktur.
- **Erstbesuch und Rückkehr:** Begrüßung beim ersten Laden angeboten; nach erneutem
  Laden kein automatisches Öffnen. Der Einführung-Button öffnet sie weiterhin.
  Gesperrter Browserspeicher ist separat im Controller-Test abgedeckt.
- **Alle 18 Schritte im Browser durchlaufen:** Ziel vorhanden und markiert,
  Schrittkarte im sichtbaren Bereich, sichtbare Markierung nicht von der Karte
  überdeckt. „Fertig“ stellt den Ausgangsbereich wieder her und entfernt Markierungen.
- **Freiwillige Übungen:** Kontextwahl, Regionsfilter, Frame-Regler, Vocabulary-Karte
  und Journalfilter ausprobiert. Die Rückmeldung erscheint ohne automatisches
  Weiterschalten. Der Regionsfilter wurde auch mit einem echten Zeigerklick geprüft.
- **Tastatur und Ausstieg:** Escape, direkte Kapitelwahl, normaler Seitenwechsel
  und Rückkehr zum Einführung-Button geprüft. Ein zusätzlich geöffneter
  Erklärdialog erhält Escape zuerst; die Tour bleibt dabei erhalten.
- **Layout:** Browserprüfung bei 320 × 568, 390 × 844, 677 × 987,
  844 × 390 und 1440 × 900. Begrüßung und Schrittkarte scrollen bei Platzmangel
  innen; Start, Weiter, Zurück und Beenden bleiben erreichbar. Kein horizontaler
  Seitenüberlauf in den geprüften schmalen Ansichten. Viewport danach zurückgesetzt.
- **Datenschutzgrenze:** Kein Verbindungs-, Freigabe-, Annotations- oder Exportklick
  durch die Tour. Die vor dem Rundgang bestehende Erfassungsrevision 4 blieb
  unverändert. „Nicht verbunden“ wurde im Browser geprüft, ohne den laufenden
  Beobachtungsdienst oder andere Tabs zu stoppen.
- **Browserkonsole:** keine Warnungen oder Fehler im abschließenden Check.

Reproduzierbarer Testaufruf vom Repository-Stamm:

```sh
rtk proxy node --test docs/dashboard-concepts/verify-onboarding.mjs docs/dashboard-concepts/verify-observatory.mjs docs/dashboard-concepts/verify-live.mjs docs/dashboard-concepts/verify-capture.mjs docs/dashboard-concepts/verify-daemon.mjs
```

Die Controller-Tests nutzen eine schlanke DOM-Testumgebung; sie ersetzen die
oben separat durchgeführte Browserprüfung nicht. Dies ist keine vollständige
Screenreader-, Hardware-Touch- oder WCAG-Zertifizierung. Die laufende Tour bleibt
nicht-modal: Eigene Klicks auf echte Erfassungsschalter sind weiterhin echte
Entscheidungen des Nutzers, keine simulierten Übungen.
