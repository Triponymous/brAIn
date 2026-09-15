# brAIn: echte Modelldaten und Wearables

Stand: 14. September 2026. Lokale Implementierung und kritische Produktentscheidung; kein Deployment.

## Entscheidung

**Ja zu zusätzlichen Körperdaten als optionalem Kontext. Nein zu der Annahme, dass mehr Sensoren automatisch besseres Lernen oder zuverlässige Emotionserkennung ergeben.**

Der erste Ausbau ist eine isolierte Live-Ansicht mit lesenden Modelldaten und eng begrenzten Erfassungsschaltern. Sie trennt tatsächliche Modelldynamik, beobachtete Desktop-Metadaten und die bestehende synthetische Designstudie. Wearable-Anbindungen sind hier **nicht implementiert**; Accounts, Bluetooth, HealthKit und Gesundheitsdaten wurden nicht geöffnet.

## Was jetzt implementiert ist

[Live session](http://127.0.0.1:4178/observatory.html#live) ergänzt das bestehende Dashboard.

- Der neue Runner verwendet wirklich `brain.core.Brain`: echte diskrete Ausgaben, echte Post-Step-Membranen, tatsächliche Modellmodulatoren. Der aktuelle Core implementiert seine LIF-Schritte selbst mit PyTorch; die Bibliotheksabhängigkeit snnTorch allein macht einen Lauf nicht zum snnTorch-Lauf.
- Jeder Frame wird unmittelbar nach einem vollständigen Modellschritt gemeinsam kopiert. Spikes werden nicht aus niedrigen Membranwerten geraten; `concept_spike_accum` wird nicht als Membran ausgegeben.
- `Brain.tick()` liefert zusätzlich die vorhandenen binären Expansion-Ausgaben zurück; die Lernregeln sind unverändert. Expansion hat kein Membranpotenzial.
- Frische, separat initialisierte Instanz mit Seed; keine bestehenden persönlichen Checkpoints laden, verändern oder überschreiben.
- Opt-in-Collector: macOS-Ereigniszähler, Leerlauf und grobe Vordergrund-App-Kategorie. Keine Key-Event-Hooks, Tasteninhalte, Mauskoordinaten, Fenstertitel, Hintergrund-App-Liste, Screenshots oder Audiodaten. App-Namen werden nur kurz zur Kategorisierung abgefragt, nicht exportiert oder gespeichert.
- **Data & privacy:** Vier unabhängige, serverbestätigte Schalter und „Stop all capture“. Ausgeschaltete Quellen werden nicht abgefragt; alle aus stoppt auch Modellschritte. Jede neue Serversitzung startet mit allen Quellen aus. Ein Seitenneuladen behält die Auswahl der laufenden Serversitzung, ohne sie im Browser zu speichern.
- Keine vorgetäuschten Rhythmusmerkmale: Zähler-Polling misst keine Inter-Key-Intervalle. Entsprechende Encoder-Slots bleiben maskiert. Fehlende Eingänge werden nicht als Stille oder Ruhe kodiert.
- Tastatur-/Maus-/Idle-Zugriff nur bei erfolgreicher macOS-Preflight-Prüfung. Keine automatischen Berechtigungsdialoge. Fehlende Berechtigung ist ein eigener Zustand. Eine verfügbare App-Kategorie allein kann dennoch den Core antreiben.
- 512 Frames im Prozess-Ringpuffer; maximal 200 pro Antwort und 200 im sichtbaren Fenster. Session-Wechsel und übersprungene Historie sind ausdrücklich markiert.
- Echte verstrichene Zeit plus UTC-Erfassungszeit. Zielrate 100 Modellschritte/s, tatsächliche Rate abhängig von Rechenzeit. Der Integrationsparameter `dt=1.0` ist **nicht** automatisch eine Sekunde oder ein biologischer Echtzeitmaßstab.
- Das bereits freigegebene 3D-Modell nutzt die tatsächlichen binären Ausgaben. Anatomische Positionen bleiben illustrativ. Alle Beispielkanten sind im Live-Modus ausgeblendet; keine behaupteten gemessenen Faserbahnen oder Gewichte.
- Selektieren, einfrieren, zeitlich inspizieren und ein privates JSON-Fenster exportieren. Ein eingefrorener Snapshot bleibt auch nach Verlassen des Ringfensters erhalten; der Chart behauptet dann keine nicht mehr vorhandene Historie.
- Es gibt keine Live-Emotionserkennung, keine trainierte Wortzuordnung, keine OS-Aktionen, keine LLM-Weitergabe und keine Wearable-Werte. Die bisherigen fünf Ansichten bleiben explizite Demo-Ansichten.

## Lokal starten

Im Repository `/Users/leonmatthies/brAIntest`:

```sh
# Verbindungsprüfung ohne Sensoren und ohne Modellschritte
rtk proxy .venv/bin/python -m server.observe

# Schalter verfügbar machen; alle Eingänge bleiben aus bis zur Auswahl im Dashboard
rtk proxy .venv/bin/python -m server.observe --desktop-metadata

# Optional: zeitlich begrenzter Versuch, automatisches Ende nach 120 Sekunden
rtk proxy .venv/bin/python -m server.observe --desktop-metadata --duration 120
```

Danach in **Data & privacy** nur die gewünschten Quellen einschalten und auf **Connect local model** klicken. Öffnen und Verbinden allein aktivieren keine Sensoren. Ohne `--desktop-metadata` kann das Dashboard keine Quelle einschalten. **Follow incoming samples** aktualisiert den untersuchten Tick; **Freeze view** stoppt nur die Darstellung. **Disconnect view** trennt den Modelldatenstrom, nicht die Erfassung. Die Erfassungsschalter bleiben separat bedienbar, auch nach dem Trennen der Ansicht. **Stop all capture** stoppt Abfragen und weitere Modellschritte. **Ctrl-C** beendet den gesamten Runner.

Ein Aus-Schalter ist keine Löschfunktion: bereits gelernte Gewichte, historische Frames und ein eventuell eingefrorener Snapshot bleiben im Arbeitsspeicher bis zum Ende des jeweiligen Prozesses beziehungsweise der Ansicht. Bereits exportierte Dateien bleiben beim Nutzer. Es wird kein Checkpoint gespeichert. Die OS-Eingabeüberwachungsberechtigung bleibt unverändert; diese Schalter kontrollieren weder andere Anwendungen noch den alten `server.braind`-Daemon.

Der minimale Runner ist absichtlich nicht `server.braind`: dessen normaler Start aktiviert zusätzliche Sensoren und Dienste. Der Beobachtungs-Runner ist kein Ersatz für diese gesamte Companion-Anwendung.

## Sicherheit und Reproduzierbarkeit

Der Dienst bindet nur an `127.0.0.1:8001`. Modelldaten bleiben GET-only. Die neue API `GET/POST /api/capture` liest/ändert ausschließlich die vier Erfassungsschalter. Änderungen verlangen eine zugelassene Observatory-Origin an Port 4178, JSON und den nicht geheimen Header `X-Brain-Control: capture-v1`. Keine Form-POSTs, keine fehlende/fremde Origin, keine unbekannten Felder; 2-KB-Bodylimit und strikte boolesche Werte. Zugriff aus nicht lokalen Verbindungen und unbekannte Hostnamen wird abgewiesen. Keine Wildcard-CORS, kein Cache, keine Access-Logs mit Nutzdaten.

Session-ID und Auswahlrevision verhindern, dass ein alter Browser eine neue Sitzung oder einen neueren Stop durch nachträgliches Einschalten überschreibt. Ausschalten ist auch mit einer älteren Revision derselben Sitzung möglich und entwertet ausstehende Einschaltbefehle. Sensorabfrage, Modellschritt und Änderung sind synchronisiert: die Stop-Antwort wird erst nach Abschluss eines bereits laufenden Schritts bestätigt, gecachte Eingangsvektoren werden verworfen. Beim Wiedereinschalten wird die Zählerbasis neu aufgenommen, damit Ereignisse aus der ausgeschalteten Zeit nicht nachträglich einfließen. Ein neuer Frame trägt die zugehörige `capture_revision`; alte Frames werden nicht umgeschrieben.

Origin, Kontrollheader und Session-ID sind keine Authentifizierung gegenüber anderen lokalen Prozessen. Dies schützt nicht gegen Schadsoftware oder andere berechtigte Prozesse auf demselben Rechner. Vor einer Verbindung vom iPhone oder einem zweiten Gerät braucht es bewusstes Pairing, Authentifizierung und geschützten Transport — **nicht** einfach `0.0.0.0` und `*` als Freigabe.

Export enthält Session-ID, Seed, Quellcode-Fingerprint, Input-Herkunft, tatsächliche Architektur, ausgewählte Einheit, Cursor-Snapshot und das erhaltene Frame-Fenster. Der SHA-256 beschreibt den erfassten lokalen Codebestand; er ist keine Signatur und kein Echtheitsbeweis. Der Export ist **kein vollständiges Experiment**, kein Checkpoint und kein wissenschaftlicher Leistungsnachweis. Er kann persönliche Aktivitätsmetadaten enthalten: nicht automatisch ins Repository, öffentliche Screenshots oder Cloud-LLM geben.

Keine automatische Speicherung auf Disk. Für eine echte Langzeitstudie fehlen noch ein zustimmungsgebundener Recorder, versionierte Labels, Session-Wiederaufnahme, Löschpfade und Evaluationspipeline. Ein neuer Runner-Start beginnt bewusst wieder mit einem frischen Modell.

## Wearables: was die Plattformen tatsächlich ermöglichen

| Quelle | Realistischer Zugang | Wichtige Grenze |
|---|---|---|
| WHOOP Cloud | OAuth 2.0, API v2: Schlaf, Recovery, Ruhepuls, HRV-RMSSD, Hauttemperatur; unbewertete/fehlende Scores berücksichtigen | Keine kontinuierliche Herzfrequenz in der öffentlichen API. Tages-/Zykluswerte sind keine Live-Messung. |
| WHOOP BLE | Herzfrequenz-Broadcast, wenn der Nutzer ihn einschaltet und das Gerät verbindet | Separater Datenweg zur Cloud; nicht automatisch Hauttemperatur, Roh-PPG oder beat-to-beat RR-Daten. |
| Apple Watch / HealthKit | Kleine native Begleit-App mit gezielter Lesefreigabe; anschließend sichere Übergabe an den Mac | Kein nutzbarer HealthKit-Datenspeicher in einer gewöhnlichen macOS-App. Hintergrund-Pulsmessungen sind nicht kontinuierlich. |
| Andere Wearables | Pro Anbieter Cloud-API oder dokumentierten BLE-Dienst prüfen | Eine Uhr kann etwas anzeigen, ohne es Drittentwicklern bereitzustellen. Kein universeller Smartwatch-Connector versprochen. |

WHOOP beschreibt ausdrücklich den Unterschied zwischen BLE-Puls und fehlendem kontinuierlichem Puls in der API. Die aktuelle API liefert unter anderem `hrv_rmssd_milli` und `skin_temp_celsius`; Recovery kann noch unbewertet oder nicht auswertbar sein. [WHOOP FAQ](https://developer.whoop.com/docs/developing/support/), [API v2](https://developer.whoop.com/api/), [Recovery-Datenmodell](https://developer.whoop.com/docs/developing/user-data/recovery/).

Apple HealthKit nutzt bei HRV **SDNN**, nicht WHOOPs RMSSD. Die Metriktypen, Messfenster und Gerätebaselines müssen getrennt bleiben. Ein Feld `hrv` für beide wäre fachlich falsch. [Apple HRV-Dokumentation](https://developer.apple.com/documentation/healthkit/hkquantitytypeidentifier/heartratevariabilitysdnn).

HealthKit-Daten sind nicht direkt auf macOS lesbar. Hintergrund-Pulsintervalle variieren nach Aktivität; kontinuierliche Workout-Erfassung ist kein Freibrief, einen ganzen Arbeitstag als künstliches Training zu deklarieren. [HealthKit-Verfügbarkeit](https://developer.apple.com/documentation/healthkit/hkhealthstore/ishealthdataavailable()), [Apple Pulsmessungen](https://support.apple.com/en-au/120277).

Apples Handgelenktemperatur ist eine nächtliche Messung relativ zu einer persönlichen Basis, kein Thermometer auf Abruf. Sie ist als Tageskontext interessanter als als unmittelbares Signal für eine Desktop-Unterbrechung. [Apple Temperaturmessung](https://support.apple.com/en-us/102674).

## Warum ich kritisch bin

1. **Mehrdeutigkeit:** Ein Pulsanstieg liefert keine eindeutige psychologische Ursache. Für das Produkt müssen körperliche Aktivität, individueller Grundzustand und die Antwort des Nutzers mit berücksichtigt werden. Kein fixer Puls-Schwellenwert als „Stress“.
2. **Datenmenge ist nicht Informationsgewinn:** Korrelierte oder verrauschte Signale können das Modell komplizierter machen, ohne Entscheidungen zu verbessern. Im ursprünglichen WESAD-Versuch brachte das zusätzliche Handgelenkgerät gegenüber dem stärkeren Brustgeräte-Setup keine weitere Verbesserung. Das ist kein Beweis gegen Wearables, aber ein konkretes Gegenbeispiel zu „mehr ist immer besser“. [WESAD, Originalpublikation 2018](https://www.eti.uni-siegen.de/ubicomp/papers/ubi_icmi2018.pdf).
3. **Labordaten sind kein Desktop-Alltag:** WESAD enthält 15 Personen in einer kontrollierten Studie. Die dortigen Erkennungszahlen sind keine belastbaren Erwartungen für deine Uhr, deinen Arbeitsalltag oder brAIn. Selbstberichte und ein späterer, zeitlich getrennter Test sind notwendig.
4. **Zeitliche Auflösung:** Ein nächtlicher Recovery-Wert darf nicht hundertmal pro Sekunde als neue Beobachtung erscheinen. Sensorzeit, Empfangszeit, Messfenster, Alter, Quelle und Gültigkeit müssen getrennt sein. Kein Auffüllen fehlender Werte mit Null; keine Interpolation, die neue Messungen vortäuscht.
5. **Kreisschlüsse:** Einen proprietären Recovery-/Stress-Score als Input verwenden und anschließend vorhersagen zu lassen, dass der Nutzer „Stress“ hat, belegt keine unabhängige Fähigkeit. Solche Scores sind Hersteller-Schätzungen, keine Ground Truth.
6. **Berechtigungen und Vertrauen:** Lokale Verarbeitung ist sinnvoll, aber kein automatischer Datenschutzbeweis. Auch gelernte Gewichte und Aktivitätsmuster können sensible Informationen tragen. Kein Verkauf persönlicher Gesundheitsdaten als vermeintlicher Produktwert.
7. **Apple-Produktregeln:** HealthKit ist kein allgemeiner Datensammelkanal für beliebige KI. Die App-Review-Regeln beschränken gesundheitsbezogenes Data Mining und verlangen bei gesundheitsbezogener Forschung mit Menschen Einwilligung sowie unabhängige Ethikprüfung. Meine Schlussfolgerung: Zweck und Verteilung der Begleit-App vor deren Bau prüfen; eine bloße Bezeichnung als „Forschungsprojekt“ löst diese Anforderungen nicht. [Apple 5.1.2 / 5.1.3](https://developer.apple.com/app-store/review/guidelines/#health-and-health-research).

## Nächster sinnvoller Versuch

**Fragestellung:** Helfen zusätzliche Wearable-Signale dabei, einen vom Nutzer akzeptierten Zeitpunkt für eine nicht dringende Unterbrechung zu erkennen?

Zunächst rein beobachtend, ohne echte Benachrichtigungen zu blockieren. Ein kurzer freiwilliger Selbstbericht („gerade unterbrechbar?“) liefert ein nutzerbezogenes Ziel; Pulswerte allein nicht.

Vergleichen:

1. Einfache Baseline aus Desktop-Metadaten.
2. Gleiche Baseline mit Wearable-Merkmalen.
3. brAIn nur mit Desktop-Metadaten.
4. brAIn mit denselben zusätzlichen Wearable-Merkmalen.

Damit trennen wir den **Nutzen zusätzlicher Sensoren** vom **Nutzen der SNN-Architektur**. Zeitlich getrennte Anpassungs- und Testtage; keine zufälligen überlappenden Zeitfenster in beiden Splits. Beim späteren Test mit mehreren Personen auch Trennung nach Personen. Persönliche Normalisierung nur aus der bereits vergangenen Kalibrierungszeit berechnen.

Messen: falsche „guter Zeitpunkt“-Vorschläge, Trefferquote bei gleicher Vorschlagszahl, Abdeckung durch tatsächlich verfügbare Daten, Synchronisationslücken, CPU-/Akkukosten und subjektiver Nutzen. Vor dem Test festlegen, welche Verbesserung den zusätzlichen Aufwand rechtfertigt. Kleine Pilotdaten zeigen Machbarkeit, keine allgemeine Validierung.

**Input-Priorität:** Puls zusammen mit Bewegungs-/Ruhekontext; Schlaf/HRV als langsam veränderlicher optionaler Kontext; Temperatur zunächst nur explorativ. Welche Verbindung zuerst gebaut wird, hängt von der bereits vorhandenen Uhr und ihren offiziell zugänglichen Daten ab. Noch keine neue Uhr allein für diese Hypothese kaufen.

## Implementierungs- und Prüfpfade

- `server/observe.py`: bewusst kleiner opt-in Research-Runner.
- `adapters/mac_desktop/metadata.py`: begrenzte Metadatenerfassung plus Maskierung nicht gemessener Features.
- `server/telemetry.py`: kohärente Frames, Ringpuffer, Provenienz und lokaler GET-Endpunkt.
- `server/capture.py`: Sitzungsrichtlinie, bestätigtes Abschalten und begrenzte Kontroll-API.
- `capture-data.js`, `capture-controls.js`, `capture-controls.css`: validierter Schaltervertrag, bestätigte Zustände, getrennte aktuelle Auswahl und historische Daten, unveränderte Observatory-Tokens.
- `live-data.js`: validierter Datenvertrag, begrenzter Verlauf und unveränderlicher Export.
- `live-workspace.js`: Verbindung, Freeze, echte Kurve, Inspector und 3D-Datenzufuhr.
- `tests/test_observation_telemetry.py`: Wertegleichheit, Isolation, Ringpuffer, Missingness, Berechtigungsgrenze und HTTP-Grenzen.
- `verify-live.mjs`: externen JSON-Vertrag, Reihenfolge, Quellwechsel, Verlauf und Export prüfen.

```sh
rtk proxy .venv/bin/python -m pytest tests/test_capture_controls.py tests/test_observation_telemetry.py tests/test_core.py tests/test_server.py -q
rtk proxy node --test docs/dashboard-concepts/verify-capture.mjs docs/dashboard-concepts/verify-live.mjs docs/dashboard-concepts/verify-observatory.mjs
rtk proxy node 3d/verify-brain.mjs
```

Repo-Compass bestätigte `python-ai-experimental`; Graphify verfolgte den bisherigen Daemon-/Tick-/Push-Pfad. Der Dashboard-Skill beeinflusste flache Controls und klare Zustände, nicht eine Änderung der freigegebenen Observatory-Palette. Der 3D-Skill führte zum Erhalt von Anatomie, Kamera und einem gemeinsamen Canvas. Keine neue Look-Dev-Richtung und keine behauptete vollständige Studiogate-Abnahme.

## Aktueller Prüfstand

- **22 Python-Tests bestanden:** Telemetrie, Core und bestehende Server-Grundfunktionen.
- **14 JavaScript-/Integrationstests bestanden:** sieben vorhandene Demo-Tests plus sieben neue Live-Tests. Einer führt den echten Python-Core aus und prüft seine Ausgabe mit demselben JavaScript-Vertrag wie der Browser.
- **3D-Verifikation PASS:** Geometrie und Netzwerk-Katalog bleiben erhalten. Live zeigt keine dieser Beispielkanten.
- Browser: Verbindung mit ausgeschalteten Sensoren → keine Frames und keine erfundenen Werte. Separater echter Core-Lauf mit ausdrücklich als `test_fixture` gekennzeichneten Eingaben → tatsächliche Modelldaten im Inspector und 3D.
- Konkrete Stichprobe: Tick 15.920 und Einheit E-499 in beiden Ansichten identisch; binärer Output 1, kein Membranpotenzial. Freeze-/Demo-/Live-Rundlauf hielt Tick 21.809 und genau ein Canvas. Der spätere Export wurde im UI angefordert; JSON-Inhalt und Unveränderlichkeit separat im Test verifiziert.
- Verbindungsausfall: letzter Snapshot bleibt historisch sichtbar, Follow wird deaktiviert. Neustart mit neuer Session und deaktivierten Sensoren leert die alte Anzeige vollständig. Keine synthetische Ersatzquelle.
- Layout bei 1.440, 390 und 320 px geprüft; kein horizontaler Seiten-Overflow in den mobilen Live-Ansichten. Viewport anschließend zurückgesetzt.
- Browser-Konsole im verbundenen Testlauf leer. Erwartete Netzwerkfehler beim absichtlichen Abschalten sind keine erfolgreichen Live-Updates.

### Aktivierung nach Nutzerfreigabe · 14. September 2026

- Der Nutzer hat die lokale Erfassung von Tastatur-/Mausanzahl, Leerlauf und grober App-Kategorie freigegeben. Der zuvor sensorlose Runner wurde beendet und mit `--desktop-metadata` neu gestartet.
- Echte App-Kategorie als verfügbarer Eingang; tatsächliche Core-Schritte im Dashboard. Follow ist aktiv, 3D und Inspector zeigen denselben Tick; genau ein Canvas und keine Browser-Konsolenfehler bei der Prüfung.
- macOS verweigert noch die Eingabeüberwachung: Tastatur, Maus und Leerlauf bleiben `permission_required`, nicht Null. Die Nutzerfreigabe im Chat ersetzt keine Systemberechtigung; es wurde keine Berechtigung umgangen oder geändert.
- Mikrofon und Cloud bleiben ausgeschaltet; Wearables sind nicht verbunden. Verarbeitung und begrenzter Verlauf bleiben lokal im Arbeitsspeicher. Keine einzelnen Aktivitätswerte werden hier protokolliert.
- Der Beobachter läuft nur für diese lokale Sitzung weiter. Stoppen im Runner-Terminal mit `Ctrl-C`; „Disconnect view“ trennt ausschließlich das Dashboard. Keine Installation als Autostart-Dienst, kein Checkpoint und keine Datenspeicherung auf Platte.

**Noch nicht verifiziert:** freigegebene macOS-Tastatur-/Maus-/Leerlaufwerte, tatsächliche Geräte-/Wearable-Synchronisation, Langzeitstabilität, biologische Interpretation, personenbezogener Nutzen, Screenreader-End-to-End und reale Mobil-GPU-Leistung. Die App-Kategorie-Anbindung belegt Datenfluss, nicht die Aussagekraft des Modells.

### Aktueller Stand: kanalweise Datenschalter · 14. September 2026

Der vorige, rein flüchtige Lauf wurde für das Update beendet; kein Checkpoint existierte oder wurde überschrieben. Der aktualisierte Runner läuft mit verfügbaren Schaltern, **alle vier Eingänge sind am Ende der Prüfung ausgeschaltet**. Historische Modellschritte aus dem kurzen Browser-Test bleiben als historische Daten sichtbar; keine laufende Erfassung wird vorgetäuscht.

- **35 Python-Tests bestanden:** davon 13 neue Capture-Tests. Prüfen reale API-Aufrufe mit instrumentierten Fakes, getrennte Kanäle, fehlende OS-Rechte, neue Zählerbasis nach Wiedereinschalten, verworfene Cache-Vektoren, keine weiteren echten Core-Schritte bei Aus, synchronisierten Stop, alte Sitzungen/Revisionen sowie HTTP-/Origin-/Body-Grenzen.
- **22 JavaScript-/Integrationstests bestanden:** davon acht neue Kontrolltests. Kein automatisches Aktivieren beim Öffnen, keine optimistisch bestätigte Abschaltung, verständlicher unbestätigter Stop, Konflikte, echte Zustände bei OS-Verweigerung, neue Sitzung aus, historische Auswahlrevisionen und der bestehende Python-zu-Browser-Datenvertrag.
- **3D-Prüfung PASS:** bestehende Geometrie, Katalog und Renderer unverändert.
- Browser: App-Kategorie ausdrücklich kurz aktiviert → echte Core-Frames und identischer Tick in 3D/Inspector. Tastatur, Maus und Idle ausgewählt → ehrlich `permission_required`, keine OS-Berechtigung geändert. App-Kategorie einzeln ausgeschaltet; anschließend Gesamtstopp erfolgreich, obwohl die Modellansicht bereits getrennt war.
- Seitenneuladen behält alle Schalter aus. Erneutes Verbinden zeigt `Capture paused`, historische Frames und zunehmendes Alter ohne neue Schritte. Kein weiterer Legacy-Dienst lauschte bei der gezielten Prüfung auf Port 8000.
- Tastaturbedienung mit Leertaste geprüft: Fokus bleibt am Schalter, sichtbare 2-px-Fokuslinie. Schalter mindestens 52 × 44 px; Layout bei 1440, 677, 390 und 320 px geprüft, kein horizontaler Überlauf in den geprüften schmalen Ansichten. Temporäre Viewport-Größen zurückgesetzt. Browser-Konsole nach dem Neuladen ohne Warnungen/Fehler.

Der Dashboard-Skill beeinflusste flache, konsistente Schalter und Tastatur-/Touch-Bedienbarkeit; Farben, Schriftbild und die freigegebene 3D-Szene bleiben erhalten. Der vorhandene Graphify-Servergraph half bei der Abgrenzung zum alten Daemon; die neue Erfassungssteuerung wurde direkt am aktuellen Code geprüft. Keine Memory-Dateien, kein Commit/Push, keine Cloud- oder Gesundheitsdatenanbindung.
