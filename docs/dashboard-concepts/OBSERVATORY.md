# Observatory — ein Forschungsinstrument, das verständlich bleibt

brAIn: Living Desktop Manager · Research workspace 03 · 14. September 2026

**Designstand der fünf Demo-Ansichten:** [Vorschau](http://127.0.0.1:4178/observatory.html#overview) · [Ausarbeitung, Grenzen und Prüfbelege](RESEARCH-WORKSPACE.md) · [Ursprüngliche drei Varianten](index.html).
Aktuell kommen die getrennte [Live session](LIVE-DATA.md) und die
[zweisprachige Einführung](EINFUEHRUNG.md) hinzu. Einstieg: [Dokumentation](../README.md).

## Die gewählte Richtung

Observatory bleibt die visuelle Grundlage. Die verständliche Erklärungsebene aus Symbiosis verbindet Kontext, Modellzustand und eigene Annotationen. Das bestätigte echte 3D-Gehirn bleibt das visuelle Zentrum.

Akademische Sorgfalt heißt hier: Quellen zeigen, Einheiten sauber benennen, Datenherkunft offenlegen und keine falsche Gewissheit erzeugen. Hochschulzugehörigkeit, Forschungsergebnisse oder Kaufinteresse werden nicht behauptet.

## Fünf zusammenhängende Ansichten

- **Observatory:** Beispielkontext, benannter oder unbekannter Zustand, erklärte Ähnlichkeit, Netzwerk, Modulatoren und sechs Eingangsbeispiele.
- **Neural explorer:** Brain/Circuit, Auswahl aus 1.000 Einheiten, Nachbarn, synchroner Inspector, 200-Sample-Replay und Spike-Raster.
- **Vocabulary:** durchsuchbare Referenzen mit Herkunft; lokale und synthetische Annotationen getrennt.
- **Learning journal:** genaue Kontext-/Frame-Snapshots, Notizen und Referenzrevisionen mit Rücksprung zum ursprünglichen Moment.
- **Methods & provenance:** Quellrevision, Generator, Seeds, Zeitbasis, Architektur, Einschränkungen und Attribution.

## Die wesentlichen Interaktionen

1. Beispielkontext wählen, 3D-Modell drehen/zoomen oder Circuit öffnen.
2. Einheit anklicken oder über Region und Index auswählen.
3. Auf dem gemeinsamen Zeitstrahl einen Frame untersuchen; Inspector und Graph bleiben synchron.
4. Einen Begriff bestätigen oder einen eigenen Begriff an genau diesen Kontext und Frame binden.
5. Den Journal-Snapshot wieder öffnen; eine spätere Änderung verschiebt die Annotation nicht.
6. JSON exportieren, um lokale Labels/Notizen zu behalten. Ein lokaler Ansichtslink enthält nur die Fixture-Auswahl.

Die Demo startet ohne Wiedergabe und stoppt Replay am Ende des Fensters. Eine Expansion-Einheit zeigt binären Output, kein fiktives LIF-Membranpotenzial. „Unbekannt“ ist ein regulärer Zustand, keine versteckte Fehlermeldung.

## Ehrliche Grenzen

Alle Instrumente verwenden versionierte prozedurale Beispieldaten, keinen laufenden snnTorch-Prozess. Die 140.000 plastischen Gewichte sind eine aus dem lokalen Python-Konstruktor abgeleitete Architekturangabe, nicht ein Telemetrieergebnis.

Die Z-Anatomy/BodyParts3D-Konturen sind echte räumliche Geometrie. SNN-Positionen und Verbindungen darin sind illustrativ, keine biologische Lokalisation. Das 3D-Asset ist separat unter CC BY-SA 4.0 lizenziert: [Attribution](assets/brain-LICENSE.md), [3D-Ausarbeitung](BRAIN-3D.md).

Annotationen bleiben im Speicher des Tabs. Die UI warnt vor unexportierten Änderungen und kennzeichnet einen angeforderten Download. Es gibt weder dauerhafte Nutzerprofile noch Live-Sensoren, Backend-Writes oder echte OS-Aktionen. Der Desktop-Vorschlag ist weiterhin eine reversible UI-Vorschau.

## Lokal betreiben

Der vorhandene Preview-Server unter Port 4178 bedient nur diesen Entwurfsordner. Der Control-Server (`server.control`, Port 8900) liefert denselben Ordner aus; dort startet und stoppt Live session auch das persistente Gehirn. Zum Weitergeben den gesamten Ordner einschließlich `vendor/`, `brain-3d/` und `assets/` kopieren und über HTTP bereitstellen. Keine Installation und keine CDN- oder Schriftabfragen zur Laufzeit.

Die Produktoberfläche bleibt für das internationale Open-Source-Projekt auf Englisch. Konzept und technische Übergabe sind auf Deutsch.

## Prüfen

```sh
rtk proxy node --test docs/dashboard-concepts/verify-observatory.mjs
rtk proxy node 3d/verify-brain.mjs
```

Aktuelle Tests, konkrete Browserbelege und offene Forschungs-/Produktionsanforderungen stehen in [RESEARCH-WORKSPACE.md](RESEARCH-WORKSPACE.md). Die früheren Prüfungen von Design Study 02 sind nicht als Prüfung des neuen Stands umetikettiert.
