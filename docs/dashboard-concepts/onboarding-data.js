const ObservatoryTourData = {
  storageKey: 'brain.observatory.introduction.v1',
  chapters: [
    ['live', '01 / Live & Datenschutz'], ['overview', '02 / Observatory'],
    ['network', '03 / Neural explorer'], ['vocabulary', '04 / Vocabulary'],
    ['journal', '05 / Learning journal'], ['methods', '06 / Methoden & Export']
  ],
  steps: [
    {
      id: 'connection', page: 'live', target: '.live-connection', source: 'Live · lokale Sitzung',
      title: 'Hier beginnt die echte Beobachtung.',
      description: '„Connect local model“ verbindet nur die Ansicht mit dem lokalen Dienst. Ob Daten ankommen, steht hier. Links wechselst du zwischen den sechs Bereichen; nur „Live session“ zeigt echte Modellschritte.',
      caution: 'Die Tour verbindet nichts automatisch. „Disconnect view“ stoppt weder Erfassung noch Lernen.'
    },
    {
      id: 'privacy', page: 'live', target: '.capture-panel', source: 'Live · deine Entscheidung',
      title: 'Du entscheidest, was beobachtet wird.',
      description: 'Vier Schalter: Tastatur- und Mausereignisse zählen, Inaktivität messen und die App grob einordnen. Keine getippten Inhalte. „On“ allein beweist keinen Zugriff; für Eingabezähler und Idle braucht es zusätzlich die macOS-Berechtigung.',
      caution: '„Stop all capture“ stoppt diesen Erfassungspfad, löscht aber keine alten Daten. Mikrofon und Wearables sind hier nicht angeschlossen. Lass die Schalter für die Tour unverändert.'
    },
    {
      id: 'live-stats', page: 'live', target: '.live-stats', source: 'Live · Messwerte lesen',
      title: 'Drei Zahlen, drei verschiedene Dinge.',
      description: '„Captured model tick“ ist der ausgewählte Rechenschritt. „Active outputs“ zählt aktive Ausgänge an diesem Schritt – nicht Hertz. „Retained samples“ zählt die aufbewahrten Schritte.',
      caution: 'Ein Strich bedeutet: kein Wert vorhanden. Das ist nicht dasselbe wie null Aktivität.'
    },
    {
      id: 'live-brain', page: 'live', target: '.live-network', source: 'Live · keine erfundene Aktivität',
      title: 'Ein Modell ansehen, nicht Gedanken lesen.',
      description: 'Die Punktfarben entsprechen „Circuit“: Blau = Input, Violett = Expansion, Mint = Concepts, Bernstein = Memory. Hellere Punkte zeigen aktive Ausgänge; die ausgewählte Einheit ist zusätzlich vergrößert. „Follow incoming samples“ folgt neuen Schritten; der Regler untersucht ältere. Ohne Beobachtungen bleibt die Fläche leer.',
      caution: 'Einfrieren betrifft nur die Ansicht. Die Gehirnform dient der Orientierung; echte Verbindungen werden ohne exportierte Gewichte nicht eingeblendet.'
    },
    {
      id: 'live-unit', page: 'live', target: '.live-inspector', source: 'Live · einzelne Recheneinheit',
      title: 'Was macht genau diese Einheit?',
      description: 'Wähle „Region“ und „Unit index“. Ausgabe und Verlauf gehören zum ausgewählten Tick. „Membrane“ ist ein interner Wert nach dem Rechenschritt; Expansion hat kein Membranpotential.',
      caution: 'DA, NE, ACh und 5-HT sind Modellvariablen. Sie messen keine Neurotransmitter oder Gefühle in deinem Körper.'
    },
    {
      id: 'live-inputs', page: 'live', target: '.live-sensors-panel', source: 'Live · Herkunft der Daten',
      title: 'Welche Eingaben gehörten zu diesem Moment?',
      description: 'Diese Zeilen zeigen die tatsächlich beobachteten Eingaben am ausgewählten Tick. Fehlende und abgeschaltete Eingänge bleiben erkennbar. Beim Zurückspulen siehst du historische Werte.',
      caution: 'Die aktuellen Schalter oben gelten für zukünftige Erfassung. Historische Werte darunter ändern sich durch Abschalten nicht nachträglich.'
    },
    {
      id: 'live-export', page: 'live', target: '.live-methods', source: 'Live · begrenztes Beobachtungsfenster',
      title: 'Ein Fenster sichern, kein ganzes Gehirn.',
      description: '„Export observed window“ lädt die beobachteten Schritte als JSON herunter. Diese Ansicht hält höchstens 200 Schritte, der Dienst bis zu 512 im RAM. Sitzung und Alter der Daten helfen bei der Einordnung.',
      caution: 'Kein vollständiges Experiment und kein Export aller Gewichte. Es gibt kein automatisches Checkpoint-Archiv. Die Tour startet keinen Download.'
    },
    {
      id: 'overview', page: 'overview', target: '#state-banner', source: 'Demo · ab hier synthetische Daten',
      title: 'Ein Wort ist eine Interpretation.',
      description: 'Im Observatory siehst du Beispielmuster wie „Flow“. „Similarity“ beschreibt eine illustrative Ähnlichkeit, keine Sicherheit über dein Befinden. „Confirm label“ und „Teach a word“ annotieren hier nur den Demo-Moment.',
      caution: 'Diese Seite und die folgenden Forschungsansichten sind keine Live-Messung. Demo-Annotationen trainieren den laufenden lokalen Dienst nicht.'
    },
    {
      id: 'contexts', page: 'overview', target: '.moment-control', source: 'Demo · vier vorbereitete Situationen',
      title: 'Wechsle die Situation.',
      description: '„Example context“ wechselt zwischen ruhiger Arbeit, häufigem App-Wechsel, Pause und Unbekanntem. Wort, Erklärung und Beispieldaten verändern sich zusammen.',
      caution: 'Die Situationen sind vorgegeben, nicht aus deinem Desktop erkannt.',
      task: 'Wähle im Menü ein anderes Beispiel. Du darfst auch einfach weitergehen.',
      event: 'change', action: '#moment-select', success: 'Beispiel gewechselt. Die Demo zeigt jetzt einen anderen Kontext.'
    },
    {
      id: 'modulators', page: 'overview', target: '.modulator-section', source: 'Demo · Signale erklären',
      title: 'Was das Lernen beeinflussen soll.',
      description: 'Die vier Karten erklären die Rollen von DA, NE, ACh und 5-HT im Modell. Ein Klick öffnet die jeweilige Erklärung. „Input context“ weiter unten beschreibt die zugehörigen Beispiel-Eingaben.',
      caution: 'Auch diese Werte sind synthetisch. Sechs gezeigte Beispielkanäle bedeuten nicht, dass sechs Sensoren eingeschaltet sind.'
    },
    {
      id: 'brain', page: 'network', target: '#graph-stage', source: 'Demo · interaktives Modell',
      title: 'Drehen. Zoomen. Einer Einheit folgen.',
      description: 'In „Brain“ drehst du per Ziehen und zoomst mit Scrollen oder den +/−-Tasten. Ein Punkt wählt eine Einheit. Side, Front und Top helfen bei der Orientierung; Home setzt die fokussierte 3D-Ansicht zurück.',
      caution: '„Atlas labels“ zeigt anatomische Orientierungspunkte aus der Vorlage. Die Farbgruppen sind brAIn-Rechenrollen, keine biologisch lokalisierten Hirnfunktionen. Falls 3D fehlt, nutze „Circuit“.'
    },
    {
      id: 'regions', page: 'network', target: '.graph-filters', source: 'Demo · vier Rechenrollen',
      title: 'Vom Eingang zum gespeicherten Kontext.',
      description: 'Input: 200 Einheiten. Expansion: 500 feste Projektionseinheiten. Concepts: 200. Memory: 100. Die Filter heben eine Gruppe hervor. „Circuit“ oberhalb des Gehirns trennt die Rollen schematisch; „Contours“ betrifft nur die Form.',
      caution: '„Connections“ zeigt Beispielpfade, keine gerade gemessenen Gewichte.',
      task: 'Klicke zum Beispiel auf „Concepts“, um diese Gruppe hervorzuheben.',
      event: 'click', action: 'button[data-region]', success: 'Region gewählt. Die Markierung und die zugehörige Erklärung sind verknüpft.'
    },
    {
      id: 'neuron', page: 'network', target: '.insight-panel', source: 'Demo · Erklärung und Detail',
      title: 'Vom großen Bild zur einzelnen Einheit.',
      description: '„In plain language“ erklärt den Kontext. „Single neuron“ zeigt die ausgewählte Einheit, ihren Verlauf und Beispielnachbarn. Region, Index und „Find“ sind die präzise Alternative zum Anklicken kleiner Punkte.',
      caution: 'Expansion zeigt binäre Ausgänge statt Membranpotential. Auch eine aktive Konzept-Einheit ist nicht automatisch ein bestimmtes Gefühl.'
    },
    {
      id: 'replay', page: 'network', target: '.replay-toolbar', source: 'Demo · denselben Moment untersuchen',
      title: 'Die Zeit liegt in deiner Hand.',
      description: 'Play spielt 200 synthetische Samples ab; der Regler friert einen Frame ein. „Speed“ ändert nur die Wiedergabe. Gehirn, Einzelkurve und das Raster mit 40 Konzept-Einheiten beziehen sich auf dieselbe Demo-Zeit.',
      caution: '„Annotate this frame“ hält diesen Moment fest. Replay startet keine Sensoren.',
      task: 'Bewege den Frame-Regler und beobachte die Zeitangabe.',
      event: 'input', action: '#frame-slider', success: 'Frame verändert. Du untersuchst jetzt einen anderen Zeitpunkt derselben Demo.'
    },
    {
      id: 'vocabulary', page: 'vocabulary', target: '.vocab-layout', source: 'Demo · deine Begriffe',
      title: 'Wörter bekommen eine Referenz.',
      description: 'Suche oder wähle ein Wort. Die Detailkarte zeigt Referenzsignatur und Herkunft. „Teach a new word“ oben öffnet ein Formular für den festgehaltenen Demo-Moment; bestehende Wörter bekommen eine nachvollziehbare Revision.',
      caution: 'Vier Wörter sind Beispiele. Eigene Demo-Labels bleiben nur in diesem Tab, bis du sie exportierst. Die Tour erstellt keine Annotation.',
      task: 'Wähle eine vorhandene Wortkarte und lies ihre Referenz.',
      event: 'click', action: '[data-word]', success: 'Wort ausgewählt. Die Detailkarte zeigt seine Referenz und Herkunft.'
    },
    {
      id: 'journal', page: 'journal', target: '.journal-layout', source: 'Demo · Spur der Annotationen',
      title: 'Welche Rückmeldung galt für welchen Moment?',
      description: 'Das Journal trennt Beispielbeobachtungen und deine Demo-Korrekturen. Wähle einen Eintrag für seinen Kontext. Eine gespeicherte Momentaufnahme lässt sich im Explorer wieder aufrufen.',
      caution: 'Keine vollständige Live-Chronik. „Your corrections“ enthält auch Beispielkorrekturen: Achte auf „SYNTHETIC“ gegenüber „LOCAL ANNOTATION“.',
      task: 'Probiere den Filter „Your corrections“ oder „Observations“ aus.',
      event: 'click', action: '[data-journal-filter]', success: 'Filter gewählt. Jetzt siehst du nur die passende Art von Einträgen.'
    },
    {
      id: 'methods', page: 'methods', target: '.methods-lead', source: 'Forschung · Belege statt Behauptungen',
      title: 'Was wissen wir – und was noch nicht?',
      description: 'Methods & provenance dokumentiert Demo-Datensatz, Seed, Quellstand, Standardarchitektur und Asset-Lizenzen. Die Architektur weiter unten zeigt 200 → 500 → 200 ↔ 100 Einheiten und die plastischen Gewichtsmatrizen.',
      caution: 'Die Angaben beschreiben die Demo und den referenzierten Code, nicht automatisch die aktuelle Live-Konfiguration oder bewiesene Modellgüte.'
    },
    {
      id: 'export', page: 'methods', target: '.session-strip', source: 'Forschung · nachvollziehbar weiterarbeiten',
      title: 'Du kennst jetzt die wichtigsten Wege.',
      description: '„Copy view“ kopiert die Demo-Auswahl als lokale URL – ohne eigene Labels, Notizen oder Kameraposition. „Export“ enthält dagegen die Demo-Annotationen als JSON. Live besitzt seinen eigenen Export unter „Live session“.',
      caution: 'Dateien vor dem Teilen prüfen. „Fertig“ bringt dich zu deinem Ausgangsbereich zurück; über „Einführung“ kannst du jederzeit ein Kapitel wiederholen.'
    }
  ]
};
