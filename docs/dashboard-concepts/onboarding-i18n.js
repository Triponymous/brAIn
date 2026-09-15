const ObservatoryTourI18n = {
  storageKey: 'brain.observatory.introduction.language.v1',
  ui: {
    en: {
      launch: 'Introduction', open: 'Open dashboard introduction', close: 'Close introduction',
      welcome: 'Get to know your Observatory.',
      intro: 'A first-play walkthrough: discover where to click, what you are looking at and what the values really mean.',
      safety: 'You set the pace. The tour never enables sensors or changes capture permissions.',
      choose: 'Start at the beginning or jump to a chapter', chapters: 'Introduction chapters',
      live: 'Live & privacy', methods: 'Methods & export', start: 'Start the tour', later: 'Later',
      storage: 'Only the “introduction seen” flag and your chosen tour language are saved in this browser. Return anytime via “Introduction”.',
      guide: 'Guided introduction', exit: 'End the tour', chapter: 'Choose a chapter', progress: 'Tour progress',
      exercise: 'Try it yourself · optional', focus: 'Explore highlighted area', back: 'Back',
      next: 'Next →', finish: 'Finish ✓', hint: 'Esc exits · You can skip any step', language: 'Tour language'
    },
    de: {
      launch: 'Einführung', open: 'Dashboard-Einführung öffnen', close: 'Einführung schließen',
      welcome: 'Lerne dein Observatory kennen.',
      intro: 'Ein Rundgang wie beim ersten Spielstart: Wir zeigen dir, wo du klicken kannst, was du siehst und was die Werte wirklich bedeuten.',
      safety: 'Du bestimmst das Tempo. Die Tour aktiviert keine Sensoren und verändert keine Erfassungsfreigaben.',
      choose: 'Ganz von vorn oder direkt zu einem Kapitel', chapters: 'Kapitel der Einführung',
      live: 'Live & Datenschutz', methods: 'Methoden & Export', start: 'Rundgang starten', later: 'Später',
      storage: 'Nur der Hinweis „Einführung gesehen“ und deine gewählte Tour-Sprache werden in diesem Browser gespeichert. Über „Einführung“ kannst du jederzeit zurückkommen.',
      guide: 'Geführte Einführung', exit: 'Rundgang beenden', chapter: 'Kapitel wählen', progress: 'Fortschritt im Rundgang',
      exercise: 'Selbst ausprobieren · freiwillig', focus: 'Markierten Bereich ansehen', back: 'Zurück',
      next: 'Weiter →', finish: 'Fertig ✓', hint: 'Esc beendet · Du kannst jeden Schritt überspringen', language: 'Sprache der Einführung'
    }
  },
  steps(language) {
    return ObservatoryTourData.steps.map(step => {
      if (language === 'de') return step;
      const [source, title, description, caution, task, success] = ObservatoryTourEnglish[step.id];
      return { ...step, source, title, description, caution, ...(step.task ? { task, success } : {}) };
    });
  }
};
