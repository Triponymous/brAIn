// Copy only: step identities, navigation and optional exercises stay shared.
const ObservatoryTourEnglish = {
  connection: [
    'Live · local session', 'Real observation starts here.',
    '“Connect local model” only connects this view to the local service. Its status tells you whether data is arriving. Use the sidebar to move between six areas; only “Live session” shows real model steps.',
    'The tour never connects automatically. “Disconnect view” stops neither capture nor learning.'
  ],
  privacy: [
    'Live · your decision', 'You decide what is observed.',
    'Four switches count keyboard and mouse events, measure idle time and classify the active app into a broad category. No typed content is collected. “On” alone does not prove access: input counters and idle time also require macOS permission.',
    '“Stop all capture” stops this capture path, but does not delete past data. Microphone and wearables are not connected here. Leave the switches unchanged during the tour.'
  ],
  'live-stats': [
    'Live · reading measurements', 'Three numbers, three different things.',
    '“Captured model tick” is the selected computation step. “Active outputs” counts active outputs at that step—not hertz. “Retained samples” counts the steps currently kept in memory.',
    'A dash means no value is available. That is not the same as zero activity.'
  ],
  'live-brain': [
    'Live · no fabricated activity', 'Inspect a model, not someone’s thoughts.',
    'Point colors match “Circuit”: blue = Input, violet = Expansion, mint = Concepts, amber = Memory. Brighter points indicate active outputs; the selected unit is also enlarged. “Follow incoming samples” follows new steps; the slider explores older ones. Without observations, the view stays empty.',
    'Freezing only affects the view. The brain shape is an orientation aid; real connections are not displayed without exported weights.'
  ],
  'live-unit': [
    'Live · one computational unit', 'What is this particular unit doing?',
    'Choose “Region” and “Unit index”. The output and trace belong to the selected tick. “Membrane” is an internal value after the computation step; Expansion has no membrane potential.',
    'DA, NE, ACh and 5-HT are model variables. They do not measure neurotransmitters or emotions in your body.'
  ],
  'live-inputs': [
    'Live · data provenance', 'Which inputs belong to this moment?',
    'These rows show the inputs actually observed at the selected tick. Missing and disabled inputs remain distinguishable. When you rewind, you are looking at historical values.',
    'The current switches above control future capture. Switching an input off does not retroactively change the historical values below.'
  ],
  'live-export': [
    'Live · a bounded observation window', 'Save a window, not an entire brain.',
    '“Export observed window” downloads the observed steps as JSON. This view retains at most 200 steps; the service keeps up to 512 in RAM. Session identity and data age help you interpret the result.',
    'This is neither a complete experiment nor an export of all weights. There is no automatic checkpoint archive. The tour does not start a download.'
  ],
  overview: [
    'Demo · synthetic data from here on', 'A word is an interpretation.',
    'The Observatory displays example patterns such as “Flow”. “Similarity” is an illustrative comparison, not certainty about how you feel. “Confirm label” and “Teach a word” only annotate the demo moment here.',
    'This page and the following research views are not live measurements. Demo annotations do not train the running local service.'
  ],
  contexts: [
    'Demo · four prepared situations', 'Change the situation.',
    '“Example context” switches between quiet work, frequent app changes, a break and an unknown pattern. The word, explanation and example data change together.',
    'These situations are predefined, not detected from your desktop.',
    'Choose another example from the menu. You can also simply move on.',
    'Example changed. The demo now shows a different context.'
  ],
  modulators: [
    'Demo · explaining signals', 'What is intended to influence learning?',
    'The four cards explain the roles of DA, NE, ACh and 5-HT in the model. Click a card to open its explanation. “Input context” further down describes the corresponding example inputs.',
    'These values are synthetic too. Six example channels do not mean six sensors are enabled.'
  ],
  brain: [
    'Demo · interactive model', 'Rotate. Zoom. Follow a unit.',
    'In “Brain”, drag to rotate and scroll or use the +/− keys to zoom. Select a point to inspect a unit. Side, Front and Top help you orient yourself; Home resets the focused 3D view.',
    '“Atlas labels” shows anatomical landmarks from the source model. Color groups represent brAIn’s computational roles, not biologically localized brain functions. If 3D is unavailable, use “Circuit”.'
  ],
  regions: [
    'Demo · four computational roles', 'From input to retained context.',
    'Input: 200 units. Expansion: 500 fixed projection units. Concepts: 200. Memory: 100. The filters highlight one group. “Circuit” above the brain separates these roles schematically; “Contours” only affects the shape.',
    '“Connections” shows example paths, not weights being measured right now.',
    'Try clicking “Concepts” to highlight that group.',
    'Region selected. Its highlight and explanation are linked.'
  ],
  neuron: [
    'Demo · explanation and detail', 'From the big picture to a single unit.',
    '“In plain language” explains the context. “Single neuron” shows the selected unit, its trace and example neighbors. Region, index and “Find” offer a precise alternative to clicking small points.',
    'Expansion shows binary outputs instead of membrane potential. An active concept unit does not automatically represent a particular emotion.'
  ],
  replay: [
    'Demo · inspecting the same moment', 'Time is in your hands.',
    'Play replays 200 synthetic samples; the slider freezes a frame. “Speed” only changes playback. The brain, individual trace and raster of 40 concept units refer to the same demo time.',
    '“Annotate this frame” captures that moment. Replay does not start any sensors.',
    'Move the frame slider and watch the time indicator.',
    'Frame changed. You are inspecting a different moment in the same demo.'
  ],
  vocabulary: [
    'Demo · your vocabulary', 'Give words a reference.',
    'Search for or select a word. Its detail card shows the reference signature and provenance. “Teach a new word” at the top opens a form for the captured demo moment; existing words receive a traceable revision.',
    'Four words are examples. Your own demo labels remain only in this tab until you export them. The tour does not create annotations.',
    'Select an existing word card and read its reference.',
    'Word selected. The detail card shows its reference and provenance.'
  ],
  journal: [
    'Demo · an annotation trail', 'Which feedback belongs to which moment?',
    'The journal separates example observations from your demo corrections. Select an entry to see its context. A saved snapshot can be reopened in the explorer.',
    'This is not a complete live history. “Your corrections” also includes example corrections: distinguish “SYNTHETIC” from “LOCAL ANNOTATION”.',
    'Try the “Your corrections” or “Observations” filter.',
    'Filter selected. You are now seeing the matching type of entries.'
  ],
  methods: [
    'Research · evidence before claims', 'What do we know—and what is still open?',
    'Methods & provenance documents the demo dataset, seed, source revision, default architecture and asset licenses. The architecture below shows 200 → 500 → 200 ↔ 100 units and the plastic weight matrices.',
    'These details describe the demo and referenced code, not necessarily the current live configuration or proven model quality.'
  ],
  export: [
    'Research · a traceable next step', 'You now know the main paths.',
    '“Copy view” copies the demo selection as a local URL—without your own labels, notes or camera position. “Export” includes demo annotations as JSON. Live has a separate export under “Live session”.',
    'Review files before sharing them. “Finish” returns you to your starting area; use “Introduction” to revisit any chapter.'
  ]
};
