const sourceRevision = 'f8ba635a5b2307934c5b0adba0ca8b0e7be952b7';
let appReady = false;
let exportBusy = false;

function slotKey(scenario = state.scenario, frame = state.frame) {
  return `${scenario}:${frame}`;
}

function freezeMoment() {
  return { scenario: state.scenario, time: current().time, frame: state.frame,
    mods: [...current().mods], title: current().title, moment: state.moment,
    selected: state.selected, region: state.region, graphMode: state.graphMode };
}

function recordAnnotation(title, copy, snapshot, label, previous = null) {
  state.revision++;
  const word = vocabulary.find(word => word.label === label);
  word.corrections++;
  word.sessionAnnotations = (word.sessionAnnotations || 0) + 1;
  events.unshift({ id: 'annotation-' + String(state.revision).padStart(3, '0'),
    time: snapshot.time, title, copy, type: 'correction', icon: 'check',
    moment: snapshot.moment, local: true, snapshot: structuredClone(snapshot),
    label, reference: structuredClone(vocabulary.find(word => word.label === label)),
    previous: previous ? structuredClone(previous) : null });
  renderSession();
}

function renderSession() {
  const count = events.filter(event => event.local).length;
  const pending = state.revision - state.exportedRevision;
  const label = pending ? `${pending} unexported annotation${pending === 1 ? '' : 's'}` :
    count ? `${count} annotation${count === 1 ? '' : 's'} · snapshot requested` : 'No local annotations';
  $('#annotation-status').textContent = label;
  $('#annotation-status').dataset.dirty = String(pending > 0);
  $('#annotation-status').title = 'Tab memory only. Export and keep the downloaded file before reloading.';
}

function viewState() {
  return { scenario: state.scenario, unit: state.selected, frame: state.frame,
    region: state.region, view: state.graphMode, surface: state.brainSurface,
    connections: $('#show-edges').checked, speed: state.speed };
}

function syncViewURL() {
  if (!appReady) return;
  const url = new URL(location.pathname, location.origin);
  url.hash = location.hash;
  if (new URLSearchParams(location.search).has('WEBGL_OFF')) url.searchParams.set('WEBGL_OFF', '1');
  if (state.page !== 'live') for (const [key, value] of Object.entries(viewState())) url.searchParams.set(key, String(value));
  history.replaceState(null, '', url);
}

function restoreViewURL() {
  const q = new URLSearchParams(location.search);
  if (Object.hasOwn(scenarios, q.get('scenario'))) state.scenario = q.get('scenario');
  if (byId.has(q.get('unit'))) state.selected = q.get('unit');
  const frame = q.has('frame') ? Number(q.get('frame')) : 42;
  if (Number.isInteger(frame) && frame >= 0 && frame < 200) state.frame = frame;
  if (['all', ...regions.map(region => region.key)].includes(q.get('region'))) state.region = q.get('region');
  if (state.region !== 'all' && byId.get(state.selected).key !== state.region) state.region = 'all';
  if (['brain', 'circuit'].includes(q.get('view'))) state.graphMode = q.get('view');
  state.brainSurface = q.get('surface') !== 'false';
  if ([.1, .5, 1].includes(Number(q.get('speed')))) state.speed = Number(q.get('speed'));
  $('#show-edges').checked = q.get('connections') !== 'false';
  $('#show-brain-surface').checked = state.brainSurface;
  $('#moment-select').value = state.scenario;
  $('#replay-speed').value = String(state.speed);
  state.moment = { focus: 5, switching: 2, rest: 3, unknown: 4 }[state.scenario];
  $('#neural-workspace').dataset.graphMode = state.graphMode;
  $$('[data-graph-view]').forEach(button => button.setAttribute('aria-pressed', button.dataset.graphView === state.graphMode));
  $$('[data-region]').forEach(button => button.setAttribute('aria-pressed', button.dataset.region === state.region));
  renderGraphControls();
}

async function copyView() {
  syncViewURL();
  try {
    await navigator.clipboard.writeText(location.href);
    notify('View link copied. It contains the fixture selection, not your labels or notes.');
  } catch {
    showInfo('<div class="eyebrow">Copy this data view</div><h2>Your view has a URL.</h2><p>Clipboard access is unavailable. Select and copy the link below. Local labels and notes are not included.</p><label for="share-view">View URL</label><input class="input" id="share-view" readonly>');
    $('#share-view').value = location.href;
    $('#share-view').select();
  }
}

async function exportSession() {
  if (state.page === 'live') return LiveWorkspace.exportWindow();
  if (exportBusy) return;
  exportBusy = true;
  const revision = state.revision;
  const dataset = { ...DemoTelemetry.meta, contexts: scenarios };
  const data = structuredClone({
    schema: 'brain.observatory.snapshot.v3', synthetic: true,
    source: 'Browser fixture; not a brain checkpoint or experimental result',
    source_revision: sourceRevision, exported_at: new Date().toISOString(),
    dataset, architecture: { sensory: 200, expansion: 500, concept: 200,
      working_memory: 100, plastic_weights: 140000 },
    sample_connections: { seed: 92712, catalog_count: edges.length, spatial_stride: 5 },
    view: { page: state.page, ...viewState() },
    selected_unit_series: DemoTelemetry.snapshotSeries(state.scenario, state.selected),
    vocabulary, annotations: events.filter(event => event.local), example_history: events.filter(event => !event.local),
    annotation_revision: revision,
    limitations: ['No live telemetry', 'Procedural activity, not a snnTorch run',
      'Authored context signals and similarity scores', 'Not calibrated confidence',
      'Anatomical positions and moving fiber lights are illustrative', 'No OS actions'],
    license_notes: { code: 'Project license; inspect repository LICENSE',
      anatomy: 'Adapted Z-Anatomy / BodyParts3D, CC BY-SA 4.0; assets/brain-LICENSE.md' }
  });
  try {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(dataset)));
    data.dataset_sha256 = [...new Uint8Array(digest)].map(n => n.toString(16).padStart(2, '0')).join('');
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url; link.download = `brain-${DemoTelemetry.meta.id}-${data.view.scenario}-r${revision}.json`;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    state.exportedRevision = revision;
    renderSession();
    notify('Snapshot download requested. Keep the JSON file; local labels and notes are included.');
  } catch {
    notify('Snapshot export failed. Keep this tab open and try Export again.');
  } finally { exportBusy = false; }
}

window.addEventListener('beforeunload', event => {
  if (state.revision <= state.exportedRevision) return;
  event.preventDefault();
  event.returnValue = '';
});
