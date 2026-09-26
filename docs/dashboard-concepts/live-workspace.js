const LiveWorkspace = (() => {
  let store, cursor, selected = 'c42', connected = false, following = false;
  let timer, abort, epoch = 0, received = 0, failures = 0, status = 'Not connected';
  // Two local models, one contract: the persistent brain (braind) and the opt-in session runner.
  const endpoints = { daemon: 'http://127.0.0.1:8000/api/telemetry', runner: 'http://127.0.0.1:8001/api/telemetry' };
  const sourceKey = 'brain.observatory.live-source';
  let source = (() => { try { return localStorage.getItem(sourceKey) === 'runner' ? 'runner' : 'daemon'; } catch { return 'daemon'; } })();
  const daemon = () => source === 'daemon';
  const choices = () => document.querySelectorAll('button[data-live-source]');  // not <body>, which carries the choice
  const node = () => ({ key: selected[0], index: Number(selected.slice(1)) });
  const text = (id, value) => { const el = document.getElementById(id); if (el.textContent !== value) el.textContent = value; };

  function compatible() {
    return store && ['s', 'e', 'c', 'w'].every((key, i) => store.meta.architecture[key] === [200, 500, 200, 100][i]);
  }
  function brainSnapshot() {
    return { selected, region: 'all', frame: cursor?.tick ?? 0, heat: [0, 0, 0, 0],
      activity: cursor ? LiveData.activity(cursor) : new Array(1000).fill(0),
      surface: true, showEdges: false, provenance: 'WEBGL · CAPTURED MODEL',
      connectionCaption: '1,000 units · no measured edges' };
  }
  function select(id) {
    selected = id;
    const n = node();
    $('#live-region').value = n.key;
    $('#live-index').max = (store?.meta.architecture[n.key] ?? 200) - 1;
    $('#live-index').value = n.index;
    renderSample();
  }
  async function poll(generation) {
    if (!connected || generation !== epoch || document.hidden || state.page !== 'live') return;
    abort = new AbortController();
    const timeout = setTimeout(() => abort?.abort(), 4000);
    try {
      const url = new URL(endpoints[source]);
      if (store) { url.searchParams.set('session', store.meta.session_id); url.searchParams.set('after', store.frames.at(-1)?.tick ?? 0); }
      const response = await fetch(url, { signal: abort.signal, cache: 'no-store', credentials: 'omit', redirect: 'error' });
      if (!response.ok) throw Error('Telemetry endpoint returned HTTP ' + response.status);
      const body = await response.text();
      if (body.length > 16_000_000) throw Error('Telemetry batch too large');
      const next = LiveData.merge(store, JSON.parse(body));
      if (!connected || generation !== epoch) return;
      const changed = store && next.meta.session_id !== store.meta.session_id;
      store = next; received = performance.now(); failures = 0;
      if (store.meta.status !== 'streaming') following = false;
      if (changed) { cursor = null; following = false; status = 'New session · inspect before following'; }
      else status = { waiting: daemon() ? 'Waiting for a shared source' : 'Waiting for an enabled input',
        streaming: 'Receiving model observations', stale: 'Stale · no recent model steps', error: 'Capture stopped after an error',
        paused: daemon() ? 'Paused · nothing shared, so the brain does not learn' : 'Capture paused · no new model steps' }[store.meta.status];
      if (!cursor || following) cursor = store.frames.at(-1);
      if (node().index >= store.meta.architecture[node().key]) selected = node().key + '0';
      render();
    } catch (error) {
      if (generation !== epoch) return;
      failures++;
      status = error.name === 'AbortError' ? 'Connection timed out · last sample retained' :
        error instanceof TypeError ? (daemon() ? 'Unavailable · start the daemon' : 'Unavailable · start the local observation service') : error.message;
      following = false;
      render();
    } finally {
      clearTimeout(timeout);
      if (connected && generation === epoch) timer = setTimeout(() => poll(generation), Math.min(8000, 1000 * 2 ** Math.min(failures, 3)));
    }
  }
  function connect() {
    connected = true; failures = 0; status = 'Connecting…'; render();
    clearTimeout(timer); poll(++epoch);
  }
  function disconnect() {
    connected = false; following = false; epoch++; clearTimeout(timer); abort?.abort();
    status = 'Disconnected · retained window is historical'; render();
  }
  function setSource(next) {
    if (!endpoints[next] || next === source) return;
    // Each model is its own history: never mix frames, and never keep reading the other one.
    connected = false; following = false; epoch++; clearTimeout(timer); abort?.abort();
    store = undefined; cursor = undefined; failures = 0; status = 'Not connected';
    source = next;
    try { localStorage.setItem(sourceKey, next); } catch {}
    onPageChange();
  }
  function onPageChange() {
    const live = state.page === 'live';
    document.body.dataset.liveSource = source;
    for (const button of choices()) {
      button.setAttribute('aria-checked', String(button.dataset.liveSource === source));
      button.tabIndex = button.dataset.liveSource === source ? 0 : -1;  // one tab stop; arrows move within
    }
    $('.source-pill').textContent = live ? (daemon() ? 'PERSISTENT BRAIN' : 'LOCAL OBSERVATION') : 'SYNTHETIC DATA';
    $('.connection-state h3').textContent = live ? (daemon() ? 'Your persistent brain' : 'Local, opt-in observation') : 'Example dataset';
    $('.connection-state p').textContent = live ? (daemon() ? 'Saved checkpoint. Every source stays off until you share it.' :
      'No microphone, cloud or desktop actions in this runner.') : 'Browser-only study. No live sensors or OS control.';
    $('.app-footer span').textContent = 'brAIn / OBSERVATORY';
    $('.app-footer span:nth-child(2)').textContent = live ? (daemon() ? 'PERSISTENT BRAIN · USER-CONTROLLED SOURCES' :
      'LOCAL OBSERVATION · USER-CONTROLLED INPUTS') : 'RESEARCH WORKSPACE 03 · EXAMPLE DATA';
    CaptureControls.refresh(); DaemonControls.refresh(); FeltPanel.refresh();
    clearTimeout(timer); epoch++; abort?.abort();
    if (!live) { following = false; return; }
    render();
    if (connected && !document.hidden) poll(epoch);
  }
  function render() {
    text('live-status', status);
    $('#live-connect').disabled = connected;
    $('#live-disconnect').disabled = !connected;
    $('#live-follow').disabled = !connected || !store?.frames.length || reduced.matches ||
      store.meta.status !== 'streaming' || failures > 0;
    $('#live-follow').textContent = following ? 'Freeze view' : 'Follow incoming samples';
    $('#live-follow').setAttribute('aria-pressed', String(following));
    $('#live-export').disabled = !store?.frames.length;
    text('live-source', store?.meta.status === 'paused' ? (store.frames.length ?
      'CAPTURE OFF · HISTORICAL MODEL DATA' : 'CAPTURE OFF · NO OBSERVATIONS') : store ? { desktop_metadata: 'ACTUAL MODEL · DESKTOP METADATA',
      desktop_sensors: 'ACTUAL MODEL · DESKTOP SENSORS', test_fixture: 'ACTUAL MODEL · TEST INPUTS' }[store.meta.source.input_kind] ??
      'SENSORS DISABLED' : 'NO DATA SOURCE');
    text('live-session', store?.meta.session_id ?? 'No session');
    text('live-input-note', store?.meta.source.input_kind === 'test_fixture' ?
      'Test inputs are not a desktop recording. They only verify the model-to-dashboard pipeline.' : daemon() ?
      'This view reads your persistent brain. Missing inputs stay missing; it never falls back to the demo.' :
      'This view reads a local model. Missing inputs stay missing; it never falls back to the demo.');
    text('live-window-count', String(store?.frames.length ?? 0));
    text('live-gap', store?.gaps ? 'History gap detected · showing retained samples only' : 'Rolling window · not a complete recording');
    text('live-age', store?.meta.age_s === null || !store ? '—' : (store.meta.age_s + (performance.now() - received) / 1000).toFixed(1) + ' s at receipt');
    const slider = $('#live-cursor');
    slider.disabled = !store?.frames.length; slider.max = Math.max(0, (store?.frames.length ?? 1) - 1);
    slider.value = Math.max(0, store?.frames.findIndex(frame => frame.tick === cursor?.tick) ?? 0);
    $('#live-region').disabled = !cursor; $('#live-index').disabled = !cursor;
    $('#live-index').max = (store?.meta.architecture[node().key] ?? 200) - 1;
    $('#live-index').value = node().index;
    text('live-view-mode', following ? 'Following latest received sample' : 'Frozen inspection · collection is independent');
    renderSample();
  }
  function renderSample() {
    const currentNode = node(), sample = cursor?.regions[currentNode.key];
    text('live-tick', cursor ? cursor.tick.toLocaleString('en-US') : '—');
    text('live-sample-time', cursor ? new Date(cursor.captured_at).toLocaleTimeString('en-GB', { hour12: false }) + ' · +' + cursor.elapsed_s.toFixed(3) + ' s' : 'Waiting for a captured sample');
    $('#live-cursor').setAttribute('aria-valuetext', cursor ? 'Captured tick ' + cursor.tick : 'No captured samples');
    text('live-unit-name', selected[0].toUpperCase() + '-' + String(currentNode.index).padStart(3, '0'));
    text('live-output', sample ? String(sample.output[currentNode.index]) : '—');
    text('live-membrane', !sample ? 'No data' : sample.membrane ? sample.membrane[currentNode.index].toFixed(4) + ' a.u.' : 'Not applicable · projection');
    text('live-active', cursor ? String(LiveData.activity(cursor).reduce((a, b) => a + b, 0)) : '—');
    text('live-model-time', cursor ? 'Integration step ' + cursor.tick + ' · dt 1.0, separate from elapsed wall time' : 'No model steps observed');
    text('live-capture-history', cursor ? 'Historical input values at selected tick · capture selection revision ' +
      (cursor.capture_revision ?? 'not controlled / test input') + '. Current switches above apply to future collection, not this snapshot.' :
      'No captured input values yet. Current collection settings are shown above.');
    for (const name of ['DA', 'NE', 'ACh', '5HT']) text('live-mod-' + name, cursor ? cursor.modulators[name].toFixed(3) : '—');
    renderSensors(); renderTrace();
    $('#live-no-data').hidden = !!cursor && !!compatible();
    text('live-no-data', !cursor ? 'No observed activity yet. Connect to your local session to inspect a real model step.' :
      'This model uses a different architecture. Numeric inspection remains available; the 1,000-unit atlas is not shown.');
    $('#live-graph-stage').hidden = !cursor || !compatible();
    if (state.page === 'live' && cursor && compatible()) buildBrainGraph();
  }
  function renderSensors() {
    const names = { keystroke_rate: 'Keyboard', mouse_rate: 'Pointer', idle: 'Idle time',
      active_app: 'App category', microphone: 'Microphone', wearable: 'Wearable' };
    $('#live-sensors').replaceChildren(...Object.entries(names).map(([key, name]) => {
      const sensor = cursor?.sensors[key], row = document.createElement('div'); row.className = 'live-sensor';
      const title = document.createElement('strong'); title.textContent = name;
      const value = document.createElement('span');
      // Small readings such as microphone loudness keep two significant digits instead of rounding to 0.0.
      value.textContent = !sensor ? 'No data' : sensor.status !== 'available' ? sensor.status.replaceAll('_', ' ') :
        typeof sensor.value === 'number' ? (sensor.value >= 1 || !sensor.value ? sensor.value.toFixed(1) : sensor.value.toPrecision(2)) + ' ' + sensor.unit : sensor.value;
      const detail = document.createElement('small');
      detail.textContent = sensor?.status === 'available' ? 'Observed ' + new Date(sensor.observed_at * 1000).toLocaleTimeString('en-GB') +
        (sensor.window_s ? ' · ' + sensor.window_s.toFixed(2) + ' s window' : '') : 'Never replaced with zero';
      row.append(title, value, detail); return row;
    }));
  }
  function renderTrace() {
    const svg = $('#live-trace'); svg.replaceChildren();
    if (!cursor || !store?.frames.length) { text('live-trace-note', 'No generated traces. Waiting for observed unit values.'); return; }
    const n = node(), frames = store.frames.filter(frame => frame.tick <= cursor.tick);
    if (!frames.length) { text('live-trace-note', 'Frozen tick has left the rolling window. Its numeric snapshot is retained.'); return; }
    const values = frames.map(frame => n.key === 'e' ? frame.regions.e.output[n.index] : frame.regions[n.key].membrane[n.index]);
    const min = Math.min(0, ...values), max = Math.max(min + .001, ...values);
    const start = frames[0].elapsed_s, duration = Math.max(.001, frames.at(-1).elapsed_s - start);
    const points = values.map((value, index) => [(44 + (frames[index].elapsed_s - start) / duration * 410).toFixed(2),
      (105 - (value - min) / (max - min) * 85).toFixed(2)]);
    svg.append(svgEl('path', { d: points.map((p, i) => (i ? 'L' : 'M') + p.join(' ')).join(' '), fill: 'none', stroke: 'var(--mint)', 'stroke-width': 1.5 }));
    svgText(svg, 38, 23, max.toFixed(2), 'end'); svgText(svg, 38, 108, min.toFixed(2), 'end');
    svgText(svg, 44, 133, '+' + start.toFixed(2) + ' s', 'start');
    svgText(svg, 454, 133, '+' + frames.at(-1).elapsed_s.toFixed(2) + ' s', 'end');
    text('live-trace-note', frames.length + ' captured steps · actual elapsed time · ' +
      (n.key === 'e' ? 'binary projection output' : 'post-step membrane, adaptive vertical scale; peaks before reset are not recorded'));
  }
  function exportWindow() {
    if (!store?.frames.length) return;
    const data = LiveData.exportWindow(store, cursor, selected);
    data.selected_sample = cursor ? structuredClone(cursor) : null;
    const url = URL.createObjectURL(new Blob([JSON.stringify(data)], { type: 'application/json' }));
    const link = document.createElement('a'); link.href = url; link.download = 'brain-observation-' + store.meta.session_id + '.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    notify('Observed-window download requested. Contains local activity metadata; keep it private.');
  }
  for (const button of choices()) {
    button.addEventListener('click', () => setSource(button.dataset.liveSource));
    button.addEventListener('keydown', event => {
      if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
      event.preventDefault();
      const other = [...choices()].find(b => b !== button);
      setSource(other.dataset.liveSource); other.focus();
    });
  }
  $('#live-connect').addEventListener('click', connect);
  $('#live-disconnect').addEventListener('click', disconnect);
  $('#live-follow').addEventListener('click', () => { following = !following; if (following) cursor = store.frames.at(-1); render(); });
  $('#live-export').addEventListener('click', exportWindow);
  $('#live-cursor').addEventListener('input', event => { following = false; cursor = store.frames[Number(event.target.value)]; render(); });
  $('#live-region').addEventListener('change', event => select(event.target.value + Math.min(node().index, store.meta.architecture[event.target.value] - 1)));
  $('#live-index').addEventListener('input', event => { const index = Number(event.target.value); if (Number.isInteger(index) && index >= 0 && index < store.meta.architecture[node().key]) select(node().key + index); });
  document.addEventListener('visibilitychange', onPageChange);
  document.addEventListener('brain-capture-change', onPageChange);
  reduced.addEventListener('change', () => { if (reduced.matches) { following = false; render(); } });
  window.addEventListener('pagehide', disconnect);
  return { onPageChange, brainSnapshot, select, exportWindow, compatible, setSource, source: () => source };
})();
LiveWorkspace.onPageChange();
