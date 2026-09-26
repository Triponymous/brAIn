const CaptureControls = (() => {
  const endpoint = 'http://127.0.0.1:8001/api/capture';
  const label = { disabled: 'Off · not queried', waiting: 'On · awaiting a sample', available: 'On · receiving',
    permission_required: 'On · macOS permission required', unavailable: 'On · no sample available', error: 'Capture error · stopped' };
  let data, verified = false, pending = false, message = '', timer, reader, writer, epoch = 0, operation = 0, failures = 0;
  const byId = id => document.getElementById(id);
  const text = (id, value) => { if (byId(id).textContent !== value) byId(id).textContent = value; };
  // Only while the session runner is the chosen source; the persistent brain has its own controls.
  const runnerChosen = () => (document.body?.dataset?.liveSource ?? 'runner') === 'runner';

  function accept(next) {
    CaptureData.validate(next);
    if (data?.session_id === next.session_id && next.revision < data.revision) return;
    const changed = data && (next.session_id !== data.session_id || next.revision !== data.revision);
    data = next; verified = true; failures = 0;
    if (changed) document.dispatchEvent(new Event('brain-capture-change'));
  }
  async function read(generation) {
    if (generation !== epoch || document.hidden || state.page !== 'live' || !runnerChosen()) return;
    if (!pending) {
      const controller = new AbortController(); reader = controller;
      const timeout = setTimeout(() => controller.abort(), 4000);
      try {
        const response = await fetch(endpoint, { signal: controller.signal, cache: 'no-store', credentials: 'omit', redirect: 'error' });
        if (!response.ok) throw Error('Controls unavailable');
        const next = await response.json();
        if (generation !== epoch) return;
        accept(next);
      } catch {
        if (generation !== epoch) return;
        verified = false; failures++;
      } finally {
        clearTimeout(timeout);
        if (generation === epoch) render();
      }
    }
    if (generation === epoch) timer = setTimeout(() => read(generation), Math.min(8000, 1000 * 2 ** Math.min(failures, 3)));
  }
  async function change(enabled) {
    if (!data) return;
    const request = CaptureData.change(data, enabled), ticket = ++operation;
    const stopping = Object.values(enabled).every(value => !value);
    writer?.abort(); const controller = new AbortController(); writer = controller;
    const signal = controller.signal, timeout = setTimeout(() => controller.abort(), 4000);
    pending = true; message = stopping ? 'Requesting stop — waiting for confirmation…' : 'Applying selection — waiting for confirmation…';
    render();
    try {
      const response = await fetch(endpoint, { method: 'POST', signal, cache: 'no-store', credentials: 'omit', redirect: 'error',
        headers: { 'Content-Type': 'application/json', 'X-Brain-Control': 'capture-v1' }, body: JSON.stringify(request) });
      if (!response.ok) throw Error(response.status === 409 ? 'Settings changed elsewhere. Review the current switches and try again.' :
        'The server did not confirm this change.');
      const next = await response.json();
      if (ticket !== operation) return;
      if (data.session_id !== request.session_id) throw Error('Session changed. Review the current switches.');
      accept(next);
      message = Object.values(data.inputs).some(input => input.enabled) ?
        'Selection confirmed. Each input shows separately whether data is arriving or permission is missing.' :
        'All capture stopped. No new sensor reads or model steps. Earlier snapshots and learned state remain in memory.';
    } catch (error) {
      if (ticket !== operation) return;
      verified = false;
      message = stopping ? 'Stop not confirmed. Capture may still be running. Reconnect or stop the runner with Ctrl-C.' :
        error.name === 'AbortError' ? 'Change not confirmed. Wait for the actual server state before trying again.' : error.message;
    } finally {
      clearTimeout(timeout);
      if (ticket === operation) { pending = false; render(); refresh(); }
    }
  }
  function render() {
    const count = data ? Object.values(data.inputs).filter(input => input.enabled).length : 0;
    text('capture-summary', !verified ? 'Capture status unverified' : data.failed ? 'Capture stopped after an error' :
      !data.armed ? 'Input controls not armed' : count ? count + ' of 4 inputs enabled' : 'All inputs off');
    byId('capture-summary').dataset.state = !verified ? 'unknown' : data.failed ? 'error' : count ? 'on' : 'off';
    text('capture-connection', !verified ?
      'Cannot verify this runner. Capture may continue if it was enabled. These switches show only the last confirmed selection.' :
      !data.armed ? 'Start server.observe with --desktop-metadata to make these controls available. It still starts with every input off.' :
      'Confirmed by the local runner · selection revision ' + data.revision + ' · new server sessions start with every input off.');
    text('capture-message', message);
    byId('capture-stop').disabled = !data || (verified && !count && !pending);
    for (const key of CaptureData.keys) {
      const input = data?.inputs[key], toggle = byId('capture-' + key);
      toggle.checked = input?.enabled ?? false;
      // Keep keyboard focus while the request is pending; the handler blocks repeat changes.
      toggle.disabled = !data || !data.armed || data.failed || (!verified && !input?.enabled);
      toggle.setAttribute('aria-disabled', String(pending || toggle.disabled));
      text('capture-state-' + key, input ? (verified ? label[input.status] : 'Unverified · last selection ' + (input.enabled ? 'on' : 'off')) : 'Status not loaded');
    }
  }
  function refresh() {
    clearTimeout(timer); epoch++; reader?.abort();
    if (state.page === 'live' && !document.hidden && runnerChosen()) read(epoch);
  }
  for (const key of CaptureData.keys) byId('capture-' + key).addEventListener('change', event => {
    if (pending) { render(); return; }
    return change({ [key]: event.target.checked });
  });
  byId('capture-stop').addEventListener('click', () => change(Object.fromEntries(CaptureData.keys.map(key => [key, false]))));
  byId('capture-refresh').addEventListener('click', refresh);
  document.addEventListener('visibilitychange', refresh);
  window.addEventListener('pagehide', () => { clearTimeout(timer); epoch++; reader?.abort(); });
  render();
  return { refresh };
})();
