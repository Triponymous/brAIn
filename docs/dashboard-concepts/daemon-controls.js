/* The persistent brain's sources and process. Only replies from braind (8000) and server.control (8900) change what is shown. */
const DaemonControls = (() => {
  const daemonUrl = 'http://127.0.0.1:8000', controlUrl = 'http://127.0.0.1:8900';
  const label = { disabled: 'Off · not captured', waiting: 'On · no data yet · macOS permission?', available: 'On · receiving' };
  let data, verified = false, pending = false, message = '', timer, reader, writer, epoch = 0, operation = 0, failures = 0;
  let proc = null, procPending = false, procMessage = '';
  const byId = id => document.getElementById(id);
  const text = (id, value) => { if (byId(id).textContent !== value) byId(id).textContent = value; };
  const active = () => document.body?.dataset?.liveSource === 'daemon' && state.page === 'live' && !document.hidden;
  const request = (url, options = {}) => fetch(url, { cache: 'no-store', credentials: 'omit', redirect: 'error', ...options });
  async function json(url, signal) {
    const response = await request(url, { signal });
    if (!response.ok) throw Error('HTTP ' + response.status);
    return response.json();
  }

  function accept(next) {
    DaemonData.consent(next);
    const changed = data && next.revision !== data.revision;
    data = next; verified = true; failures = 0;
    if (changed) document.dispatchEvent(new Event('brain-sources-change'));
  }
  async function read(generation) {
    if (generation !== epoch || !active()) return;
    const controller = new AbortController(); reader = controller;
    const timeout = setTimeout(() => controller.abort(), 4000);
    await Promise.all([
      pending ? null : json(daemonUrl + '/api/consent', controller.signal).then(accept).catch(() => {
        if (generation === epoch && !pending) { verified = false; failures++; }
      }),
      procPending ? null : json(controlUrl + '/daemon/status', controller.signal).then(next => {
        if (generation === epoch) proc = DaemonData.daemon(next);
      }).catch(() => { if (generation === epoch) proc = null; })
    ]);
    clearTimeout(timeout);
    if (generation !== epoch) return;
    render();
    timer = setTimeout(() => read(generation), Math.min(8000, 3000 * 2 ** Math.min(failures, 1)));
  }
  async function change(enabled) {
    if (!data || pending) return;
    const body = DaemonData.change(data, enabled), ticket = ++operation;
    const stopping = Object.values(enabled).every(value => !value);
    reader?.abort(); writer?.abort();
    const controller = new AbortController(); writer = controller;
    const timeout = setTimeout(() => controller.abort(), 4000);
    pending = true; message = stopping ? 'Requesting stop — waiting for the daemon to confirm…' : 'Sharing — waiting for the daemon to confirm…';
    render();
    try {
      const response = await request(daemonUrl + '/api/consent', { method: 'POST', signal: controller.signal,
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const reply = await response.json().catch(() => ({}));
      if (!response.ok) throw Error(response.status === 409 ? 'Changed elsewhere. Review the current switches and try again.' :
        typeof reply.detail === 'string' ? reply.detail : 'The daemon did not confirm this change.');
      if (ticket !== operation) return;
      accept(reply);
      message = data.paused ? 'All sources off. Nothing is captured and the brain does not learn. What it learned stays saved.' :
        'Confirmed and saved. Each source shows whether data is arriving or macOS permission is missing.';
    } catch (error) {
      if (ticket !== operation) return;
      verified = false;
      message = error.name === 'AbortError' || error instanceof TypeError ?
        (stopping ? 'Stop not confirmed. Capture may still be running. Stop the daemon above, or with Ctrl-C.' :
          'Change not confirmed. Wait for the actual daemon state before trying again.') : error.message;
    } finally {
      clearTimeout(timeout);
      if (ticket === operation) { pending = false; render(); refresh(); }
    }
  }
  async function processAction(action) {
    if (procPending) return;
    procPending = true; procMessage = action === 'start' ? 'Starting the daemon…' : 'Stopping the daemon…';
    render();
    try {
      const response = await request(controlUrl + '/daemon/' + action, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: '{}' });
      if (!response.ok) throw Error();
      procMessage = action === 'start' ? 'Start requested. The daemon loads its checkpoint and its saved sources.' :
        'Stop requested. The daemon saves its checkpoint before it exits.';
    } catch {
      procMessage = 'server.control did not answer on port 8900. Start or stop the daemon from a terminal.';
    } finally {
      procPending = false; refresh();
    }
  }
  function render() {
    const count = data ? DaemonData.keys.filter(key => data.sources[key].enabled).length : 0;
    const off = !verified && proc?.running === false;  // server.control confirms no daemon process
    text('sources-summary', off ? 'Daemon off' : !verified ? 'Daemon not verified' : data.paused ? 'Paused · nothing shared' : count + ' of 5 sources shared');
    byId('sources-summary').dataset.state = off ? 'off' : !verified ? 'unknown' : data.paused ? 'off' : 'on';
    text('sources-connection', verified ?
      'Confirmed by the daemon · saved choice, revision ' + data.revision + ' · a restart never switches a source on.' :
      proc && !proc.running ? 'The daemon is off. Start it to see and change its sources. It starts with the saved choice; with none saved, everything is off.' :
      'Cannot verify the daemon. Capture may continue if a source was shared. These switches show only the last confirmed selection.');
    text('sources-message', message);
    byId('sources-stop').disabled = !data || (verified && !count && !pending);
    for (const key of DaemonData.keys) {
      const source = data?.sources[key], toggle = byId('source-' + key);
      toggle.checked = source?.enabled ?? false;
      // Unverified: a shared source can still be switched off, never on.
      toggle.disabled = !data || (!verified && !source?.enabled);
      toggle.setAttribute('aria-disabled', String(pending || toggle.disabled));
      text('source-state-' + key, source ? (verified ? label[source.status] : 'Unverified · last selection ' + (source.enabled ? 'on' : 'off')) : 'Status not loaded');
    }
    text('daemon-state', !proc ? 'Start and stop unavailable · server.control is not answering on 8900' :
      proc.running ? 'Daemon running · pid ' + proc.pid : 'Daemon off');
    byId('daemon-state').dataset.state = !proc ? 'unknown' : proc.running ? 'on' : 'off';
    byId('daemon-start').disabled = procPending || !proc || proc.running;
    byId('daemon-stop').disabled = procPending || !proc || !proc.running;
    text('daemon-message', procMessage);
  }
  function refresh() {
    clearTimeout(timer); epoch++; reader?.abort();
    if (active()) read(epoch);
  }
  for (const key of DaemonData.keys) byId('source-' + key).addEventListener('change', event => {
    if (pending) { render(); return; }
    return change({ [key]: event.target.checked });
  });
  byId('sources-stop').addEventListener('click', () => change(Object.fromEntries(DaemonData.keys.map(key => [key, false]))));
  byId('sources-refresh').addEventListener('click', refresh);
  byId('daemon-start').addEventListener('click', () => processAction('start'));
  byId('daemon-stop').addEventListener('click', () => processAction('stop'));
  document.addEventListener('visibilitychange', refresh);
  window.addEventListener('pagehide', () => { clearTimeout(timer); epoch++; reader?.abort(); });
  render();
  return { refresh };
})();
