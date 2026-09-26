/* Words the user taught the persistent brain. A match is a similarity, never a reading of feelings. */
const FeltPanel = (() => {
  const url = 'http://127.0.0.1:8000/api/feel';
  let data, verified = false, pending = false, message = '', timer, reader, epoch = 0, failures = 0, shownWords = null;
  const byId = id => document.getElementById(id);
  const text = (id, value) => { if (byId(id).textContent !== value) byId(id).textContent = value; };
  const active = () => document.body?.dataset?.liveSource === 'daemon' && state.page === 'live' && !document.hidden;

  async function read(generation) {
    if (generation !== epoch || !active()) return;
    if (!pending) {
      const controller = new AbortController(); reader = controller;
      const timeout = setTimeout(() => controller.abort(), 4000);
      try {
        const response = await fetch(url, { signal: controller.signal, cache: 'no-store', credentials: 'omit', redirect: 'error' });
        if (!response.ok) throw Error('HTTP ' + response.status);
        const next = DaemonData.feel(await response.json());
        if (generation !== epoch) return;
        data = next; verified = true; failures = 0;
      } catch {
        if (generation !== epoch) return;
        verified = false; failures++;
      } finally {
        clearTimeout(timeout);
        if (generation === epoch) render();
      }
    }
    if (generation === epoch) timer = setTimeout(() => read(generation), Math.min(8000, 3000 * 2 ** Math.min(failures, 1)));
  }
  async function post(path, body, confirmed) {
    if (pending) return false;
    pending = true; message = 'Waiting for the daemon to confirm…'; render();
    const controller = new AbortController(), timeout = setTimeout(() => controller.abort(), 4000);
    try {
      const response = await fetch(url + path, { method: 'POST', signal: controller.signal, cache: 'no-store', credentials: 'omit',
        redirect: 'error', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
      const reply = await response.json().catch(() => ({}));
      if (!response.ok) throw Error(typeof reply.detail === 'string' ? reply.detail : 'The daemon did not confirm this.');
      message = confirmed;
      return true;
    } catch (error) {
      // Unanswered is not the same as not received: the list below is the daemon's own word.
      message = error instanceof TypeError || error.name === 'AbortError' ?
        'No confirmation from the daemon. The words below are what it last confirmed.' : error.message;
      return false;
    } finally {
      clearTimeout(timeout); pending = false; refresh();
    }
  }
  async function teach(word) {
    const label = String(word).trim(), input = byId('felt-label');
    if (!label || label.length > 32) { message = 'A word needs 1 to 32 characters.'; render(); return; }
    const answering = !!data?.pending_ask;
    const taught = await post('', { label }, answering ? '“' + label + '” now names the moment the brain asked about.' :
      '“' + label + '” now names this moment. Whether it is recognized later needs new observations.');
    if (taught && input.value.trim() === label) input.value = '';  // kept when it failed, or if a new word was typed meanwhile
  }
  function render() {
    const paused = verified && data.paused, ask = verified ? data.pending_ask : null;
    const current = !verified ? '—' : paused ? 'Paused · nothing shared' : data.recognized ?? 'Not yet named';
    text('felt-current', current);
    byId('felt-current').dataset.state = verified && !paused && data.recognized ? 'named' : 'unknown';
    text('felt-similarity', !verified ? 'Cannot reach the daemon.' : paused ? 'While nothing is shared there is no current state to recognize.' :
      data.recognized ? 'Similarity ' + data.confidence.toFixed(2) + ' to the moments you named · not a calibrated probability' :
      'No taught word is close enough. Unknown is a regular state.');
    byId('felt-ask').hidden = !ask;
    text('felt-ask-time', ask ? new Date(ask.at * 1000).toLocaleTimeString('en-GB', { hour12: false }) : '—');
    // Naming needs a moment: the current one while sources are shared, or the one the brain asked about.
    const canName = verified && !pending && (!paused || !!ask);
    byId('felt-label').disabled = !verified || (paused && !ask);
    byId('felt-teach').disabled = !canName;
    text('felt-teach', ask ? 'Name that moment' : 'Name this moment');
    byId('felt-ask-later').disabled = pending;
    const words = verified ? data.known_labels : [];
    if (shownWords !== words.join('\n')) {  // rebuilt only on change, so focus survives polling
      shownWords = words.join('\n');
      byId('felt-words').replaceChildren(...words.map(word => {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'btn small'; button.textContent = word; button.dataset.word = word;
        button.addEventListener('click', () => teach(word));
        return button;
      }));
    }
    for (const button of byId('felt-words').children) {
      button.disabled = !canName;
      button.setAttribute('aria-label', (ask ? 'Name that moment “' : 'Confirm “') + button.dataset.word + (ask ? '”' : '” for this moment'));
    }
    byId('felt-words-empty').hidden = words.length > 0;
    text('felt-message', message);
  }
  function refresh() {
    clearTimeout(timer); epoch++; reader?.abort();
    if (active()) read(epoch);
  }
  byId('felt-teach').addEventListener('click', () => teach(byId('felt-label').value));
  byId('felt-label').addEventListener('keydown', event => { if (event.key === 'Enter' && !byId('felt-teach').disabled) teach(event.target.value); });
  byId('felt-ask-later').addEventListener('click', () => post('/dismiss', {}, 'Left unnamed. The brain may ask again at another moment.'));
  document.addEventListener('visibilitychange', refresh);
  window.addEventListener('pagehide', () => { clearTimeout(timer); epoch++; reader?.abort(); });
  render();
  return { refresh };
})();
