import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

const read = file => readFileSync(new URL(file, import.meta.url), 'utf8');
const keys = ['keystroke_rate', 'mouse_rate', 'idle', 'active_app', 'mic'];
const consent = (revision = 0, shared = []) => ({ schema: 'brain.consent.v1', revision, paused: !shared.length,
  sources: Object.fromEntries(keys.map(key => [key, { enabled: shared.includes(key), changed_at: shared.includes(key) ? 1.5 : null,
    status: shared.includes(key) ? 'available' : 'disabled' }])) });
const feel = (over = {}) => ({ recognized: null, confidence: 0, signature: null, known_labels: [], pending_ask: null, paused: false, ...over });

function harness({ liveSource = 'daemon', consentReply = consent(), statusReply = { running: true, pid: 42 }, feelReply = feel() } = {}) {
  const elements = new Map(), requests = [];
  const node = () => ({ textContent: '', disabled: false, checked: false, hidden: false, value: '', dataset: {}, handlers: {}, attributes: {}, children: [],
    setAttribute(name, value) { this.attributes[name] = value; },
    addEventListener(event, callback) { this.handlers[event] = callback; },
    replaceChildren(...children) { this.children = children; },
    fire(event = 'click', extra = {}) { return this.handlers[event]({ target: this, preventDefault() {}, ...extra }); } });
  const element = id => { if (!elements.has(id)) elements.set(id, node()); return elements.get(id); };
  const replies = { consent: consentReply, status: statusReply, feel: feelReply };
  let responder = async (url, options) => {
    const reply = url.endsWith('/api/consent') ? replies.consent : url.endsWith('/daemon/status') ? replies.status :
      url.endsWith('/api/feel') ? replies.feel : {};
    return { ok: true, status: 200, json: async () => structuredClone(reply) };
  };
  const context = vm.createContext({ AbortController, Event, structuredClone, Date,
    setTimeout: () => 1, clearTimeout: () => {}, state: { page: 'live' },
    document: { hidden: false, body: { dataset: { liveSource } }, getElementById: element, createElement: node,
      addEventListener: () => {}, dispatchEvent: () => {} },
    window: { addEventListener: () => {} },
    fetch: (url, options = {}) => { requests.push({ url, ...options }); return responder(url, options); } });
  vm.runInContext(read('daemon-data.js') + '\n' + read('daemon-controls.js') + '\n' + read('felt-panel.js') +
    ';globalThis.controls=DaemonControls;globalThis.felt=FeltPanel;globalThis.contract=DaemonData;', context);
  return { element, requests, replies, controls: context.controls, felt: context.felt, contract: context.contract,
    respond: fn => responder = fn, TypeError: vm.runInContext('TypeError', context) };  // a failed fetch rejects with the page's own TypeError
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('daemon scripts parse and the contract rejects contradictions', () => {
  for (const file of ['daemon-data.js', 'daemon-controls.js', 'felt-panel.js']) new vm.Script(read(file));
  const { contract } = harness();
  assert.equal(contract.consent(consent(3, ['idle'])).revision, 3);
  const broken = [data => data.sources.mic.status = 'available', data => data.sources.idle.enabled = true,
    data => data.paused = false, data => delete data.sources.mic, data => data.sources.screen = data.sources.mic,
    data => data.revision = -1, data => data.schema = 'brain.capture.v1', data => data.sources.idle.changed_at = 'now'];
  for (const mutate of broken) { const data = consent(); mutate(data); assert.throws(() => contract.consent(data)); }
  assert.throws(() => contract.change(consent(), { screen: true }));
  assert.throws(() => contract.change(consent(), { mic: 'true' }));
  assert.throws(() => contract.feel(feel({ recognized: 'flow' })));                       // not a taught word
  assert.throws(() => contract.feel(feel({ recognized: 'flow', known_labels: ['flow'], paused: true })));
  assert.throws(() => contract.feel(feel({ confidence: 1.5 })));
  assert.throws(() => contract.daemon({ running: true, pid: null }));
});

test('opening the page reads state but never shares a source', async () => {
  const h = harness(); h.controls.refresh(); await settle();
  assert.equal(h.element('sources-summary').textContent, 'Paused · nothing shared');
  assert.equal(h.element('source-mic').checked, false);
  assert.equal(h.element('source-mic').disabled, false);
  assert.equal(h.element('sources-stop').disabled, true);
  assert.equal(h.element('daemon-state').textContent, 'Daemon running · pid 42');
  assert.equal(h.element('daemon-start').disabled, true);
  assert(h.requests.every(request => !request.method && request.credentials === 'omit'));
});

test('the session runner source leaves the daemon unpolled', async () => {
  const h = harness({ liveSource: 'runner' }); h.controls.refresh(); h.felt.refresh(); await settle();
  assert.equal(h.requests.length, 0);
});

test('a stop waits for the daemon instead of claiming an optimistic off state', async () => {
  const h = harness({ consentReply: consent(1, ['mic']) }); h.controls.refresh(); await settle();
  let release;
  h.respond((url, options) => new Promise(resolve => { release = () => resolve({ ok: true, status: 200, json: async () => consent(2) }); }));
  const toggle = h.element('source-mic'); toggle.checked = false; toggle.fire('change'); await settle();
  assert.equal(toggle.checked, true);                                                    // still the confirmed state
  assert.equal(toggle.attributes['aria-disabled'], 'true');
  assert.match(h.element('sources-message').textContent, /waiting for the daemon to confirm/);
  const post = h.requests.find(request => request.method === 'POST');
  assert.equal(post.url, 'http://127.0.0.1:8000/api/consent');
  assert.deepEqual(JSON.parse(post.body), { revision: 1, enabled: { mic: false } });
  release(); await settle(); await settle();
  assert.equal(toggle.checked, false);
  assert.equal(h.element('sources-summary').textContent, 'Paused · nothing shared');
  assert.match(h.element('sources-message').textContent, /What it learned stays saved/);
});

test('a conflict never flips a switch on, and an unsaved stop is reported as the daemon said', async () => {
  const h = harness(); h.controls.refresh(); await settle();
  h.respond(async () => ({ ok: false, status: 409, json: async () => ({ detail: 'Consent changed elsewhere' }) }));
  const toggle = h.element('source-idle'); toggle.checked = true; toggle.fire('change'); await settle(); await settle();
  assert.equal(toggle.checked, false);
  assert.match(h.element('sources-message').textContent, /Changed elsewhere/);
  const saved = harness({ consentReply: consent(4, ['mic']) }); saved.controls.refresh(); await settle();
  saved.respond(async () => ({ ok: false, status: 500, json: async () => ({ detail: 'Could not save the choice. What you switched off is off now, but a restart would bring back the last saved choice.' }) }));
  const mic = saved.element('source-mic'); mic.checked = false; mic.fire('change'); await settle(); await settle();
  assert.match(saved.element('sources-message').textContent, /restart would bring back/);
});

test('an unreachable daemon is unverified; a daemon that is off says how to start it', async () => {
  const h = harness({ statusReply: { running: false, pid: null } });
  h.respond(async url => url.endsWith('/daemon/status') ? { ok: true, status: 200, json: async () => ({ running: false, pid: null }) } :
    Promise.reject(new TypeError('Failed to fetch')));
  h.controls.refresh(); await settle(); await settle();
  assert.equal(h.element('sources-summary').textContent, 'Daemon off');
  assert.match(h.element('sources-connection').textContent, /The daemon is off. Start it/);
  assert.equal(h.element('source-idle').disabled, true);
  assert.equal(h.element('daemon-start').disabled, false);
  h.element('daemon-start').fire(); await settle();
  const start = h.requests.find(request => request.method === 'POST');
  assert.equal(start.url, 'http://127.0.0.1:8900/daemon/start');
  const unknown = harness(); unknown.respond(async () => Promise.reject(new TypeError('Failed to fetch')));
  unknown.controls.refresh(); await settle(); await settle();
  assert.equal(unknown.element('sources-summary').textContent, 'Daemon not verified');
  assert.match(unknown.element('sources-connection').textContent, /Capture may continue/);
});

test('named states: paused shows no current state, an open ask can still be answered', async () => {
  const h = harness({ feelReply: feel({ paused: true, known_labels: ['focused'], pending_ask: { at: 1758870000 } }) });
  h.felt.refresh(); await settle();
  assert.equal(h.element('felt-current').textContent, 'Paused · nothing shared');
  assert.equal(h.element('felt-ask').hidden, false);
  assert.equal(h.element('felt-teach').disabled, false);
  assert.equal(h.element('felt-teach').textContent, 'Name that moment');
  assert.equal(h.element('felt-words').children.length, 1);
  h.element('felt-label').value = ' tired ';
  h.element('felt-teach').fire(); await settle();
  const post = h.requests.find(request => request.method === 'POST');
  assert.equal(post.url, 'http://127.0.0.1:8000/api/feel');
  assert.deepEqual(JSON.parse(post.body), { label: 'tired' });
  await settle();
  assert.match(h.element('felt-message').textContent, /names the moment the brain asked about/);
  assert.equal(h.element('felt-label').value, '');
});

test('named states: a word the daemon did not confirm is kept for another try', async () => {
  const h = harness({ feelReply: feel({ known_labels: ['focused'] }) });
  h.felt.refresh(); await settle();
  h.respond(async (_url, options) => options.method === 'POST' ? Promise.reject(new h.TypeError('Failed to fetch')) :
    { ok: true, status: 200, json: async () => feel({ known_labels: ['focused'] }) });
  h.element('felt-label').value = 'tired';
  await h.element('felt-teach').fire(); await settle();
  assert.equal(h.element('felt-label').value, 'tired');
  assert.match(h.element('felt-message').textContent, /No confirmation from the daemon/);
  h.respond(async (_url, options) => options.method === 'POST' ? { ok: false, status: 422, json: async () => ({ detail: [] }) } :
    { ok: true, status: 200, json: async () => feel({ known_labels: ['focused'] }) });
  await h.element('felt-teach').fire(); await settle();
  assert.equal(h.element('felt-label').value, 'tired');
  assert.equal(h.element('felt-message').textContent, 'The daemon did not confirm this.');
});

test('named states: a recognized word is a similarity, and live naming needs a shared source', async () => {
  const h = harness({ feelReply: feel({ recognized: 'focused', confidence: .72, known_labels: ['focused', 'tired'] }) });
  h.felt.refresh(); await settle();
  assert.equal(h.element('felt-current').textContent, 'focused');
  assert.match(h.element('felt-similarity').textContent, /Similarity 0.72 .* not a calibrated probability/);
  assert.equal(h.element('felt-ask').hidden, true);
  const first = h.element('felt-words').children[0];
  h.felt.refresh(); await settle();
  assert.equal(h.element('felt-words').children[0], first);                               // not rebuilt: focus survives
  const paused = harness({ feelReply: feel({ paused: true, known_labels: ['focused'] }) }); paused.felt.refresh(); await settle();
  assert.equal(paused.element('felt-teach').disabled, true);
  assert.equal(paused.element('felt-words').children[0].disabled, true);
});

test('actual daemon telemetry from Python passes the browser contract', () => {
  const python = `import json, torch, tempfile, pathlib
from brain.core import Brain
from adapters.mac_desktop.adapter import MacDesktopAdapter
from server.consent import ConsentStore
from server.daemon_telemetry import DaemonTelemetry
torch.set_num_threads(1); torch.manual_seed(17)
brain=Brain(); adapter=MacDesktopAdapter(mock_mode=True, enabled={"idle": True, "active_app": True})
adapter.bus.write("idle", {"seconds": 3.0}); adapter.bus.write("active_app", {"name": "Safari"})
d=pathlib.Path(tempfile.mkdtemp())
t=DaemonTelemetry(brain, adapter, checkpoint=d/"b.sqlite", loaded=False, consent=ConsentStore(d/"consent.json"))
t.read()
for _ in range(50): t.record(brain, brain.tick(adapter.encode()))
print(json.dumps(t.read(), allow_nan=False))`;
  const data = JSON.parse(execFileSync('.venv/bin/python', ['-c', python], {
    cwd: new URL('../../', import.meta.url), encoding: 'utf8', maxBuffer: 8_000_000 }));
  const context = vm.createContext({ structuredClone });
  vm.runInContext(read('capture-data.js') + '\n' + read('live-data.js') + ';globalThis.api=LiveData', context);
  const valid = context.api.validate(data);
  assert.equal(valid.frames.length, 10);                                                  // every 5th of 50 ticks
  assert.equal(context.api.activity(valid.frames.at(-1)).length, 1000);
  assert.equal(valid.source.runner, 'braind');
  assert.equal(valid.frames.at(-1).sensors.active_app.value, 'browser');                  // category, not "Safari"
  assert.equal(valid.frames.at(-1).sensors.keystroke_rate.status, 'disabled');
  assert.equal(valid.frames.at(-1).sensors.microphone.status, 'disabled');
});

test('the source choice is two radio buttons; the page that carries it never takes their keys or clicks', () => {
  let focused = null;
  const fake = name => ({ name, dataset: {}, attributes: {}, handlers: {}, textContent: '', style: {},
    setAttribute(key, value) { this.attributes[key] = String(value); }, getAttribute(key) { return this.attributes[key] ?? null; },
    addEventListener(event, callback) { (this.handlers[event] ??= []).push(callback); }, focus() { focused = this; } });
  const body = fake('body'); body.dataset.liveSource = 'daemon';
  const buttons = ['daemon', 'runner'].map(key => Object.assign(fake(key), { dataset: { liveSource: key } }));
  const elements = new Map(), stored = new Map();
  const $ = selector => { if (!elements.has(selector)) elements.set(selector, fake(selector)); return elements.get(selector); };
  const context = vm.createContext({ $, state: { page: 'overview' }, reduced: { matches: false, addEventListener() {} },
    setTimeout: () => 1, clearTimeout: () => {},
    CaptureControls: { refresh() {} }, DaemonControls: { refresh() {} }, FeltPanel: { refresh() {} },
    localStorage: { getItem: key => stored.get(key) ?? null, setItem: (key, value) => stored.set(key, value) },
    window: { addEventListener() {} },
    document: { body, hidden: false, addEventListener() {}, getElementById: id => $('#' + id),
      // As in a browser: the bare attribute selector also matches <body>, which carries the current choice.
      querySelectorAll: selector => selector === 'button[data-live-source]' ? buttons :
        selector === '[data-live-source]' ? [body, ...buttons] : [] } });
  vm.runInContext(readFileSync(new URL('./live-workspace.js', import.meta.url), 'utf8') + ';globalThis.api=LiveWorkspace;', context);
  assert.deepEqual(Object.keys(body.handlers), []);
  assert.equal(body.tabIndex, undefined);
  assert.equal(body.attributes['aria-checked'], undefined);
  assert.deepEqual(buttons.map(b => [b.attributes['aria-checked'], b.tabIndex]), [['true', 0], ['false', -1]]);
  let prevented = false;
  buttons[0].handlers.keydown[0]({ key: 'ArrowRight', preventDefault: () => { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(context.api.source(), 'runner');
  assert.equal(focused, buttons[1]);
  assert.equal(body.dataset.liveSource, 'runner');
  assert.equal(stored.get('brain.observatory.live-source'), 'runner');
  assert.deepEqual(buttons.map(b => [b.attributes['aria-checked'], b.tabIndex]), [['false', -1], ['true', 0]]);
  buttons[1].handlers.keydown[0]({ key: 'Enter', preventDefault: () => assert.fail('only arrow keys move the choice') });
  assert.equal(context.api.source(), 'runner');
});
