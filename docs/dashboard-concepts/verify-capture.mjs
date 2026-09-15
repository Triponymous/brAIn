import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

const contract = readFileSync(new URL('./capture-data.js', import.meta.url), 'utf8');
const controls = readFileSync(new URL('./capture-controls.js', import.meta.url), 'utf8');
const keys = ['keystroke_rate', 'mouse_rate', 'idle', 'active_app'];
const fixture = (revision = 0, enabled = false) => ({ schema: 'brain.capture.v1', session_id: 'capture-test',
  revision, armed: true, failed: false, scope: 'observation_runner', persistence: 'session_only',
  inputs: Object.fromEntries(keys.map(key => [key, { enabled: key === 'active_app' && enabled,
    status: key === 'active_app' && enabled ? 'available' : 'disabled' }])) });

function harness(initial = fixture()) {
  const elements = new Map(), requests = [];
  const element = id => {
    if (!elements.has(id)) elements.set(id, { textContent: '', disabled: false, checked: false, dataset: {}, handlers: {}, attributes: {},
      setAttribute(name, value) { this.attributes[name] = value; },
      addEventListener(event, callback) { this.handlers[event] = callback; },
      fire(event = 'click') { return this.handlers[event]({ target: this }); } });
    return elements.get(id);
  };
  let responder = async (_url, options) => ({ ok: true, status: 200, json: async () => structuredClone(initial) });
  const context = vm.createContext({ AbortController, Event,
    setTimeout: () => 1, clearTimeout: () => {}, state: { page: 'live' },
    document: { hidden: false, getElementById: element, addEventListener: () => {}, dispatchEvent: () => {} },
    window: { addEventListener: () => {} },
    fetch: (url, options) => { requests.push({ url, ...options }); return responder(url, options); } });
  vm.runInContext(contract + '\n' + controls + ';globalThis.api=CaptureControls;globalThis.contract=CaptureData;', context);
  return { element, requests, api: context.api, contract: context.contract, respond: fn => responder = fn };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('capture contract rejects contradictory states and unknown channels', () => {
  const { contract: api } = harness();
  assert.equal(api.validate(fixture()).revision, 0);
  for (const change of [d => d.armed = 'true', d => d.revision = -1, d => d.revision = 1.5,
    d => d.inputs.active_app.status = 'available', d => d.inputs.microphone = { enabled: true, status: 'available' },
    d => d.scope = 'whole_mac', d => d.session_id = '', d => d.persistence = 'disk']) {
    const data = fixture(); change(data); assert.throws(() => api.validate(data));
  }
  assert.throws(() => api.change(fixture(), { microphone: true }));
  assert.throws(() => api.change(fixture(), { active_app: 'false' }));
});

test('opening controls reads policy but never enables or samples anything', async () => {
  const h = harness(); h.api.refresh(); await settle();
  assert.equal(h.element('capture-summary').textContent, 'All inputs off');
  assert.equal(h.element('capture-active_app').checked, false);
  assert.equal(h.element('capture-active_app').disabled, false);
  assert.equal(h.element('capture-stop').disabled, true);
  assert(h.requests.every(request => !request.method));
  assert(h.requests.every(request => request.credentials === 'omit'));
});

test('stop waits for server acknowledgement instead of claiming an optimistic off state', async () => {
  const h = harness(fixture(1, true)); h.api.refresh(); await settle();
  let finish;
  h.respond(async (_url, options) => options.method === 'POST' ? new Promise(resolve => finish = resolve) :
    { ok: true, json: async () => fixture(1, true) });
  const pending = h.element('capture-stop').fire();
  assert.equal(h.element('capture-active_app').checked, true);
  assert.equal(h.element('capture-active_app').disabled, false);
  assert.equal(h.element('capture-active_app').attributes['aria-disabled'], 'true');
  assert.match(h.element('capture-message').textContent, /waiting for confirmation/);
  const request = h.requests.at(-1), body = JSON.parse(request.body);
  assert.equal(request.headers['X-Brain-Control'], 'capture-v1');
  assert.equal(body.session_id, 'capture-test'); assert.equal(body.revision, 1);
  assert.deepEqual(body.enabled, Object.fromEntries(keys.map(key => [key, false])));
  finish({ ok: true, json: async () => fixture(2) });
  await pending; await settle();
  assert.equal(h.element('capture-active_app').checked, false);
  assert.equal(h.element('capture-summary').textContent, 'All inputs off');
  assert.match(h.element('capture-message').textContent, /Earlier snapshots and learned state remain/);
});

test('failed stop warns that collection may continue and retains the last confirmed selection', async () => {
  const h = harness(fixture(1, true)); h.api.refresh(); await settle();
  h.respond(async () => { throw new TypeError('offline'); });
  await h.element('capture-stop').fire(); await settle();
  assert.equal(h.element('capture-active_app').checked, true);
  assert.equal(h.element('capture-stop').disabled, false);
  assert.equal(h.element('capture-summary').textContent, 'Capture status unverified');
  assert.match(h.element('capture-message').textContent, /Capture may still be running/);
  assert.equal(h.element('capture-keystroke_rate').disabled, true);
});

test('enable is explicit and a conflict never flips the confirmed switch on', async () => {
  const h = harness(); h.api.refresh(); await settle();
  h.respond(async (_url, options) => options.method === 'POST' ? { ok: false, status: 409 } :
    { ok: true, json: async () => fixture() });
  const toggle = h.element('capture-active_app'); toggle.checked = true;
  await toggle.fire('change'); await settle();
  assert.equal(toggle.checked, false);
  assert.match(h.element('capture-message').textContent, /Settings changed elsewhere/);
  assert.deepEqual(JSON.parse(h.requests.find(request => request.method === 'POST').body).enabled, { active_app: true });
});

test('macOS denial is distinct from off and cannot be mistaken for measured data', async () => {
  const data = fixture(); data.inputs.keystroke_rate = { enabled: true, status: 'permission_required' };
  const h = harness(data); h.api.refresh(); await settle();
  assert.equal(h.element('capture-keystroke_rate').checked, true);
  assert.equal(h.element('capture-state-keystroke_rate').textContent, 'On · macOS permission required');
  assert.equal(h.element('capture-stop').disabled, false);
});

test('unarmed and restarted sessions stay off; no enable request is replayed', async () => {
  const data = fixture(); data.armed = false;
  const h = harness(data); h.api.refresh(); await settle();
  assert.equal(h.element('capture-active_app').disabled, true);
  h.respond(async () => ({ ok: true, json: async () => ({ ...fixture(), session_id: 'restarted' }) }));
  h.api.refresh(); await settle();
  assert.equal(h.element('capture-active_app').checked, false);
  assert.equal(h.element('capture-summary').textContent, 'All inputs off');
  assert(h.requests.every(request => !request.method));
});

test('paused telemetry preserves the historical selection revision', () => {
  const context = vm.createContext({ structuredClone });
  vm.runInContext(contract + '\n' + readFileSync(new URL('./live-data.js', import.meta.url), 'utf8') + ';globalThis.api=LiveData;', context);
  const data = { schema: 'brain.telemetry.v1', session_id: 'capture-test', status: 'paused', age_s: 5,
    source: { input_kind: 'desktop_metadata' }, architecture: { s: 1, e: 1, c: 1, w: 1 },
    capture: fixture(2), frames: [{ tick: 1, elapsed_s: 1, captured_at: '2026-09-14T12:00:00Z', capture_revision: 1,
      modulators: { DA: 0, NE: 0, ACh: 0, '5HT': 0 },
      regions: Object.fromEntries(['s', 'e', 'c', 'w'].map(key => [key, { output: [0], membrane: key === 'e' ? null : [0] }])),
      sensors: Object.fromEntries([...keys, 'microphone', 'wearable'].map(key => [key, { status: 'unavailable', value: null }])) }] };
  assert.equal(context.api.validate(data).frames[0].capture_revision, 1);
  data.frames[0].capture_revision = 3;
  assert.throws(() => context.api.validate(data), /capture revision/);
});
