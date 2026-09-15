import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';

const source = ['capture-data.js', 'live-data.js'].map(file => readFileSync(new URL(file, import.meta.url), 'utf8')).join('\n');
const context = vm.createContext({ structuredClone });
vm.runInContext(source + ';globalThis.api=LiveData', context);
const api = context.api;
const fixture = (ticks = [1, 2]) => ({
  schema: 'brain.telemetry.v1', session_id: 'test-session', status: 'streaming', age_s: 0,
  source: { input_kind: 'test_fixture' }, architecture: { s: 2, e: 3, c: 2, w: 1 }, reset: false, gap: false,
  frames: ticks.map(tick => ({ tick, elapsed_s: tick / 97, captured_at: '2026-09-14T12:00:00Z',
    modulators: { DA: .1, NE: .2, ACh: .3, '5HT': .4 },
    regions: Object.fromEntries(Object.entries({ s: 2, e: 3, c: 2, w: 1 }).map(([key, count]) =>
      [key, { output: Array.from({ length: count }, (_, i) => (tick + i) % 2),
        membrane: key === 'e' ? null : Array.from({ length: count }, (_, i) => i - 2.5) }])),
    sensors: Object.fromEntries(['keystroke_rate', 'mouse_rate', 'idle', 'active_app', 'microphone', 'wearable']
      .map(key => [key, { status: 'unavailable', value: null }])) }))
});

test('live scripts parse and real-data vectors preserve negative membrane values', () => {
  for (const file of ['live-workspace.js', 'live-data.js', 'capture-data.js', 'capture-controls.js']) new vm.Script(readFileSync(new URL(file, import.meta.url), 'utf8'));
  const data = api.validate(fixture());
  assert.equal(data.frames[0].regions.c.membrane[0], -2.5);
  assert.deepEqual(Array.from(api.activity(data.frames[0])), [1, 0, 1, 0, 1, 1, 0, 1]);
});
test('schema rejects bad arrays, fabricated expansion membranes and untrusted sources', () => {
  const mutations = [data => data.schema = 'legacy', data => data.frames[0].regions.e.membrane = [0, 0, 0],
    data => data.frames[0].regions.c.output = [1], data => data.frames[0].regions.c.output[0] = .5,
    data => data.frames[0].regions.c.membrane[0] = NaN, data => data.source.input_kind = 'mystery',
    data => data.frames[1].tick = 1, data => data.frames[1].elapsed_s = -3];
  for (const mutate of mutations) { const data = fixture(); mutate(data); assert.throws(() => api.validate(data)); }
});
test('missing sensors remain missing, never zero or a demo value', () => {
  const data = api.merge(null, fixture());
  assert.equal(data.frames[0].sensors.keystroke_rate.value, null);
  assert.equal(data.frames[0].sensors.keystroke_rate.status, 'unavailable');
});
test('a session cannot silently change architecture or input provenance', () => {
  const previous = api.merge(null, fixture());
  const next = fixture([]); next.architecture.c = 3;
  assert.throws(() => api.merge(previous, next), /Source changed/);
  const switched = fixture([]); switched.source.input_kind = 'desktop_metadata';
  assert.throws(() => api.merge(previous, switched), /Source changed/);
});
test('merge deduplicates, resets sessions, exposes gaps and bounds history', () => {
  let store = api.merge(null, fixture());
  store = api.merge(store, fixture([2, 3]));
  assert.deepEqual(Array.from(store.frames, f => f.tick), [1, 2, 3]);
  const restart = fixture(); restart.session_id = 'new';
  store = api.merge(store, restart); assert.equal(store.frames.length, 2);
  const gap = fixture([100, 101]); gap.gap = true;
  store = api.merge(store, gap); assert.equal(store.gaps, 1); assert.equal(store.frames[0].tick, 100);
  store = api.merge(store, fixture(Array.from({ length: 200 }, (_, i) => i + 102)));
  assert.equal(store.frames.length, 200);
});
test('export freezes exact observations and does not turn a test source into real inputs', () => {
  const store = api.merge(null, fixture()), snap = api.exportWindow(store, store.frames[0], 'c0');
  store.frames[0].regions.c.membrane[0] = 50;
  assert.equal(snap.frames[0].regions.c.membrane[0], -2.5);
  assert.equal(snap.provenance.source.input_kind, 'test_fixture');
  assert.equal(snap.selected_tick, 1);
  assert.equal(snap.frames[0].elapsed_s, 1 / 97);
  assert.doesNotThrow(() => JSON.parse(JSON.stringify(snap)));
});
test('the browser contract accepts actual Python Brain output end to end', () => {
  const python = `import json, torch
from brain.core import Brain
from server.telemetry import TelemetryBuffer
from tests.test_observation_telemetry import sensor_fixture
from adapters.mac_desktop.metadata import encode_metadata
torch.set_num_threads(1)
torch.manual_seed(17)
brain=Brain()
buffer=TelemetryBuffer(brain,input_kind="test_fixture",seed=17)
sensors=sensor_fixture()
for _ in range(25):
    output=brain.tick(encode_metadata(sensors))
    buffer.record(brain,output,sensors)
print(json.dumps(buffer.read(),allow_nan=False))`;
  const data = JSON.parse(execFileSync('.venv/bin/python', ['-c', python], {
    cwd: new URL('../../', import.meta.url), encoding: 'utf8', maxBuffer: 8_000_000
  }));
  const valid = api.validate(data);
  assert.equal(valid.frames.length, 25);
  assert.equal(valid.frames.at(-1).tick, 25);
  assert.equal(api.activity(valid.frames.at(-1)).length, 1000);
  assert.equal(valid.source.input_kind, 'test_fixture');
  assert.equal(valid.frames.at(-1).regions.e.membrane, null);
});
