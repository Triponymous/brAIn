import { readFileSync } from 'node:fs';
import { createContext, runInContext, Script } from 'node:vm';
import { webcrypto, createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import test from 'node:test';

const read = name => readFileSync(new URL(name, import.meta.url), 'utf8');
const catalog = read('observatory-data.js');
const telemetry = read('demo-telemetry.js');

function fixture() {
  const context = createContext({ structuredClone });
  runInContext(catalog + '\n' + telemetry, context);
  return { context, ...runInContext('({nodes,edges,scenarios,state,vocabulary,events,DemoTelemetry})', context) };
}

test('all local application scripts parse', () => {
  for (const file of ['observatory.js','observatory-data.js','demo-telemetry.js','observatory-session.js','brain-view.js']) new Script(read(file));
  const html = read('observatory.html');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length, 'IDs must be unique across the shared workspace');
  assert(!html.includes('match-arc'), 'Do not use a progress ring for uncalibrated similarity');
  assert(html.includes('aria-labelledby="teach-title"'));
  assert(html.includes('href="#methods" aria-label="Methods"'));
  assert(html.includes('data-action="export" aria-label="Export"'));
});

test('source defaults, graph catalog and displayed architecture agree', () => {
  const { nodes, edges } = fixture();
  assert.equal(nodes.length,1000);assert.equal(edges.length,3386);
  assert.equal(new Set(edges.map(edge => edge.from+'>'+edge.to)).size, edges.length);
  const core = read('../../brain/core.py');
  for(const [name,count] of [['sensory',200],['expansion',500],['concept',200],['wm',100]]) {
    assert(new RegExp('num_'+name+': int = '+count).test(core));
  }
  assert.equal(500*200+200*100+100*200,140000);
});

test('800,000 procedural samples have bounded traces, binary outputs and exact event counts', () => {
  const { nodes, scenarios, DemoTelemetry: demo } = fixture();
  for (const scenario of Object.keys(scenarios)) for (const node of nodes) {
    const trace = demo.series(scenario,node.id);let total=0;
    assert.equal(trace.voltage.length,200);
    for(let frame=0;frame<200;frame++) {
      assert(Number.isFinite(trace.voltage[frame]) && trace.voltage[frame]>=0 && trace.voltage[frame]<=1);
      assert(trace.output[frame]===0||trace.output[frame]===1);
      total+=trace.output[frame];assert.equal(trace.cumulative[frame],total);
      if(node.key==='e') assert.equal(trace.voltage[frame],trace.output[frame]);
    }
  }
});

test('fixture is repeatable, contexts differ, glow matches the same unit series', () => {
  const first=fixture(),second=fixture();
  for(const scenario of Object.keys(first.scenarios)) {
    const a=first.DemoTelemetry.snapshotSeries(scenario,'c42'),b=second.DemoTelemetry.snapshotSeries(scenario,'c42');
    assert.equal(JSON.stringify(a),JSON.stringify(b));
    const activity=first.DemoTelemetry.activity(scenario,42);
    for(const [index,node] of first.nodes.entries()) {
      const sample=first.DemoTelemetry.sample(scenario,node.id,42);
      assert(Math.abs(activity[index]-sample.glow)<1e-6);
      assert.equal(sample.output,first.DemoTelemetry.series(scenario,node.id).output[42]);
    }
  }
  assert.notEqual(JSON.stringify(first.DemoTelemetry.snapshotSeries('focus','c42')),JSON.stringify(first.DemoTelemetry.snapshotSeries('rest','c42')));
});

function sessionHarness(query='') {
  const f=fixture(),elements=new Map(),downloads=[],urls=[];
  const element=selector=>{
    if(!elements.has(selector))elements.set(selector,{checked:true,dataset:{},value:'',setAttribute(){}});
    return elements.get(selector);
  };
  class CaptureURL extends URL {
    static createObjectURL(blob){downloads.push(blob);return 'blob:test'}
    static revokeObjectURL(){}
  }
  Object.assign(f.context,{crypto:webcrypto,TextEncoder,Date,Blob,URL:CaptureURL,URLSearchParams,
    location:{search:query,href:'http://localhost/observatory.html'+query+'#network',pathname:'/observatory.html',origin:'http://localhost',hash:'#network'},
    history:{replaceState(...args){urls.push(String(args[2]))}},
    window:{addEventListener(){}},document:{createElement(){return {click(){}}}},
    $:element,$$:()=>[],setTimeout(){},notify(){},renderGraphControls(){}});
  runInContext('const current=()=>scenarios[state.scenario];\n'+read('observatory-session.js'),f.context);
  return {...f,elements,downloads,urls};
}

test('URL boundary accepts only known fixture fields and omits annotations', () => {
  const h=sessionHarness('?scenario=rest&unit=e499&region=e&frame=199&view=circuit&surface=false&connections=false&speed=1&note=private');
  runInContext('restoreViewURL();appReady=true;syncViewURL()',h.context);
  assert.equal(h.state.scenario,'rest');assert.equal(h.state.selected,'e499');
  assert.equal(h.state.frame,199);assert.equal(h.state.graphMode,'circuit');assert.equal(h.state.brainSurface,false);
  assert.equal(h.elements.get('#show-edges').checked,false);
  const shared=runInContext('viewState()',h.context);assert(!('note' in shared));
  assert(!h.urls[0].includes('private'));assert(!h.urls[0].includes('note='));
  const invalid=sessionHarness('?scenario=__proto__&unit=x42&region=z&frame=Infinity&speed=99&view=unknown');
  runInContext('restoreViewURL()',invalid.context);
  assert.equal(invalid.state.scenario,'focus');assert.equal(invalid.state.selected,'c42');assert.equal(invalid.state.frame,42);
  assert.equal(invalid.state.speed,.1);assert.equal(invalid.state.graphMode,'brain');
});

test('annotations preserve exact moments and before/after vocabulary references', () => {
  const h=sessionHarness();
  runInContext("const snap=freezeMoment();const old=structuredClone(vocabulary[0]);vocabulary[0].note='A revised note';recordAnnotation('Confirm','test',snap,'Flow',old);snap.frame=99;state.frame=18;vocabulary[0].note='Changed again';",h.context);
  assert.equal(h.events[0].snapshot.frame,42);
  assert.equal(h.events[0].reference.note,'A revised note');
  assert.equal(h.events[0].reference.corrections,h.events[0].previous.corrections+1);
  assert.equal(h.events[0].reference.sessionAnnotations,1);
  assert.notEqual(h.events[0].previous.note,'A revised note');
  assert.equal(h.state.revision,1);
  assert.notEqual(runInContext("slotKey('focus',42)",h.context),runInContext("slotKey('focus',43)",h.context));
});

test('snapshot contains matching series, annotations and verifiable SHA-256', async () => {
  const h=sessionHarness();
  runInContext("recordAnnotation('Confirm','test',freezeMoment(),'Flow',vocabulary[0]);",h.context);
  await runInContext('exportSession()',h.context);
  assert.equal(h.downloads.length,1);
  const snapshot=JSON.parse(await h.downloads[0].text());
  assert.equal(snapshot.schema,'brain.observatory.snapshot.v3');assert.equal(snapshot.synthetic,true);
  assert.equal(snapshot.selected_unit_series.unit,snapshot.view.unit);
  assert.equal(snapshot.selected_unit_series.output.length,200);
  assert.equal(snapshot.annotations[0].snapshot.frame,42);
  assert.equal(snapshot.dataset_sha256,createHash('sha256').update(JSON.stringify(snapshot.dataset)).digest('hex'));
  assert.equal(h.state.exportedRevision,1);
});
