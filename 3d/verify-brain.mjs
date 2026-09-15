import { readFileSync, existsSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { runInNewContext, Script } from 'node:vm';
import assert from 'node:assert/strict';

const base = new URL('../docs/dashboard-concepts/', import.meta.url);
registerHooks({ resolve(specifier, context, nextResolve) {
  return specifier === 'three' ? { url: new URL('vendor/three/three.module.min.js', base).href, shortCircuit: true }
    : nextResolve(specifier, context);
} });
const T = await import('three');
const { neuralLayout, edgeCurve, fibers, neuronCloud, colorNeurons } = await import(new URL('brain-3d/network.js', base));
const { projectLandmarks } = await import(new URL('brain-3d/anatomy.js', base));
const html = readFileSync(new URL('observatory.html', base), 'utf8');
for (const match of html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)) {
  if (match[0].includes('importmap')) JSON.parse(match[1]); else new Script(match[1]);
}
const dataScript = readFileSync(new URL('observatory-data.js', base), 'utf8');
const telemetryScript = readFileSync(new URL('demo-telemetry.js', base), 'utf8');
const { nodes, edges, activity } = runInNewContext(dataScript + '\n' + telemetryScript + '\n({nodes,edges,activity:DemoTelemetry.activity("focus",42)})');
assert.equal(nodes.length, 1000);
assert.equal(new Set(nodes.map(n => n.id)).size, 1000);
assert(edges.every(e => nodes.some(n => n.id === e.from) && nodes.some(n => n.id === e.to)));

const glb = readFileSync(new URL('assets/brain-contours.glb', base));
assert.equal(glb.readUInt32LE(0), 0x46546c67);
assert.equal(glb.readUInt32LE(8), glb.length);
const size = glb.readUInt32LE(12), gltf = JSON.parse(glb.subarray(20, 20 + size));
const a = gltf.accessors[2], v = gltf.bufferViews[a.bufferView];
const sampleData = new Float32Array(glb.buffer, glb.byteOffset + 28 + size + v.byteOffset, a.count * 3);
const samples = new T.BufferAttribute(sampleData, 3);
const positions = neuralLayout(nodes, samples), repeated = neuralLayout(nodes, samples);
const atlasSamples = new Set(Array.from({ length: samples.count }, (_, i) => new T.Vector3().fromBufferAttribute(samples, i).toArray().join(',')));
const bounds = new T.Box3().setFromPoints([...positions.values()]).getSize(new T.Vector3());
assert(bounds.x > 2 && bounds.y > 1 && bounds.z > 1, 'The layout must have real depth');
for (const n of nodes) {
  assert(positions.get(n.id).toArray().every(Number.isFinite));
  assert.deepEqual(positions.get(n.id).toArray(), repeated.get(n.id).toArray());
  assert(atlasSamples.has(positions.get(n.id).toArray().join(',')), 'Every unit remains on an actual atlas surface');
}
assert.equal(new Set([...positions.values()].map(p => p.toArray().join(','))).size, nodes.length, 'No coincident unit positions');
const atlas = gltf.nodes.find(n => n.name === 'samples').extras;
assert.equal(atlas.regions.reduce((sum, r) => sum + r.count, 0), samples.count);
assert.equal(atlas.landmarks.length, 12);
for (const side of ['l', 'r']) {
  const anchor = id => atlas.landmarks.find(p => p.id === id && p.hemisphere === side).position;
  assert(anchor('frontal')[0] < anchor('occipital')[0], 'Anterior is -X, posterior is +X');
  assert(anchor('parietal')[1] > anchor('temporal')[1], 'Parietal anchor is superior to temporal');
  assert(anchor('brainstem')[1] < anchor('frontal')[1]);
  assert(anchor('cerebellum')[1] < anchor('temporal')[1]);
  assert(atlas.landmarks.filter(p => p.hemisphere === side).every(p => (p.position[2] > 0) === (side === 'l')));
}
for (const [width, height] of [[920, 608], [560, 545], [300, 387], [256, 330]]) {
  const camera = new T.PerspectiveCamera(35, width / height, .1, 100);
  for (const direction of [[2.3, 1.25, 7.5], [0, .2, 8], [-8, .2, 0], [0, 8, .001], [0, .2, -8]]) {
    camera.position.fromArray(direction).setLength(Math.max(width < 500 ? 8 : 7.3, 5.2 / camera.aspect));
    camera.lookAt(0, .03, 0); camera.updateMatrixWorld();
    const labels = projectLandmarks(atlas.landmarks, camera, width, height);
    assert.equal(labels.length, 6, `Six atlas landmarks at ${width}x${height}, camera ${direction}`);
    assert(labels.every(p => p.labelX >= 0 && p.labelX <= width && p.labelY >= 100 && p.labelY <= height - 40));
    for (const side of [0, 1]) {
      const ys = labels.filter(p => p.side === side).map(p => p.labelY).sort((a, b) => a - b);
      assert(ys.every(y => y >= (side ? 160 : 120)), 'Labels clear camera and Atlas controls');
      assert(ys.every((y, i) => !i || y - ys[i - 1] >= 23.9), 'No same-side label collisions');
    }
  }
}
for (const e of edges) {
  const curve = edgeCurve(e, positions);
  assert(curve.getPoint(0).distanceTo(positions.get(e.from)) < 1e-6);
  assert(curve.getPoint(1).distanceTo(positions.get(e.to)) < 1e-6);
}
const shown = edges.filter((_, i) => i % 5 === 0), geometry = fibers(shown, positions);
assert.equal(geometry.attributes.position.count, shown.length * 28 * 2);
assert.equal(geometry.attributes.position.count, geometry.attributes.aProgress.count);
const css = readFileSync(new URL('observatory.css', base), 'utf8');
const palette = Object.fromEntries(['s', 'e', 'c', 'w'].map(key => [key, css.match(new RegExp('--' + key + ':(#[0-9a-f]{6})'))[1]]));
const cloud = neuronCloud(nodes, positions, palette);
colorNeurons(cloud, nodes, { region: 'e', frame: 42, selected: 'e42', activity });
assert([...cloud.geometry.attributes.color.array].every(Number.isFinite));
assert(cloud.geometry.attributes.aSize.getX(242) > cloud.geometry.attributes.aSize.getX(0));
const originalColors = Array.from(cloud.geometry.attributes.color.array);
for (const region of ['all', 's', 'e', 'c', 'w']) {
  for (const selected of ['s42', 'e42', 'c42', 'w42']) {
    for (const output of [activity, new Uint8Array(nodes.length), new Uint8Array(nodes.length).fill(1)]) {
      colorNeurons(cloud, nodes, { region, selected, activity: output });
      assert.deepEqual(Array.from(cloud.geometry.attributes.color.array), originalColors, 'Role colors survive activity, filtering and selection');
    }
  }
}
nodes.forEach((n, i) => assert.equal(new T.Color().fromBufferAttribute(cloud.geometry.attributes.color, i).getHexString(), palette[n.key].slice(1), 'Exact Circuit CSS color after sRGB round-trip'));
assert.equal(cloud.material.toneMapped, false);
assert.equal(cloud.material.blending, T.NormalBlending);
assert.equal(cloud.renderOrder, 2);
colorNeurons(cloud, nodes, { region: 'all', selected: 'c42', activity: new Uint8Array(nodes.length) });
assert(Math.abs(cloud.geometry.attributes.aOpacity.getX(742) - .42) < 1e-6, 'Selecting an inactive unit must not invent activity');
assert(!cloud.material.fragmentShader.includes('tonemapping_fragment'));
assert(cloud.material.fragmentShader.includes('colorspace_fragment'));
for (const match of html.matchAll(/(?:src|href)="([^"#]+)"/g)) {
  if (!match[1].includes(':')) assert(existsSync(new URL(match[1], base)), 'Missing local asset: ' + match[1]);
}
geometry.dispose(); cloud.geometry.dispose(); cloud.material.dispose();
console.log(JSON.stringify({ result: 'PASS', units: nodes.length, sampleCatalog: edges.length,
  visiblePaths: shown.length, contourSegments: (gltf.accessors[0].count + gltf.accessors[1].count) / 2,
  anatomyBytes: glb.length, spatialExtent: bounds.toArray().map(n => +n.toFixed(2)),
  checks: ['inline syntax', 'import map', 'local assets', '1000 unique units', 'deterministic 3D positions',
    'all edge endpoints', 'fiber attributes', 'exact shared Circuit palette', 'no activity hue shift',
    '1000 unique atlas-surface positions', '12 source landmarks', 'anatomical axes', '20 atlas projection layouts'] }, null, 2));
