// Derivative anatomical geometry: CC BY-SA 4.0. See ASSETS.md.
// Converts the source atlas into unlit contour lines and surface samples, not neural tracts.
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Matrix4, Vector3, Quaternion, Box3 } from '../docs/dashboard-concepts/vendor/three/three.module.min.js';

const input = readFileSync(new URL('./sources/z-anatomy-brain.glb', import.meta.url));
const jsonLength = input.readUInt32LE(12);
const source = JSON.parse(input.subarray(20, 20 + jsonLength));
const binary = input.subarray(28 + jsonLength);
const types = { 5126: Float32Array, 5125: Uint32Array, 5123: Uint16Array };
function accessor(index) {
  const a = source.accessors[index], v = source.bufferViews[a.bufferView];
  const Type = types[a.componentType], size = a.type === 'VEC3' ? 3 : 1;
  if (v.byteStride) throw new Error('Unexpected interleaved source accessor');
  const data = binary.subarray((v.byteOffset || 0) + (a.byteOffset || 0));
  return new Type(data.buffer, data.byteOffset, a.count * size);
}

const parts = [], box = new Box3();
function visit(index, parent) {
  const n = source.nodes[index], transform = new Matrix4();
  if (n.matrix) transform.fromArray(n.matrix);
  else transform.compose(new Vector3(...(n.translation || [0, 0, 0])),
    new Quaternion(...(n.rotation || [0, 0, 0, 1])), new Vector3(...(n.scale || [1, 1, 1])));
  transform.premultiply(parent);
  if (n.mesh !== undefined && /gyrus|gyri|lobul|pole|pons|medulla oblongata|cuneus|cerebellum/i.test(n.name)) {
    const primitive = source.meshes[n.mesh].primitives[0];
    const p = accessor(primitive.attributes.POSITION), vertices = [];
    for (let i = 0; i < p.length; i += 3) {
      const point = new Vector3(p[i], p[i + 1], p[i + 2]).applyMatrix4(transform);
      box.expandByPoint(point); vertices.push(point);
    }
    parts.push({ name: n.name, vertices, indices: accessor(primitive.indices) });
  }
  for (const child of n.children || []) visit(child, transform);
}
for (const root of source.scenes[source.scene || 0].nodes) visit(root, new Matrix4());
const center = box.getCenter(new Vector3()), scale = 3.5 / box.getSize(new Vector3()).z;
for (const part of parts) for (const v of part.vertices) {
  v.sub(center).multiplyScalar(scale); const z = v.z; v.z = v.x; v.x = -z;
}

let seed = 601;
const random = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
const cortex = [], hindbrain = [], samples = [], sampleRegions = [];
for (const part of parts) {
  const isHind = /lobul|cerebell|pons|medulla/i.test(part.name) && !/parietal/i.test(part.name);
  const lines = isHind ? hindbrain : cortex;
  const direction = new Vector3(.2 + random() * .4, .35 + random() * .8, .3).normalize();
  const levels = part.vertices.map(v => v.dot(direction));
  const min = Math.min(...levels), max = Math.max(...levels), cuts = isHind ? 9 : 13;
  for (let cut = 1; cut < cuts; cut++) {
    const height = min + (max - min) * cut / cuts;
    for (let i = 0; i < part.indices.length; i += 3) {
      const ids = Array.from(part.indices.subarray(i, i + 3)), hits = [];
      for (let e = 0; e < 3; e++) {
        const a = ids[e], b = ids[(e + 1) % 3], da = levels[a] - height, db = levels[b] - height;
        if ((da < 0) !== (db < 0)) hits.push(part.vertices[a].clone().lerp(part.vertices[b], da / (da - db)));
      }
      if (hits.length === 2) for (const p of hits) lines.push(...p.toArray());
    }
  }
  if (!isHind) {
    const start = samples.length / 3;
    for (let i = 0; i < part.indices.length; i += 27) {
      const a = part.vertices[part.indices[i]], b = part.vertices[part.indices[i + 1]], c = part.vertices[part.indices[i + 2]];
      const u = Math.sqrt(random()), v = random();
      samples.push(...a.clone().multiplyScalar(1 - u).addScaledVector(b, u * (1 - v)).addScaledVector(c, u * v).toArray());
    }
    sampleRegions.push({ structure: part.name, start, count: samples.length / 3 - start });
  }
}

// Atlas anchors come from named source meshes, never a hand-positioned ellipsoid.
const anchorDefinitions = [
  ['frontal', 'Frontal lobe', 'Middle frontal gyrus'],
  ['parietal', 'Parietal lobe', 'Superior parietal lobule'],
  ['temporal', 'Temporal lobe', 'Middle temporal gyrus'],
  ['occipital', 'Occipital lobe', 'Lateral occipital gyrus (Middle occipital gyrus*)'],
  ['cerebellum', 'Cerebellum', 'Inferior semilunar lobule'],
  ['brainstem', 'Brainstem', 'Pons']
];
const landmarks = anchorDefinitions.flatMap(([id, label, name]) => ['l', 'r'].map(hemisphere => {
  const part = parts.find(p => p.name === name + '.' + hemisphere);
  if (!part) throw new Error('Missing atlas structure: ' + name + '.' + hemisphere);
  const center = part.vertices.reduce((sum, p) => sum.add(p), new Vector3()).divideScalar(part.vertices.length);
  const point = part.vertices.reduce((nearest, p) => p.distanceToSquared(center) < nearest.distanceToSquared(center) ? p : nearest);
  return { id, label, hemisphere, structure: part.name, position: point.toArray() };
}));

// Standard glTF line/point primitives keep the bake usable outside this renderer.
const out = { asset: { version: '2.0', generator: 'brAIn contour bake',
  copyright: 'BodyParts3D © DBCLS; Z-Anatomy contributors; derivative CC BY-SA 4.0' },
  scene: 0, scenes: [{ nodes: [0, 1, 2] }], nodes: [], meshes: [], accessors: [], bufferViews: [], buffers: [] };
const buffers = []; let offset = 0;
for (const [i, [name, values, mode]] of [['cortex', cortex, 1], ['hindbrain', hindbrain, 1], ['samples', samples, 0]].entries()) {
  const typed = new Float32Array(values.map(v => Math.round(v * 100000) / 100000));
  const bounds = new Box3().setFromBufferAttribute({ count: typed.length / 3,
    getX: i => typed[i * 3], getY: i => typed[i * 3 + 1], getZ: i => typed[i * 3 + 2] });
  const bytes = Buffer.from(typed.buffer); buffers.push(bytes);
  out.bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: bytes.length }); offset += bytes.length;
  out.accessors.push({ bufferView: i, componentType: 5126, count: typed.length / 3, type: 'VEC3',
    min: bounds.min.toArray(), max: bounds.max.toArray() });
  out.meshes.push({ name, primitives: [{ attributes: { POSITION: i }, mode }] });
  out.nodes.push({ name, mesh: i, ...(name === 'samples' ? { extras: { regions: sampleRegions, landmarks } } : {}) });
}
out.buffers.push({ byteLength: offset });
let json = Buffer.from(JSON.stringify(out));
json = Buffer.concat([json, Buffer.alloc((4 - json.length % 4) % 4, 32)]);
const bin = Buffer.concat(buffers), header = Buffer.alloc(20), binHeader = Buffer.alloc(8);
header.writeUInt32LE(0x46546c67, 0); header.writeUInt32LE(2, 4);
header.writeUInt32LE(28 + json.length + bin.length, 8);
header.writeUInt32LE(json.length, 12); header.writeUInt32LE(0x4e4f534a, 16);
binHeader.writeUInt32LE(bin.length, 0); binHeader.writeUInt32LE(0x004e4942, 4);
const result = Buffer.concat([header, json, binHeader, bin]);
writeFileSync(new URL('../docs/dashboard-concepts/assets/brain-contours.glb', import.meta.url), result);
console.log(JSON.stringify({ sourceSHA256: createHash('sha256').update(input).digest('hex'),
  structures: parts.length, contourSegments: (cortex.length + hindbrain.length) / 6,
  surfaceSamples: samples.length / 3, bytes: result.length, normalizedSize: box.getSize(new Vector3()).multiplyScalar(scale).toArray() }, null, 2));
