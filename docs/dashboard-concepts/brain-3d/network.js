import * as T from 'three';

export function neuralLayout(nodes, samples) {
  let seed = 441;
  const rand = () => ((seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296);
  const indices = Array.from({ length: samples.count }, (_, i) => i);
  for (let i = indices.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [indices[i], indices[j]] = [indices[j], indices[i]];
  }
  const positions = new Map();
  // Model roles have no known anatomical homology: use real atlas surfaces, not invented centers.
  nodes.forEach((n, i) => positions.set(n.id, new T.Vector3().fromBufferAttribute(samples, indices[i])));
  return positions;
}

export function edgeCurve(edge, positions) {
  const a = positions.get(edge.from), b = positions.get(edge.to);
  // Shared waypoints make readable bundles; these are not measured anatomical tracts.
  const hub = p => new T.Vector3(p.x * .22, .2 + p.y * .28, Math.sign(p.z) * .38);
  const c1 = a.clone().lerp(hub(a), .83), c2 = b.clone().lerp(hub(b), .83);
  c1.y += .26; c2.y += .26;
  return new T.CubicBezierCurve3(a, c1, c2, b);
}

export function fibers(edges, positions) {
  const vertices = [], progress = [], phases = [];
  edges.forEach((edge, i) => {
    const curve = edgeCurve(edge, positions), steps = 28;
    for (let j = 0; j < steps; j++) {
      vertices.push(...curve.getPoint(j / steps).toArray(), ...curve.getPoint((j + 1) / steps).toArray());
      progress.push(j / steps, (j + 1) / steps); phases.push((i * .618034) % 1, (i * .618034) % 1);
    }
  });
  const g = new T.BufferGeometry();
  g.setAttribute('position', new T.Float32BufferAttribute(vertices, 3));
  g.setAttribute('aProgress', new T.Float32BufferAttribute(progress, 1));
  g.setAttribute('aPhase', new T.Float32BufferAttribute(phases, 1));
  return g;
}

export function fiberMaterial(selected = false) {
  return new T.ShaderMaterial({
    transparent: true, depthWrite: false, blending: T.AdditiveBlending,
    uniforms: { uTime: { value: 0 }, uOpacity: { value: selected ? .7 : .18 },
      uSignals: { value: 1 }, uSelected: { value: selected ? 1 : 0 } },
    vertexShader: `attribute float aProgress; attribute float aPhase;
      varying float vProgress; varying float vPhase; varying float vDepth;
      void main(){vProgress=aProgress;vPhase=aPhase;
        vec4 mv=modelViewMatrix*vec4(position,1.0);vDepth=-mv.z;
        gl_Position=projectionMatrix*mv;}`,
    fragmentShader: `uniform float uTime; uniform float uOpacity; uniform float uSignals;
      uniform float uSelected; varying float vProgress; varying float vPhase; varying float vDepth;
      void main(){float head=fract(uTime*.19+vPhase);
        float gap=abs(vProgress-head);float spark=exp(-gap*gap*17000.0)*uSignals;
        float trail=exp(-gap*gap*450.0)*.32*uSignals;
        vec3 blue=mix(vec3(.015,.065,.11),vec3(.08,.21,.29),clamp((10.0-vDepth)/5.0,0.0,1.0));
        vec3 ink=mix(blue,vec3(1.0,.47,.045),max(spark,trail));
        ink=mix(ink,vec3(.67,.77,.57),uSelected*.46);
        gl_FragColor=vec4(ink+vec3(.8,.35,.025)*spark,uOpacity+spark*.72);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
      }`
  });
}

export function neuronCloud(nodes, positions, palette) {
  const geometry = new T.BufferGeometry();
  geometry.setAttribute('position', new T.Float32BufferAttribute(nodes.flatMap(n => positions.get(n.id).toArray()), 3));
  geometry.setAttribute('color', new T.Float32BufferAttribute(nodes.flatMap(n => new T.Color(palette[n.key]).toArray()), 3));
  geometry.setAttribute('aSize', new T.Float32BufferAttribute(new Float32Array(nodes.length), 1));
  geometry.setAttribute('aOpacity', new T.Float32BufferAttribute(new Float32Array(nodes.length), 1));
  const material = new T.ShaderMaterial({
    transparent: true, depthWrite: false, vertexColors: true, blending: T.NormalBlending, toneMapped: false,
    uniforms: { uDpr: { value: 1 } },
    vertexShader: `attribute float aSize; attribute float aOpacity;
      varying vec3 vColor; varying float vOpacity; uniform float uDpr;
      void main(){vColor=color;vOpacity=aOpacity;vec4 mv=modelViewMatrix*vec4(position,1.0);
        gl_PointSize=clamp(aSize*uDpr*7.0/-mv.z,1.0,32.0);
        gl_Position=projectionMatrix*mv;}`,
    fragmentShader: `varying vec3 vColor; varying float vOpacity;
      void main(){float d=length(gl_PointCoord-.5)*2.0;if(d>1.0)discard;
        float core=1.0-smoothstep(.28,.52,d);float glow=exp(-d*d*5.0)*.28;
        gl_FragColor=vec4(vColor,min(1.0,core+glow)*vOpacity);
        #include <colorspace_fragment>
      }`
  });
  const points = new T.Points(geometry, material);
  // Draw after additive fibers so a selected path cannot repaint a categorical point.
  points.renderOrder = 2;
  return points;
}

export function colorNeurons(cloud, nodes, state) {
  const sizes = cloud.geometry.attributes.aSize, opacity = cloud.geometry.attributes.aOpacity;
  nodes.forEach((n, i) => {
    const dim = state.region !== 'all' && n.key !== state.region;
    const active = state.activity[i] > 0;
    opacity.setX(i, dim ? .08 : active ? .96 : .42);
    sizes.setX(i, dim ? 1.6 : n.id === state.selected ? 18 : active ? 10 : 3.2);
  });
  opacity.needsUpdate = true; sizes.needsUpdate = true;
}
