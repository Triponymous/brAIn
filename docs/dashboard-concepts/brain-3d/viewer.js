import * as T from 'three';
import { OrbitControls } from '../vendor/three/OrbitControls.js';
import { GLTFLoader } from '../vendor/three/GLTFLoader.js';
import { neuralLayout, fibers, fiberMaterial, neuronCloud, colorNeurons } from './network.js?v=atlas-4';
import { createAtlasOverlay } from './anatomy.js?v=atlas-4';

export async function createBrainViewer(nodes, edges, onSelect, onHover) {
  const started = performance.now(), root = document.createElement('div');
  root.className = 'brain-viewport';
  root.innerHTML = `<div class="brain-hud"><span>SPATIAL NETWORK <b>3D</b></span><span data-brain-status>Preparing anatomy</span></div>
    <div class="brain-camera" role="group" aria-label="Brain camera views">
      <button data-camera="perspective" aria-pressed="true">Perspective</button>
      <button data-camera="side" aria-pressed="false">Side</button>
      <button data-camera="front" aria-pressed="false">Front</button>
      <button data-camera="top" aria-pressed="false">Top</button>
    </div><button class="brain-selection" aria-label="Inspect selected neuron"></button>
    <div class="brain-navigation"><span>Drag to orbit · scroll to zoom · click a neuron</span>
      <span data-brain-count>1,000 units</span></div>`;
  const renderer = new T.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setClearColor('#091417', 0);
  // Contours and illustrative fibers keep their look; categorical point colors bypass tone mapping.
  renderer.toneMapping = T.NeutralToneMapping;
  renderer.outputColorSpace = T.SRGBColorSpace;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0; canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', 'Interactive 3D brain. Drag to rotate, pinch or scroll to zoom. Arrow keys rotate; plus and minus zoom; Home resets. Use the Single neuron inspector to select by ID.');
  root.prepend(canvas);
  const scene = new T.Scene(), camera = new T.PerspectiveCamera(35, 1, .1, 100);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = false; controls.enablePan = false;
  controls.minDistance = 3.3; controls.maxDistance = 15; controls.rotateSpeed = .65; controls.zoomSpeed = .65;
  const asset = await new GLTFLoader().loadAsync(new URL('../assets/brain-contours.glb?v=atlas-4', import.meta.url).href)
    .catch(error => { controls.dispose(); renderer.dispose(); throw error; });
  const contourMaterial = new T.LineBasicMaterial({ color: '#4a879e', transparent: true,
    opacity: .17, depthWrite: false, blending: T.AdditiveBlending });
  const contour = new T.Group();
  let sample, atlas;
  asset.scene.traverse(object => {
    if (!object.geometry) return;
    object.material.dispose();
    if (object.name === 'samples') { sample = object.geometry.attributes.position; atlas = object.userData; }
    else contour.add(new T.LineSegments(object.geometry, contourMaterial));
  });
  const styles = getComputedStyle(document.documentElement);
  const palette = Object.fromEntries(['s', 'e', 'c', 'w'].map(key => [key, styles.getPropertyValue('--' + key).trim()]));
  const positions = neuralLayout(nodes, sample), nodeCloud = neuronCloud(nodes, positions, palette);
  const anatomy = createAtlasOverlay(root, atlas.landmarks);
  root.dataset.palette = JSON.stringify(palette);
  root.dataset.positionSource = 'atlas-surface';
  const shownEdges = edges.filter((_, i) => i % 5 === 0);
  const connections = new T.LineSegments(fibers(shownEdges, positions), fiberMaterial());
  const selected = new T.LineSegments(new T.BufferGeometry(), fiberMaterial(true));
  scene.add(contour, connections, nodeCloud, selected);
  const label = root.querySelector('.brain-selection'), status = root.querySelector('[data-brain-status]');
  const raycaster = new T.Raycaster(), pointer = new T.Vector2();
  raycaster.params.Points.threshold = .05;
  const abort = new AbortController(), listener = { signal: abort.signal };
  let snapshot, active = false, inView = true, lost = false, selectedId, width = 1, height = 1, dpr = 1.5;
  let rect, pointerDown, resizeTimer, renderCount = 0, view = 'perspective', baseDistance = 8.5;
  let timings = [], lastQualityChange = 0;
  root.querySelector('[data-brain-count]').textContent = `${nodes.length.toLocaleString('en-US')} units · ${shownEdges.length} sample paths`;

  function render() {
    if (!active || !inView || lost || !snapshot || document.hidden) return;
    const start = performance.now(); renderer.render(scene, camera);
    anatomy.render(camera, width, height);
    const p = positions.get(snapshot.selected).clone().project(camera);
    const x = (p.x * .5 + .5) * width, y = (-p.y * .5 + .5) * height;
    label.hidden = x < 0 || x > width || y < 0 || y > height || p.z > 1;
    label.style.transform = `translate(${Math.min(width - 96, Math.max(8, x + 13))}px,${Math.min(height - 72, Math.max(76, y - 15))}px)`;
    root.dataset.camera = camera.position.toArray().map(v => v.toFixed(3)).join(',');
    root.dataset.renders = String(++renderCount);
    root.dataset.geometries = String(renderer.info.memory.geometries);
    root.dataset.drawCalls = String(renderer.info.render.calls);
    timings.push(performance.now() - start); if (timings.length > 300) timings.shift();
    if (timings.length === 300 && renderCount % 60 === 0) {
      const sorted = [...timings].sort((a, b) => a - b);
      root.dataset.renderP95 = sorted[285].toFixed(2);
      if (start - lastQualityChange > 5000 && sorted[285] > 22 && dpr > 1) {
        dpr = 1; lastQualityChange = start; size(true);
      }
    }
  }

  function size(force = false) {
    if (!active) return;
    const w = root.clientWidth, h = root.clientHeight;
    if (!w || !h || (!force && w === width && h === height)) return;
    const zoomRatio = width === 1 ? 1 : camera.position.distanceTo(controls.target) / baseDistance;
    width = w; height = h; rect = canvas.getBoundingClientRect();
    renderer.setPixelRatio(Math.min(devicePixelRatio || 1, dpr, w < 500 ? 1.25 : 1.5));
    renderer.setSize(w, h, false);
    nodeCloud.material.uniforms.uDpr.value = renderer.getPixelRatio() * Math.min(1, w / 620);
    camera.aspect = w / h; camera.updateProjectionMatrix();
    // Fit horizontal anatomy as well as height; narrow phones retain the complete silhouette.
    baseDistance = Math.max(w < 500 ? 8.0 : 7.3, 5.2 / camera.aspect);
    camera.position.sub(controls.target).setLength(baseDistance * zoomRatio).add(controls.target);
    controls.update();
    render();
  }

  function preset(name) {
    view = name;
    const directions = { perspective: [2.3, 1.25, 7.5], side: [0, .2, 8], front: [-8, .2, 0], top: [0, 8, .001] };
    camera.position.fromArray(directions[name]).setLength(baseDistance);
    controls.target.set(0, .03, 0); controls.update();
    root.querySelectorAll('[data-camera]').forEach(b => b.setAttribute('aria-pressed', b.dataset.camera === name));
    render();
  }

  function update(next) {
    snapshot = next;
    if (!lost) status.textContent = next.provenance || 'WEBGL · SYNTHETIC SNN';
    root.querySelector('[data-brain-count]').textContent = next.connectionCaption || `${nodes.length.toLocaleString('en-US')} units · ${shownEdges.length} sample paths`;
    colorNeurons(nodeCloud, nodes, next);
    contour.visible = next.surface;
    contourMaterial.opacity = next.region === 'all' ? .15 : .045;
    connections.visible = next.showEdges;
    connections.material.uniforms.uOpacity.value = next.region === 'all' ? .1 : .018;
    selected.visible = next.showEdges;
    if (selectedId !== next.selected) {
      const adjacent = edges.filter(e => e.from === next.selected || e.to === next.selected);
      selected.geometry.dispose(); selected.geometry = fibers(adjacent, positions);
      selectedId = next.selected;
      label.textContent = nodes.find(n => n.id === next.selected).label + ' ↗';
      root.dataset.selected = next.selected; root.dataset.adjacent = adjacent.length;
    }
    root.dataset.region = next.region; root.dataset.frame = next.frame;
    root.dataset.surface = next.surface; root.dataset.connections = next.showEdges;
    tick(next.frame);
  }

  function tick(frame) {
    connections.material.uniforms.uTime.value = frame / 6;
    selected.material.uniforms.uTime.value = frame / 6;
    root.dataset.signalTime = frame.toFixed(2); render();
  }

  function pick(event) {
    rect = canvas.getBoundingClientRect();
    pointer.set((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(nodeCloud).filter(h => snapshot.region === 'all' || nodes[h.index].key === snapshot.region);
    hits.sort((a, b) => a.distanceToRay - b.distanceToRay);
    return hits.length ? nodes[hits[0].index] : null;
  }

  controls.addEventListener('change', render);
  canvas.addEventListener('pointerdown', e => { pointerDown = [e.clientX, e.clientY]; onHover(null); }, listener);
  canvas.addEventListener('pointermove', e => {
    if (e.buttons) { onHover(null); return; }
    const node = pick(e); canvas.style.cursor = node ? 'pointer' : 'grab'; onHover(node, e);
  }, listener);
  canvas.addEventListener('pointerup', e => {
    if (pointerDown && Math.hypot(e.clientX - pointerDown[0], e.clientY - pointerDown[1]) < 5) {
      const n = pick(e); if (n) onSelect(n.id);
    }
    pointerDown = null;
  }, listener);
  canvas.addEventListener('pointerleave', () => onHover(null), listener);
  canvas.addEventListener('pointercancel', () => { pointerDown = null; }, listener);
  canvas.addEventListener('keydown', e => {
    if (['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(e.key)) {
      e.preventDefault(); const s = new T.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
      s.theta += e.key === 'ArrowLeft' ? -.13 : e.key === 'ArrowRight' ? .13 : 0;
      s.phi += e.key === 'ArrowUp' ? -.13 : e.key === 'ArrowDown' ? .13 : 0;
      s.makeSafe(); camera.position.setFromSpherical(s).add(controls.target); controls.update();
    } else if (e.key === 'Home') { e.preventDefault(); preset('perspective'); }
    else if (['+', '=', '-'].includes(e.key)) { e.preventDefault(); zoom(e.key === '-' ? -.15 : .15); }
  }, listener);
  root.addEventListener('click', e => { if (e.target.dataset.camera) preset(e.target.dataset.camera); }, listener);
  label.addEventListener('click', () => onSelect(snapshot.selected), listener);
  anatomy.button.addEventListener('click', () => { anatomy.toggle(); render(); }, listener);
  canvas.addEventListener('webglcontextlost', e => {
    e.preventDefault(); lost = true; root.classList.add('context-lost');
    status.textContent = '3D paused — graphics context lost. Reload or use Circuit.';
  }, listener);
  canvas.addEventListener('webglcontextrestored', () => { lost = false; root.classList.remove('context-lost'); status.textContent = 'WEBGL · RESTORED'; render(); }, listener);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) render(); }, listener);
  const resize = new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => size(), 250); });
  resize.observe(root);
  const visibility = new IntersectionObserver(([entry]) => {
    inView = entry.isIntersecting; root.dataset.inView = inView; if (inView) render();
  });
  visibility.observe(root);
  preset('perspective');
  status.textContent = 'WEBGL · SYNTHETIC SNN'; root.dataset.ready = 'true';
  root.dataset.loadMs = (performance.now() - started).toFixed(0);
  function zoom(delta, reset = false) {
    if (reset) { preset('perspective'); return; }
    const length = T.MathUtils.clamp(camera.position.distanceTo(controls.target) * (1 - delta), 3.3, 15);
    camera.position.sub(controls.target).setLength(length).add(controls.target); controls.update(); render();
  }
  return {
    mount(host) { active = true; if (root.parentNode !== host) host.replaceChildren(root); size(true); render(); },
    suspend() { active = false; onHover(null); }, update, tick, zoom,
    dispose() {
      active = false; abort.abort(); resize.disconnect(); visibility.disconnect(); clearTimeout(resizeTimer); controls.dispose();
      const geometries = new Set(), materials = new Set();
      scene.traverse(o => { if (o.geometry) geometries.add(o.geometry); if (o.material) materials.add(o.material); });
      geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); renderer.dispose(); root.remove();
    }
  };
}
