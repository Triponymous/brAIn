import { Matrix4, Vector3 } from 'three';

export function projectLandmarks(landmarks, camera, width, height) {
  const hemisphere = camera.position.z >= 0 ? 'l' : 'r';
  const points = landmarks.filter(p => p.hemisphere === hemisphere).map(p => {
    const projected = new Vector3(...p.position).project(camera);
    return { ...p, x: (projected.x * .5 + .5) * width,
      y: (.5 - projected.y * .5) * height, depth: projected.z };
  }).filter(p => p.depth < 1 && p.depth > -1 && p.x > 0 && p.x < width && p.y > 0 && p.y < height);
  const ordered = [...points].sort((a, b) => a.x - b.x);
  for (const side of [0, 1]) {
    const half = ordered.slice(side ? Math.ceil(ordered.length / 2) : 0, side ? undefined : Math.ceil(ordered.length / 2));
    half.sort((a, b) => a.y - b.y);
    let last = side ? 160 : 120;
    half.forEach((p, i) => {
      p.labelX = side ? width - 12 : 12;
      p.labelY = Math.max(last, Math.min(height - 64 - (half.length - i - 1) * 24, p.y));
      p.side = side; last = p.labelY + 24;
    });
  }
  return points;
}

export function createAtlasOverlay(root, landmarks) {
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.classList.add('brain-atlas');
  svg.setAttribute('role', 'img');
  svg.setAttribute('aria-label', 'Anatomical atlas landmarks. Labels follow source geometry, not SNN functions.');
  const entries = new Map();
  for (const landmark of landmarks.filter(p => p.hemisphere === 'l')) {
    const group = document.createElementNS(svgNS, 'g');
    group.dataset.atlasRegion = landmark.id;
    const path = document.createElementNS(svgNS, 'path'), dot = document.createElementNS(svgNS, 'circle');
    const text = document.createElementNS(svgNS, 'text'), title = document.createElementNS(svgNS, 'title');
    dot.setAttribute('r', '2.5');
    group.append(title, path, dot, text); svg.append(group);
    entries.set(landmark.id, { group, path, dot, text, title });
  }
  const button = document.createElement('button');
  button.className = 'brain-atlas-toggle'; button.textContent = 'Atlas labels';
  button.setAttribute('aria-pressed', 'true');
  button.setAttribute('aria-label', 'Show anatomical atlas labels');
  root.append(svg, button);
  const previousCamera = new Matrix4();
  let enabled = true, dirty = true, previousWidth = 0, previousHeight = 0;
  return {
    button,
    toggle() { enabled = !enabled; dirty = true; button.setAttribute('aria-pressed', String(enabled)); svg.style.display = enabled ? '' : 'none'; },
    render(camera, width, height) {
      if (!enabled) return;
      // Model ticks do not move atlas landmarks; avoid rewriting SVG labels on every sample.
      if (!dirty && width === previousWidth && height === previousHeight && previousCamera.equals(camera.matrixWorld)) return;
      dirty = false; previousWidth = width; previousHeight = height; previousCamera.copy(camera.matrixWorld);
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      for (const { group } of entries.values()) group.style.display = 'none';
      for (const p of projectLandmarks(landmarks, camera, width, height)) {
        const { group, path, dot, text, title } = entries.get(p.id);
        group.style.display = '';
        group.dataset.structure = p.structure;
        group.dataset.hemisphere = p.hemisphere;
        const endX = p.side ? p.labelX - 88 : p.labelX + 88;
        path.setAttribute('d', `M${p.x},${p.y}L${endX},${p.labelY - 3}H${p.labelX}`);
        dot.setAttribute('cx', p.x); dot.setAttribute('cy', p.y);
        text.setAttribute('x', p.labelX); text.setAttribute('y', p.labelY - 8);
        text.setAttribute('text-anchor', p.side ? 'end' : 'start');
        text.textContent = p.label.replace(' lobe', '') + ' · ' + p.hemisphere.toUpperCase();
        title.textContent = p.label + ' — atlas anchor: ' + p.structure;
      }
    }
  };
}
