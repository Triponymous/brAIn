import { useRef, useEffect } from "react";
import type { BrainState } from "../lib/ws";

type ExtendedState = BrainState & {
  spike_counts?: { sensory?: number; feature?: number; concept?: number };
};

/**
 * Brain visualization — side-view anatomical layout with 7 regions.
 *
 * Each region is positioned roughly anatomically:
 *   - Sensory cortex: back (right side) — receives raw input
 *   - Feature cortex: upper-back — extracts patterns
 *   - Association cortex: center-top — cross-modal binding
 *   - Concept (WTA): center — sparse representations
 *   - Working Memory: frontal — holds context
 *   - Motor: front-top (left) — output/action
 *   - Meta: brainstem area (bottom-center) — modulators
 *
 * Activity shown as glowing blobs with particle neurons inside.
 * Connections drawn as curved pathways with flow indicators.
 */

type Region = {
  name: string;
  label: string;
  x: number; // fraction of W
  y: number; // fraction of H
  rx: number; // ellipse radius x (fraction of W)
  ry: number; // ellipse radius y (fraction of H)
  color: string;
};

const REGIONS: Region[] = [
  { name: "sensory",     label: "Sensory",     x: 0.78, y: 0.38, rx: 0.10, ry: 0.18, color: "80, 200, 120" },
  { name: "feature",     label: "Feature",     x: 0.62, y: 0.22, rx: 0.09, ry: 0.12, color: "100, 180, 220" },
  { name: "association", label: "Association",  x: 0.48, y: 0.30, rx: 0.10, ry: 0.14, color: "180, 140, 220" },
  { name: "concept",     label: "Concept",      x: 0.35, y: 0.40, rx: 0.12, ry: 0.16, color: "52, 211, 153" },
  { name: "wm",          label: "Working Mem",  x: 0.22, y: 0.32, rx: 0.08, ry: 0.11, color: "200, 130, 230" },
  { name: "motor",       label: "Motor",        x: 0.15, y: 0.20, rx: 0.07, ry: 0.10, color: "230, 160, 80" },
  { name: "meta",        label: "Meta",         x: 0.50, y: 0.72, rx: 0.08, ry: 0.06, color: "200, 200, 100" },
];

const PATHWAYS = [
  ["sensory", "feature"],
  ["feature", "association"],
  ["association", "concept"],
  ["concept", "wm"],
  ["concept", "motor"],
];

export function BrainPanel({ state }: { state: BrainState | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!state || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    const s = state as ExtendedState;
    const spikes = s.spike_counts ?? {};
    const conceptMem = state.concept_membrane ?? [];
    const maxConcept = Math.max(1, ...conceptMem.map(Math.abs));
    const mods = state.modulators;

    // Clear
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, W, H);

    // Draw brain silhouette (subtle outline)
    drawBrainOutline(ctx, W, H);

    // Draw pathways between regions
    for (const [from, to] of PATHWAYS) {
      const r1 = REGIONS.find(r => r.name === from)!;
      const r2 = REGIONS.find(r => r.name === to)!;
      drawPathway(ctx, W, H, r1, r2, spikes);
    }

    // Draw each region
    for (const region of REGIONS) {
      const activity = getRegionActivity(region.name, spikes, conceptMem, maxConcept, state);
      drawRegion(ctx, W, H, region, activity, mods);
    }

    // Draw concept neurons as small dots inside the concept region
    const conceptRegion = REGIONS.find(r => r.name === "concept")!;
    drawConceptNeurons(ctx, W, H, conceptRegion, conceptMem, maxConcept);

    // Legend
    ctx.fillStyle = "#4b5563";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.fillText("Brighter = more active  •  Particles = individual neuron spikes", 10, H - 8);

  }, [state]);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Waiting for brain state...
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      <canvas
        ref={canvasRef}
        className="flex-1 w-full rounded-lg"
        style={{ background: "#0a0a0f" }}
      />
    </div>
  );
}

function drawBrainOutline(ctx: CanvasRenderingContext2D, W: number, H: number) {
  // Draw a subtle brain silhouette (side view)
  ctx.save();
  ctx.strokeStyle = "rgba(100, 120, 140, 0.15)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();

  const cx = W * 0.45;
  const cy = H * 0.40;
  const rw = W * 0.38;
  const rh = H * 0.32;

  // Main brain shape — slightly irregular ellipse
  ctx.ellipse(cx, cy, rw, rh, 0, 0, Math.PI * 2);
  ctx.stroke();

  // Frontal lobe bump
  ctx.beginPath();
  ctx.ellipse(W * 0.15, H * 0.30, W * 0.10, H * 0.15, -0.3, 0, Math.PI * 2);
  ctx.stroke();

  // Cerebellum (back-bottom)
  ctx.beginPath();
  ctx.ellipse(W * 0.72, H * 0.62, W * 0.10, H * 0.08, 0.2, 0, Math.PI * 2);
  ctx.stroke();

  // Central sulcus hint
  ctx.beginPath();
  ctx.moveTo(W * 0.38, H * 0.10);
  ctx.quadraticCurveTo(W * 0.40, H * 0.40, W * 0.42, H * 0.65);
  ctx.strokeStyle = "rgba(100, 120, 140, 0.08)";
  ctx.stroke();

  // Lateral sulcus hint
  ctx.beginPath();
  ctx.moveTo(W * 0.30, H * 0.42);
  ctx.quadraticCurveTo(W * 0.50, H * 0.45, W * 0.70, H * 0.50);
  ctx.stroke();

  ctx.restore();
}

function drawPathway(
  ctx: CanvasRenderingContext2D, W: number, H: number,
  from: Region, to: Region,
  spikes: Record<string, number>
) {
  const x1 = from.x * W, y1 = from.y * H;
  const x2 = to.x * W, y2 = to.y * H;

  // Pathway strength based on spike flow
  const fromSpikes = spikes[from.name as keyof typeof spikes] ?? 0;
  const alpha = Math.min(0.4, 0.05 + (fromSpikes / 200) * 0.35);

  ctx.save();
  ctx.strokeStyle = `rgba(120, 200, 180, ${alpha})`;
  ctx.lineWidth = 1 + alpha * 4;

  // Curved pathway
  const midX = (x1 + x2) / 2;
  const midY = (y1 + y2) / 2 - 20;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.quadraticCurveTo(midX, midY, x2, y2);
  ctx.stroke();

  // Flow dots along pathway (animated feel via tick-based offset)
  if (fromSpikes > 0) {
    const numDots = Math.min(5, Math.ceil(fromSpikes / 40));
    for (let i = 0; i < numDots; i++) {
      const t = ((Date.now() / 800 + i / numDots) % 1);
      const px = x1 + (x2 - x1) * t;
      const py = y1 + (y2 - y1) * t + Math.sin(t * Math.PI) * -20;
      ctx.beginPath();
      ctx.arc(px, py, 2, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(52, 211, 153, ${0.3 + alpha})`;
      ctx.fill();
    }
  }

  ctx.restore();
}

function getRegionActivity(
  name: string,
  spikes: Record<string, number>,
  conceptMem: number[],
  maxConcept: number,
  state: BrainState
): number {
  if (name === "concept") {
    const active = conceptMem.filter(v => v > maxConcept * 0.1).length;
    return Math.min(1, active / Math.max(1, conceptMem.length) * 3);
  }
  if (name === "wm") {
    const wm = state.wm_membrane ?? [];
    const active = wm.filter(v => Math.abs(v) > 0.01).length;
    return Math.min(1, active / Math.max(1, wm.length) * 2);
  }
  if (name === "meta") {
    const m = state.modulators;
    return Math.min(1, (Math.abs(m.DA ?? 0) + Math.abs(m.NE ?? 0) + Math.abs(m.ACh ?? 0)) / 1.5);
  }
  const count = spikes[name as keyof typeof spikes] ?? 0;
  return Math.min(1, count / 100);
}

function drawRegion(
  ctx: CanvasRenderingContext2D, W: number, H: number,
  region: Region, activity: number,
  mods: Record<string, number>
) {
  const x = region.x * W;
  const y = region.y * H;
  const rx = region.rx * W;
  const ry = region.ry * H;

  ctx.save();

  // Outer glow
  const glowAlpha = 0.05 + activity * 0.25;
  const grad = ctx.createRadialGradient(x, y, 0, x, y, Math.max(rx, ry) * 1.8);
  grad.addColorStop(0, `rgba(${region.color}, ${glowAlpha})`);
  grad.addColorStop(1, `rgba(${region.color}, 0)`);
  ctx.fillStyle = grad;
  ctx.fillRect(x - rx * 2, y - ry * 2, rx * 4, ry * 4);

  // Region body
  const bodyAlpha = 0.08 + activity * 0.35;
  ctx.beginPath();
  ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
  ctx.fillStyle = `rgba(${region.color}, ${bodyAlpha})`;
  ctx.fill();
  ctx.strokeStyle = `rgba(${region.color}, ${0.2 + activity * 0.4})`;
  ctx.lineWidth = 1;
  ctx.stroke();

  // Sparkle particles inside (simulate neuron firing)
  if (activity > 0.05) {
    const numParticles = Math.floor(activity * 30);
    for (let i = 0; i < numParticles; i++) {
      const angle = (i / numParticles) * Math.PI * 2 + Date.now() / 3000;
      const dist = 0.3 + Math.sin(i * 7 + Date.now() / 500) * 0.5;
      const px = x + Math.cos(angle) * rx * dist;
      const py = y + Math.sin(angle) * ry * dist;
      const pr = 1 + Math.random() * 1.5;
      ctx.beginPath();
      ctx.arc(px, py, pr, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${region.color}, ${0.4 + Math.random() * 0.4})`;
      ctx.fill();
    }
  }

  // Label
  ctx.fillStyle = `rgba(220, 220, 230, ${0.4 + activity * 0.4})`;
  ctx.font = "10px sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(region.label, x, y + ry + 14);

  ctx.restore();
}

function drawConceptNeurons(
  ctx: CanvasRenderingContext2D, W: number, H: number,
  region: Region, membrane: number[], maxVal: number
) {
  if (membrane.length === 0) return;
  const x = region.x * W;
  const y = region.y * H;
  const rx = region.rx * W * 0.85;
  const ry = region.ry * H * 0.85;

  // Draw the top-N most active concepts as bright dots with IDs
  const indexed = membrane.map((v, i) => ({ i, v: v / maxVal }));
  indexed.sort((a, b) => b.v - a.v);
  const topN = indexed.slice(0, 8).filter(c => c.v > 0.05);

  for (let k = 0; k < topN.length; k++) {
    const { i, v } = topN[k];
    // Spiral layout inside the region
    const angle = (k / 8) * Math.PI * 2 - Math.PI / 2;
    const dist = 0.3 + k * 0.08;
    const px = x + Math.cos(angle) * rx * dist;
    const py = y + Math.sin(angle) * ry * dist;
    const r = 3 + v * 6;

    // Bright dot
    ctx.beginPath();
    ctx.arc(px, py, r, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(52, 211, 153, ${0.5 + v * 0.5})`;
    ctx.fill();

    // ID label
    ctx.fillStyle = `rgba(255, 255, 255, ${0.4 + v * 0.5})`;
    ctx.font = "8px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`#${i}`, px, py - r - 3);
  }
}
