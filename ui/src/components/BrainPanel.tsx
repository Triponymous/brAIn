import { useRef, useEffect, useState } from "react";
import type { BrainState } from "../lib/ws";

type ExtendedState = BrainState & {
  spike_counts?: { sensory?: number; feature?: number; concept?: number };
};

type ViewMode = "network" | "regions" | "raster";

export function BrainPanel({ state }: { state: BrainState | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [view, setView] = useState<ViewMode>("network");
  const historyRef = useRef<number[][]>([]);

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
    const conceptMem = state.concept_membrane ?? [];
    const maxC = Math.max(0.01, ...conceptMem.map(Math.abs));
    const spikes = s.spike_counts ?? {};

    // Track history for raster view
    const hist = historyRef.current;
    hist.push(conceptMem.map(v => v / maxC));
    if (hist.length > 200) hist.shift();

    ctx.fillStyle = "#08090d";
    ctx.fillRect(0, 0, W, H);

    if (view === "network") drawNetworkView(ctx, W, H, conceptMem, maxC, spikes, state);
    else if (view === "regions") drawRegionView(ctx, W, H, conceptMem, maxC, spikes, state);
    else if (view === "raster") drawRasterView(ctx, W, H, hist);

  }, [state, view]);

  if (!state) {
    return <div className="h-full flex items-center justify-center text-gray-500">Waiting for brain state...</div>;
  }

  return (
    <div className="h-full flex flex-col gap-1">
      <div className="flex gap-1 px-1">
        {(["network", "regions", "raster"] as const).map(v => (
          <button
            key={v}
            onClick={() => setView(v)}
            className={`px-2 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider transition-colors ${
              view === v ? "bg-emerald-900 text-emerald-300" : "bg-gray-900 text-gray-500 hover:text-gray-300"
            }`}
          >
            {v === "network" ? "Neural Net" : v === "regions" ? "Brain Map" : "Spike Raster"}
          </button>
        ))}
      </div>
      <canvas ref={canvasRef} className="flex-1 w-full rounded-lg" style={{ background: "#08090d" }} />
    </div>
  );
}

// ═══════════════════════════════════════════════════
// VIEW 1: NEURAL NETWORK — individual neurons + synapses
// ═══════════════════════════════════════════════════

function drawNetworkView(
  ctx: CanvasRenderingContext2D, W: number, H: number,
  conceptMem: number[], maxC: number,
  spikes: Record<string, number>, state: BrainState
) {
  const n = conceptMem.length;
  if (n === 0) return;

  // Layout: neurons in a force-like layout based on index
  // Group into clusters based on activity similarity
  const normalized = conceptMem.map(v => Math.abs(v) / maxC);

  // Create layers: spread neurons across the canvas
  // Active neurons (>0.1) are bigger and brighter
  const positions: { x: number; y: number; v: number; id: number }[] = [];

  // Spiral galaxy layout — active neurons near center, inactive at edges
  const sorted = normalized.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);

  const cx = W * 0.5;
  const cy = H * 0.48;

  for (let k = 0; k < sorted.length; k++) {
    const { v, i } = sorted[k];
    // Golden angle spiral
    const angle = k * 2.39996; // golden angle in radians
    const r = 20 + Math.sqrt(k) * (Math.min(W, H) * 0.04);
    const x = cx + Math.cos(angle) * r;
    const y = cy + Math.sin(angle) * r;
    positions[i] = { x, y, v, id: i };
  }

  // Draw connections between co-active neurons (top 20 most active)
  const activeIds = sorted.slice(0, 20).filter(s => s.v > 0.05).map(s => s.i);

  // Draw faint grid of all connections first
  ctx.globalAlpha = 0.03;
  ctx.strokeStyle = "#34d399";
  ctx.lineWidth = 0.3;
  for (let i = 0; i < activeIds.length; i++) {
    for (let j = i + 1; j < activeIds.length; j++) {
      const a = positions[activeIds[i]];
      const b = positions[activeIds[j]];
      if (!a || !b) continue;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;

  // Draw synaptic connections between top active neurons (brighter)
  for (let i = 0; i < Math.min(8, activeIds.length); i++) {
    for (let j = i + 1; j < Math.min(8, activeIds.length); j++) {
      const a = positions[activeIds[i]];
      const b = positions[activeIds[j]];
      if (!a || !b) continue;
      const strength = (a.v + b.v) / 2;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      // Curved connection
      const mx = (a.x + b.x) / 2 + (Math.sin(activeIds[i] + activeIds[j]) * 20);
      const my = (a.y + b.y) / 2 + (Math.cos(activeIds[i] - activeIds[j]) * 20);
      ctx.quadraticCurveTo(mx, my, b.x, b.y);
      ctx.strokeStyle = `rgba(52, 211, 153, ${strength * 0.5})`;
      ctx.lineWidth = 0.5 + strength * 2;
      ctx.stroke();

      // Pulse dot traveling along connection
      const t = (Date.now() / 1000 + i * 0.3) % 1;
      const px = a.x + (b.x - a.x) * t;
      const py = a.y + (b.y - a.y) * t;
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(52, 211, 153, ${strength * 0.8})`;
      ctx.fill();
    }
  }

  // Draw all neurons
  for (let i = 0; i < n; i++) {
    const p = positions[i];
    if (!p) continue;
    const v = p.v;
    const isActive = v > 0.05;

    // Glow for active neurons
    if (v > 0.2) {
      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 12 + v * 15);
      grad.addColorStop(0, `rgba(52, 211, 153, ${v * 0.3})`);
      grad.addColorStop(1, "rgba(52, 211, 153, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(p.x - 25, p.y - 25, 50, 50);
    }

    // Neuron dot
    const r = isActive ? 2.5 + v * 5 : 1.2;
    ctx.beginPath();
    ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
    ctx.fillStyle = isActive
      ? `rgba(52, 211, 153, ${0.4 + v * 0.6})`
      : "rgba(52, 70, 80, 0.3)";
    ctx.fill();

    // Label for top-8 active
    if (activeIds.indexOf(i) >= 0 && activeIds.indexOf(i) < 8) {
      ctx.fillStyle = `rgba(255, 255, 255, ${0.5 + v * 0.5})`;
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`#${i}`, p.x, p.y - r - 4);
    }
  }

  // Pipeline indicator (bottom)
  const py = H - 22;
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  const pipeline = [
    { name: "Sensory", count: spikes.sensory ?? 0, color: "#6ee7b7" },
    { name: "Feature", count: spikes.feature ?? 0, color: "#67e8f9" },
    { name: "Concept", count: spikes.concept ?? 0, color: "#34d399" },
  ];
  const totalW = 300;
  const startX = (W - totalW) / 2;
  pipeline.forEach((p, i) => {
    const x = startX + i * 110;
    ctx.fillStyle = p.color;
    ctx.globalAlpha = 0.7;
    ctx.fillText(`${p.name}: ${p.count}`, x + 40, py);
    if (i < pipeline.length - 1) {
      ctx.fillText("→", x + 90, py);
    }
  });
  ctx.globalAlpha = 1;
}

// ═══════════════════════════════════════════════════
// VIEW 2: BRAIN MAP — anatomical regions with heatmaps
// ═══════════════════════════════════════════════════

type Region = {
  name: string; label: string;
  x: number; y: number; rx: number; ry: number;
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
  ["sensory", "feature"], ["feature", "association"],
  ["association", "concept"], ["concept", "wm"], ["concept", "motor"],
];

function drawRegionView(
  ctx: CanvasRenderingContext2D, W: number, H: number,
  conceptMem: number[], maxC: number,
  spikes: Record<string, number>, state: BrainState
) {
  // Brain outline
  ctx.strokeStyle = "rgba(100, 120, 140, 0.12)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(W * 0.45, H * 0.40, W * 0.38, H * 0.32, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(W * 0.15, H * 0.30, W * 0.10, H * 0.15, -0.3, 0, Math.PI * 2);
  ctx.stroke();
  ctx.beginPath();
  ctx.ellipse(W * 0.72, H * 0.62, W * 0.10, H * 0.08, 0.2, 0, Math.PI * 2);
  ctx.stroke();

  // Pathways
  for (const [from, to] of PATHWAYS) {
    const r1 = REGIONS.find(r => r.name === from)!;
    const r2 = REGIONS.find(r => r.name === to)!;
    const flow = (spikes[from as keyof typeof spikes] ?? 0) / 200;
    const alpha = 0.05 + Math.min(0.35, flow * 0.5);
    ctx.strokeStyle = `rgba(120, 200, 180, ${alpha})`;
    ctx.lineWidth = 1 + flow * 3;
    ctx.beginPath();
    ctx.moveTo(r1.x * W, r1.y * H);
    ctx.quadraticCurveTo(
      (r1.x + r2.x) / 2 * W, (r1.y + r2.y) / 2 * H - 15,
      r2.x * W, r2.y * H
    );
    ctx.stroke();
  }

  // Regions
  for (const region of REGIONS) {
    const activity = getActivity(region.name, spikes, conceptMem, maxC, state);
    const x = region.x * W, y = region.y * H;
    const rx = region.rx * W, ry = region.ry * H;

    // Glow
    const grad = ctx.createRadialGradient(x, y, 0, x, y, Math.max(rx, ry) * 1.8);
    grad.addColorStop(0, `rgba(${region.color}, ${0.05 + activity * 0.25})`);
    grad.addColorStop(1, `rgba(${region.color}, 0)`);
    ctx.fillStyle = grad;
    ctx.fillRect(x - rx * 2, y - ry * 2, rx * 4, ry * 4);

    // Body
    ctx.beginPath();
    ctx.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(${region.color}, ${0.08 + activity * 0.35})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(${region.color}, ${0.2 + activity * 0.4})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Particles
    const numP = Math.floor(activity * 25);
    for (let i = 0; i < numP; i++) {
      const a = (i / numP) * Math.PI * 2 + Date.now() / 3000;
      const d = 0.3 + Math.sin(i * 7 + Date.now() / 500) * 0.5;
      ctx.beginPath();
      ctx.arc(x + Math.cos(a) * rx * d, y + Math.sin(a) * ry * d, 1 + Math.random(), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${region.color}, ${0.3 + Math.random() * 0.4})`;
      ctx.fill();
    }

    // Label
    ctx.fillStyle = `rgba(220, 220, 230, ${0.4 + activity * 0.4})`;
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(region.label, x, y + ry + 14);
  }
}

function getActivity(name: string, spikes: Record<string, number>, cm: number[], maxC: number, state: BrainState): number {
  if (name === "concept") return Math.min(1, cm.filter(v => v > maxC * 0.1).length / Math.max(1, cm.length) * 3);
  if (name === "wm") return Math.min(1, (state.wm_membrane ?? []).filter(v => Math.abs(v) > 0.01).length / 50);
  if (name === "meta") { const m = state.modulators; return Math.min(1, (Math.abs(m.DA ?? 0) + Math.abs(m.NE ?? 0) + Math.abs(m.ACh ?? 0)) / 1.5); }
  return Math.min(1, (spikes[name as keyof typeof spikes] ?? 0) / 100);
}

// ═══════════════════════════════════════════════════
// VIEW 3: SPIKE RASTER — classic neuroscience plot
// ═══════════════════════════════════════════════════

function drawRasterView(ctx: CanvasRenderingContext2D, W: number, H: number, history: number[][]) {
  if (history.length === 0) return;

  const numNeurons = history[0].length;
  const numSteps = history.length;
  const margin = { top: 25, bottom: 20, left: 50, right: 10 };
  const plotW = W - margin.left - margin.right;
  const plotH = H - margin.top - margin.bottom;

  // Title
  ctx.fillStyle = "#6b7280";
  ctx.font = "10px monospace";
  ctx.textAlign = "center";
  ctx.fillText(`Spike Raster — ${numNeurons} concept neurons × ${numSteps} timesteps`, W / 2, 14);

  // Y axis label
  ctx.save();
  ctx.translate(12, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "#4b5563";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Neuron ID", 0, 0);
  ctx.restore();

  // X axis label
  ctx.fillStyle = "#4b5563";
  ctx.font = "9px monospace";
  ctx.textAlign = "center";
  ctx.fillText("Time →", W / 2, H - 4);

  // Draw raster
  const cellW = plotW / numSteps;
  const cellH = plotH / numNeurons;

  for (let t = 0; t < numSteps; t++) {
    const frame = history[t];
    for (let n = 0; n < numNeurons; n++) {
      const v = frame[n];
      if (v < 0.02) continue; // skip silent neurons

      const x = margin.left + t * cellW;
      const y = margin.top + n * cellH;

      // Color intensity based on activation
      const alpha = Math.min(1, v * 1.5);
      ctx.fillStyle = v > 0.5
        ? `rgba(52, 211, 153, ${alpha})`
        : `rgba(52, 160, 130, ${alpha * 0.6})`;
      ctx.fillRect(x, y, Math.max(1.5, cellW), Math.max(1, cellH * 0.8));
    }
  }

  // Y axis ticks
  ctx.fillStyle = "#4b5563";
  ctx.font = "8px monospace";
  ctx.textAlign = "right";
  for (let n = 0; n < numNeurons; n += Math.max(1, Math.floor(numNeurons / 10))) {
    const y = margin.top + n * cellH + cellH / 2;
    ctx.fillText(`${n}`, margin.left - 4, y + 3);
  }

  // Border
  ctx.strokeStyle = "rgba(100, 120, 140, 0.2)";
  ctx.lineWidth = 1;
  ctx.strokeRect(margin.left, margin.top, plotW, plotH);
}
