import { useRef, useEffect } from "react";
import type { BrainState } from "../lib/ws";

/**
 * Brain center panel — per design doc:
 * 1. Region graph (animated, 7 nodes, modulator-colored)
 * 2. Spike raster (5s window)
 * 3. Concept cloud (top-12 with labels, click-to-label)
 */

type ExtendedState = BrainState & {
  sensors?: { app?: string; keys?: number; mouse?: number; idle?: number; mic_rms?: number };
  spike_counts?: { sensory?: number; feature?: number; concept?: number };
};

// ── Region graph layout ──
type RegionNode = { name: string; label: string; x: number; y: number; color: string };

const REGION_NODES: RegionNode[] = [
  { name: "sensory",     label: "Sensory",     x: 0.08, y: 0.5,  color: "#6ee7b7" },
  { name: "feature",     label: "Feature",     x: 0.25, y: 0.3,  color: "#67e8f9" },
  { name: "association", label: "Association",  x: 0.42, y: 0.5,  color: "#c084fc" },
  { name: "concept",     label: "Concept",      x: 0.60, y: 0.3,  color: "#34d399" },
  { name: "wm",          label: "Work. Mem",    x: 0.60, y: 0.7,  color: "#a78bfa" },
  { name: "motor",       label: "Motor",        x: 0.78, y: 0.3,  color: "#fbbf24" },
  { name: "meta",        label: "Meta",         x: 0.78, y: 0.7,  color: "#f87171" },
];

const EDGES: [string, string][] = [
  ["sensory", "feature"],
  ["feature", "association"],
  ["association", "concept"],
  ["concept", "wm"],
  ["concept", "motor"],
];

// ── Spike raster history ──
const RASTER_FRAMES = 150; // ~5s at 30fps push rate
let rasterHistory: number[][] = [];

export function BrainPanel({ state }: { state: BrainState | null }) {
  const regionRef = useRef<HTMLCanvasElement>(null);
  const rasterRef = useRef<HTMLCanvasElement>(null);

  // Region graph
  useEffect(() => {
    if (!state || !regionRef.current) return;
    const canvas = regionRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;

    const s = state as ExtendedState;
    const spikes = s.spike_counts ?? {};
    const mods = state.modulators;

    ctx.fillStyle = "#0a0b10";
    ctx.fillRect(0, 0, W, H);

    // ── Draw edges with signal flow ──
    for (const [from, to] of EDGES) {
      const n1 = REGION_NODES.find(r => r.name === from)!;
      const n2 = REGION_NODES.find(r => r.name === to)!;
      const x1 = n1.x * W, y1 = n1.y * H;
      const x2 = n2.x * W, y2 = n2.y * H;

      const flow = Math.min(1, (spikes[from as keyof typeof spikes] ?? 0) / 150);
      const alpha = 0.08 + flow * 0.5;

      // Edge line
      ctx.strokeStyle = `rgba(150, 220, 200, ${alpha})`;
      ctx.lineWidth = 1 + flow * 3;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Animated signal dots
      if (flow > 0.02) {
        const numDots = 1 + Math.floor(flow * 4);
        for (let d = 0; d < numDots; d++) {
          const t = ((Date.now() / 600 + d / numDots) % 1);
          const px = x1 + (x2 - x1) * t;
          const py = y1 + (y2 - y1) * t;
          ctx.beginPath();
          ctx.arc(px, py, 2 + flow * 2, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(52, 211, 153, ${0.4 + flow * 0.5})`;
          ctx.fill();
        }
      }

      // Edge label
      ctx.fillStyle = `rgba(150, 160, 170, ${0.3 + flow * 0.4})`;
      ctx.font = "8px monospace";
      ctx.textAlign = "center";
      const mx = (x1 + x2) / 2, my = (y1 + y2) / 2 - 6;
      ctx.fillText(`${spikes[from as keyof typeof spikes] ?? 0}→`, mx, my);
    }

    // ── Draw region nodes ──
    for (const node of REGION_NODES) {
      const x = node.x * W, y = node.y * H;
      const activity = getNodeActivity(node.name, spikes, state);
      const r = 22 + activity * 18;

      // Outer glow
      if (activity > 0.05) {
        const grad = ctx.createRadialGradient(x, y, r * 0.5, x, y, r * 2.5);
        grad.addColorStop(0, node.color + Math.floor(activity * 60).toString(16).padStart(2, '0'));
        grad.addColorStop(1, node.color + "00");
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(x, y, r * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Node circle
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      const bodyAlpha = 0.15 + activity * 0.6;
      ctx.fillStyle = node.color + Math.floor(bodyAlpha * 255).toString(16).padStart(2, '0');
      ctx.fill();
      ctx.strokeStyle = node.color + "80";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // Node label
      ctx.fillStyle = "#e5e7eb";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(node.label, x, y + 3);

      // Activity value
      ctx.fillStyle = "#9ca3af";
      ctx.font = "8px monospace";
      ctx.fillText(`${Math.round(activity * 100)}%`, x, y + 14);
    }
  }, [state]);

  // Spike raster
  useEffect(() => {
    if (!state || !rasterRef.current) return;
    const canvas = rasterRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width, H = rect.height;

    const conceptMem = state.concept_membrane ?? [];
    const maxC = Math.max(0.01, ...conceptMem.map(Math.abs));
    const normalized = conceptMem.map(v => Math.abs(v) / maxC);

    rasterHistory.push(normalized);
    if (rasterHistory.length > RASTER_FRAMES) rasterHistory.shift();

    ctx.fillStyle = "#0a0b10";
    ctx.fillRect(0, 0, W, H);

    const n = conceptMem.length;
    const frames = rasterHistory.length;
    if (n === 0 || frames === 0) return;

    const cellW = W / RASTER_FRAMES;
    const cellH = H / n;

    for (let t = 0; t < frames; t++) {
      for (let i = 0; i < n; i++) {
        const v = rasterHistory[t][i];
        if (v < 0.03) continue;
        ctx.fillStyle = v > 0.3
          ? `rgba(52, 211, 153, ${Math.min(1, v * 1.5)})`
          : `rgba(40, 80, 70, ${v * 2})`;
        ctx.fillRect(t * cellW, i * cellH, Math.max(1.5, cellW), Math.max(0.8, cellH * 0.9));
      }
    }
  }, [state]);

  if (!state) {
    return <div className="h-full flex items-center justify-center text-gray-500">Waiting for brain state...</div>;
  }

  const s = state as ExtendedState;
  const conceptMem = state.concept_membrane ?? [];
  const maxC = Math.max(0.01, ...conceptMem.map(Math.abs));

  // Top-12 concepts for the concept cloud
  const indexed = conceptMem.map((v, i) => ({ id: i, activation: Math.abs(v) / maxC }));
  indexed.sort((a, b) => b.activation - a.activation);
  const top12 = indexed.slice(0, 12).filter(c => c.activation > 0.02);

  return (
    <div className="h-full flex flex-col gap-2 overflow-hidden">
      {/* Region graph */}
      <div className="flex-[3] min-h-0">
        <canvas ref={regionRef} className="w-full h-full rounded-lg" style={{ background: "#0a0b10" }} />
      </div>

      {/* Concept cloud */}
      <div className="flex-[1] min-h-0 bg-gray-900/50 rounded-lg p-2 overflow-y-auto">
        <div className="text-gray-500 text-[10px] font-mono uppercase tracking-wider mb-1">
          Concept Cloud — Top {top12.length} active
        </div>
        <div className="flex flex-wrap gap-1.5">
          {top12.map(c => (
            <button
              key={c.id}
              className="px-2 py-0.5 rounded-full text-[11px] font-mono border transition-all hover:scale-105"
              style={{
                borderColor: `rgba(52, 211, 153, ${0.3 + c.activation * 0.7})`,
                backgroundColor: `rgba(52, 211, 153, ${c.activation * 0.2})`,
                color: `rgba(220, 240, 230, ${0.5 + c.activation * 0.5})`,
              }}
              title={`Concept #${c.id} — activation: ${(c.activation * 100).toFixed(0)}% — click to label`}
              onClick={() => {
                const label = prompt(`Label for Concept #${c.id}:`);
                if (label) {
                  fetch('/api/label', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ concept_id: c.id, label }),
                  });
                }
              }}
            >
              #{c.id} <span className="text-gray-500 ml-0.5">{(c.activation * 100).toFixed(0)}%</span>
            </button>
          ))}
          {top12.length === 0 && <span className="text-gray-600 text-[10px]">No active concepts</span>}
        </div>
      </div>

      {/* Spike raster */}
      <div className="flex-[1.5] min-h-0">
        <div className="text-gray-500 text-[10px] font-mono uppercase tracking-wider mb-0.5 px-1">
          Spike Raster — {conceptMem.length} neurons × 5s
        </div>
        <canvas ref={rasterRef} className="w-full h-full rounded-lg" style={{ background: "#0a0b10" }} />
      </div>
    </div>
  );
}

function getNodeActivity(name: string, spikes: Record<string, number>, state: BrainState): number {
  if (name === "concept") {
    const cm = state.concept_membrane ?? [];
    const maxC = Math.max(0.01, ...cm.map(Math.abs));
    return Math.min(1, cm.filter(v => Math.abs(v) / maxC > 0.1).length / Math.max(1, cm.length) * 3);
  }
  if (name === "wm") {
    const wm = state.wm_membrane ?? [];
    return Math.min(1, wm.filter(v => Math.abs(v) > 0.01).length / Math.max(1, wm.length) * 2);
  }
  if (name === "meta") {
    const m = state.modulators;
    return Math.min(1, (Math.abs(m.DA ?? 0) + Math.abs(m.NE ?? 0) + Math.abs(m.ACh ?? 0)) / 1.5);
  }
  return Math.min(1, (spikes[name as keyof typeof spikes] ?? 0) / 150);
}
