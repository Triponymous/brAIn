import { useRef, useEffect, useState, useCallback, useMemo, lazy, Suspense } from "react";
import type { BrainState } from "../lib/ws";

// Lazy import — react-force-graph-2d uses `window` at import time
const ForceGraph2D = lazy(() => import("react-force-graph-2d"));

/**
 * Brain visualization as a force-directed knowledge graph (Obsidian-style).
 *
 * Nodes:
 *   - Sensor nodes (fixed left): Mic, Keys, Mouse, App, Idle, Time
 *   - Region nodes (larger): Sensory, Feature, Association, Concept, WM, Motor
 *   - Concept neurons (small, appear when active, fade when inactive)
 *
 * Links:
 *   - Sensor → Sensory region (when sensor is active)
 *   - Region → Region (pipeline flow, thickness = spike count)
 *   - Concept ↔ Concept (co-active concepts link together)
 *
 * The graph starts nearly empty and grows as the brain learns.
 */

type ExtendedState = BrainState & {
  sensors?: { app?: string; keys?: number; mouse?: number; idle?: number; mic_rms?: number };
  spike_counts?: { sensory?: number; feature?: number; concept?: number };
};

type GNode = {
  id: string;
  type: "sensor" | "region" | "concept";
  label: string;
  val: number;      // size
  color: string;
  active: boolean;
  fx?: number;       // fixed x (for sensors)
  fy?: number;
};

type GLink = {
  source: string;
  target: string;
  value: number;     // thickness
  color: string;
};

// Track which concepts have ever been seen
const seenConcepts = new Set<number>();

export function BrainPanel({ state }: { state: BrainState | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [dims, setDims] = useState({ w: 800, h: 500 });

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([e]) => {
      setDims({ w: e.contentRect.width, h: e.contentRect.height });
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  const graphData = useMemo(() => {
    if (!state) return { nodes: [], links: [] };
    return buildGraph(state as ExtendedState, dims.w, dims.h);
  }, [state, dims]);

  const paintNode = useCallback((node: any, ctx: CanvasRenderingContext2D) => {
    const n = node as GNode;
    const x = node.x as number;
    const y = node.y as number;
    // Guard against NaN/Infinity before any canvas ops
    if (!isFinite(x) || !isFinite(y)) return;
    const r = Math.max(1, Math.sqrt(Math.max(0, n.val)) * 2);

    // Glow for active nodes
    if (n.active && n.type !== "concept" && r > 0) {
      const grad = ctx.createRadialGradient(x, y, r, x, y, r * 3);
      grad.addColorStop(0, n.color + "40");
      grad.addColorStop(1, n.color + "00");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(x, y, r * 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Node body
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = n.active ? n.color : n.color + "30";
    ctx.fill();

    if (n.type !== "concept") {
      ctx.strokeStyle = n.color + "80";
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Label
    if (n.type !== "concept" || n.active) {
      ctx.fillStyle = n.active ? "#e5e7eb" : "#4b556380";
      ctx.font = n.type === "region" ? "bold 9px sans-serif" : n.type === "sensor" ? "8px sans-serif" : "7px monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      if (n.type === "region") {
        ctx.fillText(n.label, x, y);
      } else {
        ctx.fillText(n.label, x, y + r + 8);
      }
    }
  }, []);

  const paintLink = useCallback((link: any, ctx: CanvasRenderingContext2D) => {
    const src = link.source as any;
    const tgt = link.target as any;
    if (!isFinite(src.x) || !isFinite(src.y) || !isFinite(tgt.x) || !isFinite(tgt.y)) return;

    ctx.strokeStyle = link.color || "rgba(52, 211, 153, 0.1)";
    ctx.lineWidth = Math.max(0.3, link.value * 3);
    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.stroke();

    // Animated particle along active links
    if (link.value > 0.1) {
      const t = (Date.now() / 800) % 1;
      const px = src.x + (tgt.x - src.x) * t;
      const py = src.y + (tgt.y - src.y) * t;
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(52, 211, 153, 0.6)";
      ctx.fill();
    }
  }, []);

  if (!state) {
    return <div className="h-full flex items-center justify-center text-gray-500">Waiting for brain state...</div>;
  }

  return (
    <div ref={containerRef} className="h-full w-full rounded-lg overflow-hidden" style={{ background: "#08090d" }}>
      <Suspense fallback={<div className="flex items-center justify-center h-full text-gray-500">Loading graph...</div>}>
        <ForceGraph2D
          width={dims.w}
          height={dims.h}
          graphData={graphData}
          nodeCanvasObject={paintNode}
          linkCanvasObject={paintLink}
          nodeRelSize={4}
          linkDirectionalParticles={0}
          cooldownTicks={50}
          d3AlphaDecay={0.05}
          d3VelocityDecay={0.3}
          backgroundColor="#08090d"
          onNodeClick={(node: any) => {
            if (node.type === "concept") {
              const label = prompt(`Label for ${node.id}:`);
              if (label) {
                fetch("/api/label", {
                  method: "POST",
                  headers: { "Content-Type": "application/json" },
                  body: JSON.stringify({ concept_id: parseInt(node.id.replace("c", "")), label }),
                });
              }
            }
          }}
        />
      </Suspense>
    </div>
  );
}

function buildGraph(state: ExtendedState, W: number, H: number): { nodes: GNode[]; links: GLink[] } {
  const nodes: GNode[] = [];
  const links: GLink[] = [];
  const sensors = state.sensors ?? {};
  const spikes = state.spike_counts ?? {};
  const conceptMem = state.concept_membrane ?? [];
  const maxC = Math.max(0.01, ...conceptMem.map(Math.abs));
  const mods = state.modulators;

  // ── Sensor nodes (fixed left column) ──
  const sensorDefs = [
    { id: "s_app",   label: sensors.app || "App",   active: !!(sensors.app),                    color: "#60a5fa" },
    { id: "s_keys",  label: `⌨ ${sensors.keys ?? 0}`,  active: (sensors.keys ?? 0) > 0,        color: "#34d399" },
    { id: "s_mouse", label: `🖱 ${sensors.mouse ?? 0}`, active: (sensors.mouse ?? 0) > 0,       color: "#a78bfa" },
    { id: "s_idle",  label: `💤 ${sensors.idle?.toFixed(0) ?? 0}s`, active: (sensors.idle ?? 0) < 5, color: "#fbbf24" },
    { id: "s_mic",   label: `🎤 ${(sensors.mic_rms ?? 0).toFixed(3)}`, active: (sensors.mic_rms ?? 0) > 0.005, color: "#f87171" },
    { id: "s_time",  label: "⏰ Time",                  active: true,                             color: "#818cf8" },
  ];

  sensorDefs.forEach((s, i) => {
    nodes.push({
      id: s.id, type: "sensor", label: s.label, color: s.color,
      val: s.active ? 8 : 4, active: s.active,
      fx: 40, fy: 60 + i * ((H - 80) / 6),
    });
  });

  // ── Region nodes ──
  const regionDefs = [
    { id: "r_sensory",     label: "Sensory",     color: "#6ee7b7", activity: Math.min(1, (spikes.sensory ?? 0) / 100) },
    { id: "r_feature",     label: "Feature",     color: "#67e8f9", activity: Math.min(1, (spikes.feature ?? 0) / 150) },
    { id: "r_association", label: "Association",  color: "#c084fc", activity: Math.min(1, (spikes.feature ?? 0) / 200) },
    { id: "r_concept",     label: "Concept",      color: "#34d399", activity: Math.min(1, (spikes.concept ?? 0) / 10) },
    { id: "r_wm",          label: "Memory",       color: "#a78bfa", activity: 0.3 },
    { id: "r_motor",       label: "Motor",        color: "#fbbf24", activity: 0.2 },
  ];

  regionDefs.forEach((r, i) => {
    nodes.push({
      id: r.id, type: "region", label: r.label, color: r.color,
      val: 20 + r.activity * 30, active: r.activity > 0.05,
    });
  });

  // ── Sensor → Sensory links ──
  sensorDefs.forEach(s => {
    if (s.active) {
      links.push({
        source: s.id, target: "r_sensory",
        value: 0.3, color: s.color + "40",
      });
    }
  });

  // ── Region → Region pipeline links ──
  const pipeline: [string, string, number][] = [
    ["r_sensory", "r_feature", spikes.sensory ?? 0],
    ["r_feature", "r_association", spikes.feature ?? 0],
    ["r_association", "r_concept", spikes.feature ?? 0],
    ["r_concept", "r_wm", spikes.concept ?? 0],
    ["r_concept", "r_motor", spikes.concept ?? 0],
  ];

  pipeline.forEach(([from, to, count]) => {
    const v = Math.min(1, count / 150);
    links.push({
      source: from, target: to,
      value: 0.05 + v * 0.8,
      color: `rgba(120, 200, 180, ${0.1 + v * 0.5})`,
    });
  });

  // ── Concept neurons (only show active ones — they APPEAR as brain learns) ──
  const indexed = conceptMem.map((v, i) => ({ i, v: Math.abs(v) / maxC }));
  indexed.sort((a, b) => b.v - a.v);
  const activeConcepts = indexed.filter(c => c.v > 0.05).slice(0, 30);

  activeConcepts.forEach(c => {
    seenConcepts.add(c.i);
    nodes.push({
      id: `c${c.i}`, type: "concept",
      label: `#${c.i}`,
      color: "#34d399",
      val: 2 + c.v * 8,
      active: c.v > 0.1,
    });
    // Link to Concept region
    links.push({
      source: "r_concept", target: `c${c.i}`,
      value: c.v * 0.5,
      color: `rgba(52, 211, 153, ${c.v * 0.4})`,
    });
  });

  // ── Concept ↔ Concept links (co-active = connected) ──
  const topConcepts = activeConcepts.slice(0, 10);
  for (let i = 0; i < topConcepts.length; i++) {
    for (let j = i + 1; j < topConcepts.length; j++) {
      const strength = (topConcepts[i].v + topConcepts[j].v) / 2;
      if (strength > 0.2) {
        links.push({
          source: `c${topConcepts[i].i}`, target: `c${topConcepts[j].i}`,
          value: strength * 0.3,
          color: `rgba(52, 211, 153, ${strength * 0.2})`,
        });
      }
    }
  }

  return { nodes, links };
}
