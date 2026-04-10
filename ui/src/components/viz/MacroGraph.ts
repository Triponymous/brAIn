// ui/src/components/viz/MacroGraph.ts
// Builds the Macro-level graph: 7 region blobs + 6 sensor nodes + pipeline edges.

import { REGION_DEFS, SENSOR_DEFS, SYNAPSE_PIPELINE, sensorPosition } from "./constants";
import type { VizNode, VizLink, VizGraph } from "./constants";

export type MacroState = {
  sensors?: {
    app?: string; keys?: number; mouse?: number;
    idle?: number; mic_rms?: number;
  };
  spike_counts?: Record<string, number>;
  modulators?: Record<string, number>;
  concept_membrane?: number[];
};

// Long-term concept memory: tracks peak activation per concept across the session.
// Concepts that were once active stay visible (dimmed) even when idle.
const _conceptPeaks = new Map<number, number>();

const MAX_SPIKES: Record<string, number> = {
  sensory: 200, feature: 200, association: 500, concept: 200,
  wm: 100, motor: 50, meta: 10,
};

export function buildMacroGraph(state: MacroState): VizGraph {
  const nodes: VizNode[] = [];
  const links: VizLink[] = [];
  const sensors = state.sensors ?? {};
  const spikes = state.spike_counts ?? {};

  // Which sensors are active?
  const sensorActive: Record<string, boolean> = {
    s_app: !!sensors.app,
    s_keys: (sensors.keys ?? 0) > 0,
    s_mouse: (sensors.mouse ?? 0) > 0,
    s_idle: (sensors.idle ?? 0) < 5,
    s_mic: (sensors.mic_rms ?? 0) > 0.005,
    s_time: true,
  };

  // Sensor nodes — arc left of sensory
  // ALWAYS include all sensor links (active or not) to prevent topology changes
  // that would reset the force graph. Inactive links are just invisible.
  for (const s of SENSOR_DEFS) {
    const active = sensorActive[s.id] ?? false;
    const [fx, fy, fz] = sensorPosition(s.offset);
    nodes.push({
      id: s.id, type: "sensor", label: s.label, color: s.color,
      val: active ? 5 : 2, activity: active ? 0.8 : 0,
      fx, fy, fz,
    });
    // Always create link — just make it invisible when inactive
    links.push({
      source: s.id, target: "r_sensory",
      value: active ? 0.5 : 0.01,
      color: active ? s.color : "rgba(0,0,0,0)",
      particles: active ? 2 : 0,
      particleSpeed: active ? 0.008 : 0,
    });
  }

  // Region nodes — semi-fixed along information flow
  for (const r of REGION_DEFS) {
    const raw = spikes[r.id] ?? 0;
    const maxS = MAX_SPIKES[r.id] ?? 100;
    const activity = Math.min(1, raw / maxS);
    nodes.push({
      id: `r_${r.id}`, type: "region", regionId: r.id,
      label: r.label, color: r.color,
      val: 12 + activity * 10, activity,
      fx: r.target[0], fy: r.target[1], fz: r.target[2],
    });
  }

  // Pipeline edges with spike-driven particles
  for (const [from, to] of SYNAPSE_PIPELINE) {
    const fromSpikes = spikes[from] ?? 0;
    const maxS = MAX_SPIKES[from] ?? 100;
    const v = Math.min(1, fromSpikes / maxS);
    links.push({
      source: `r_${from}`, target: `r_${to}`,
      value: 0.6 + v * 2.5,
      color: `rgba(120,200,180,${(0.2 + v * 0.5).toFixed(2)})`,
      particles: v > 0.05 ? Math.ceil(v * 4) : 0,
      particleSpeed: 0.003 + v * 0.01,
    });
  }

  // ── Concept Neurons (shown around the Concept region) ──
  // Track concepts that have EVER been significantly active this session.
  // This way they stay visible even when the user is idle.
  const cm = state.concept_membrane ?? [];
  if (cm.length > 0) {
    const conceptRegion = REGION_DEFS.find((r) => r.id === "concept")!;
    const cx = conceptRegion.target[0];
    const cy = conceptRegion.target[1];
    const cz = conceptRegion.target[2];

    // Update long-term concept memory (persists across renders)
    for (let i = 0; i < cm.length; i++) {
      const val = Math.abs(cm[i]);
      const prev = _conceptPeaks.get(i) || 0;
      if (val > prev) _conceptPeaks.set(i, val);
      // Slow decay of peaks (half-life ~5 min at 30Hz push rate)
      if (prev > 0.01) _conceptPeaks.set(i, prev * 0.9999);
    }

    // Show concepts that are currently active OR were recently active
    const sorted = Array.from(_conceptPeaks.entries())
      .map(([i, peak]) => ({ i, raw: Math.abs(cm[i] ?? 0), peak }))
      .filter((c) => c.peak > 1.0)  // ever reached significance
      .sort((a, b) => b.peak - a.peak)
      .slice(0, 15);

    for (let idx = 0; idx < sorted.length; idx++) {
      const { i, raw, peak } = sorted[idx];
      const isActive = raw > 0.5;  // currently firing
      const activity = isActive ? Math.min(1, raw / 10) : 0.1;  // dim when sleeping
      const angle = ((i * 137.5) % 360) * (Math.PI / 180);
      const radius = 35 + (i % 5) * 8;

      nodes.push({
        id: `c_${i}`,
        type: "neuron",
        regionId: "concept",
        label: `C${i}${isActive ? " ●" : " ○"} (${raw.toFixed(1)})`,
        color: isActive ? "#fbbf24" : "#fbbf2430",  // bright gold when active, dim when sleeping
        val: isActive ? 2 + activity * 3 : 1.5,     // smaller when sleeping
        activity,
        fx: cx + Math.cos(angle) * radius,
        fy: cy + Math.sin(angle) * radius * 0.7,
        fz: cz + (i % 3 - 1) * 8,
      });

      // Connect FROM region TO concept (spikes flow outward, not back)
      // No particles — concepts are the END of the pipeline, not a source
      links.push({
        source: "r_concept",
        target: `c_${i}`,
        value: activity * 0.3,
        color: `rgba(251,191,36,${(activity * 0.2).toFixed(2)})`,
        particles: 0,
        particleSpeed: 0,
      });
    }
  }

  return { nodes, links };
}

/**
 * Update an existing Macro graph in-place (no topology change).
 * Returns true if topology changed and a full rebuild is needed.
 */
export function updateMacroGraph(
  graph: VizGraph,
  state: MacroState,
  nodeIndex: Map<string, VizNode>,
): boolean {
  if (graph.nodes.length === 0) return true;

  const fresh = buildMacroGraph(state);

  // Only count non-concept nodes for topology comparison.
  // Concept nodes change frequently (top-15 shifts) and should NOT
  // trigger a full graph rebuild — that causes the white screen.
  const coreNodesOld = graph.nodes.filter((n) => n.type !== "neuron").length;
  const coreNodesNew = fresh.nodes.filter((n) => n.type !== "neuron").length;
  if (coreNodesNew !== coreNodesOld) return true;

  // In-place update for ALL existing nodes
  for (const fn of fresh.nodes) {
    const existing = nodeIndex.get(fn.id);
    if (existing) {
      existing.val = fn.val;
      existing.activity = fn.activity;
      existing.color = fn.color;
    }
    // New concept nodes that don't exist yet → skip (will be added on next rebuild)
  }

  // Update link properties (only for existing links)
  const minLen = Math.min(fresh.links.length, graph.links.length);
  for (let i = 0; i < minLen; i++) {
    const fl = fresh.links[i];
    const el = graph.links[i];
    el.value = fl.value;
    el.color = fl.color;
    el.particles = fl.particles;
    el.particleSpeed = fl.particleSpeed;
  }

  return false;
}
