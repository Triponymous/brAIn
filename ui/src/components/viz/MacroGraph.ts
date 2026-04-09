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

  // ── Active Concept Neurons (shown around the Concept region) ──
  const cm = state.concept_membrane ?? [];
  if (cm.length > 0) {
    const maxMem = Math.max(0.1, ...cm.map(Math.abs));
    const conceptRegion = REGION_DEFS.find((r) => r.id === "concept")!;
    const cx = conceptRegion.target[0];
    const cy = conceptRegion.target[1];
    const cz = conceptRegion.target[2];

    // Show top 20 most active concepts as small nodes
    const ranked = cm
      .map((v, i) => ({ i, activity: Math.abs(v) / maxMem }))
      .sort((a, b) => b.activity - a.activity)
      .slice(0, 20)
      .filter((c) => c.activity > 0.05);

    for (let rank = 0; rank < ranked.length; rank++) {
      const { i, activity } = ranked[rank];
      const angle = (rank / Math.max(1, ranked.length)) * Math.PI * 2;
      const radius = 30 + Math.floor(rank / 8) * 15;

      nodes.push({
        id: `c_${i}`,
        type: "neuron",
        regionId: "concept",
        label: `C${i} (${(activity * 100).toFixed(0)}%)`,
        color: activity > 0.7 ? "#ffffff" : activity > 0.3 ? "#fbbf24" : "#fbbf2460",
        val: 1.5 + activity * 4,
        activity,
        fx: cx + Math.cos(angle) * radius,
        fy: cy + Math.sin(angle) * radius * 0.7,
        fz: cz + (rank % 3 - 1) * 8,
      });

      // Connect to concept region
      links.push({
        source: `c_${i}`,
        target: "r_concept",
        value: activity * 0.3,
        color: `rgba(251,191,36,${(activity * 0.3).toFixed(2)})`,
        particles: activity > 0.5 ? 1 : 0,
        particleSpeed: 0.005,
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
  if (fresh.nodes.length !== graph.nodes.length) return true;

  // In-place property update — O(n) with Map lookup
  for (const fn of fresh.nodes) {
    const existing = nodeIndex.get(fn.id);
    if (!existing) return true;
    existing.val = fn.val;
    existing.activity = fn.activity;
    existing.color = fn.color;
  }

  // Update link properties
  const minLen = Math.min(fresh.links.length, graph.links.length);
  if (fresh.links.length !== graph.links.length) return true;
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
