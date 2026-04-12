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
  sensory: 200, concept: 1000, wm: 100,
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

  // Region nodes — all are active (we removed the unused ones)
  const activeRegions = new Set(["sensory", "concept", "wm"]);

  for (const r of REGION_DEFS) {
    const raw = spikes[r.id] ?? 0;
    const maxS = MAX_SPIKES[r.id] ?? 100;
    const activity = Math.min(1, raw / maxS);
    const isActive = activeRegions.has(r.id) || raw > 0;
    nodes.push({
      id: `r_${r.id}`, type: "region", regionId: r.id,
      label: r.label,
      color: isActive ? r.color : r.color + "30",
      val: isActive ? 12 + activity * 10 : 5,
      activity: isActive ? activity : 0.1,
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

  // ── Concept Clusters (from ConceptTracker — stable IDs!) ──
  const concepts = (state as any).concepts;
  if (concepts && concepts.clusters) {
    const conceptRegion = REGION_DEFS.find((r) => r.id === "concept")!;
    const cx = conceptRegion.target[0];
    const cy = conceptRegion.target[1];
    const cz = conceptRegion.target[2];
    const currentCluster = concepts.current_cluster;

    const clusters = concepts.clusters as Array<{
      id: number; label: string | null; count: number; active: boolean;
    }>;

    // Find max count for relative sizing
    const maxCount = Math.max(1, ...clusters.map((c) => c.count));

    for (let idx = 0; idx < Math.min(clusters.length, 15); idx++) {
      const c = clusters[idx];
      const isActive = c.id === currentCluster;
      // Activity based on how often this cluster has been seen (relative to most-seen)
      const relativeSize = c.count / maxCount;
      const activity = isActive ? 1.0 : relativeSize * 0.6;
      const angle = ((c.id * 137.5) % 360) * (Math.PI / 180);
      const radius = 30 + (idx % 4) * 12;

      const displayLabel = c.label
        ? `${c.label} (${c.count}x)${isActive ? " ●" : ""}`
        : `Muster #${c.id} (${c.count}x)${isActive ? " ●" : ""}`;

      nodes.push({
        id: `cl_${c.id}`,
        type: "neuron",
        regionId: "concept",
        label: displayLabel,
        // Active = bright gold, labeled = medium gold, unknown = dim
        color: isActive ? "#fbbf24" : c.label ? `rgba(251,191,36,${0.3 + relativeSize * 0.5})` : "#fbbf2430",
        // Size scales with count — frequently seen clusters are bigger
        val: isActive ? 4 + relativeSize * 3 : 1.5 + relativeSize * 3,
        activity,
        fx: cx + Math.cos(angle) * radius,
        fy: cy + Math.sin(angle) * radius * 0.7,
        fz: cz + (c.id % 3 - 1) * 8,
      });

      links.push({
        source: "r_concept",
        target: `cl_${c.id}`,
        value: isActive ? 0.5 : 0.1,
        color: isActive ? "rgba(251,191,36,0.4)" : "rgba(251,191,36,0.1)",
        particles: isActive ? 2 : 0,
        particleSpeed: 0.008,
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
