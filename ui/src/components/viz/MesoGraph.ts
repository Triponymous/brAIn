// ui/src/components/viz/MesoGraph.ts
// Builds a Meso-level graph: individual neurons within one focused region.

import { REGION_DEFS } from "./constants";
import type { VizNode, VizLink, VizGraph } from "./constants";

export type MesoState = {
  region_spikes?: Record<string, number[]>;
  concept_membrane?: number[];  // this is actually spike_accum (0-20 range)
  spike_counts?: Record<string, number>;
};

/**
 * Build a Meso graph for a single region.
 * Shows individual neurons with size/brightness proportional to activity.
 */
export function buildMesoGraph(state: MesoState, focusRegion: string): VizGraph {
  const nodes: VizNode[] = [];
  const links: VizLink[] = [];

  const regionDef = REGION_DEFS.find((r) => r.id === focusRegion);
  if (!regionDef) return { nodes, links };

  const numNeurons = regionDef.neurons;
  const membrane = focusRegion === "concept" ? (state.concept_membrane ?? []) : [];

  // Find max membrane value for relative scaling
  const maxMem = Math.max(0.1, ...membrane.map(Math.abs));

  const cx = regionDef.target[0];
  const cy = regionDef.target[1];
  const cz = regionDef.target[2];

  // Only show neurons with some activity (skip completely silent ones)
  // This makes the view cleaner and more meaningful
  const neuronData: { i: number; activity: number }[] = [];
  for (let i = 0; i < numNeurons; i++) {
    const mem = membrane[i] ?? 0;
    const activity = Math.abs(mem) / maxMem;  // normalize to 0-1 relative to strongest
    neuronData.push({ i, activity });
  }

  // Sort by activity, show top 50 most active + any with activity > 10%
  neuronData.sort((a, b) => b.activity - a.activity);
  const visibleNeurons = neuronData.filter((n, idx) => idx < 50 || n.activity > 0.1);

  for (let rank = 0; rank < visibleNeurons.length; rank++) {
    const { i, activity } = visibleNeurons[rank];

    // Spherical layout: most active in center, less active further out
    const ringIdx = Math.floor(rank / 8);
    const angleIdx = rank % 8;
    const radius = 20 + ringIdx * 15;
    const angle = (angleIdx / 8) * Math.PI * 2 + ringIdx * 0.3; // slight rotation per ring

    const x = cx + radius * Math.cos(angle);
    const y = cy + radius * Math.sin(angle) * 0.7; // slightly flattened
    const z = cz + (Math.random() - 0.5) * 10; // slight depth variation

    // Color: bright for active, dim for inactive
    const brightness = Math.floor(80 + activity * 175);
    const color = activity > 0.5
      ? `rgb(${brightness}, 255, ${brightness})`  // green-white for strong
      : activity > 0.1
        ? regionDef.color  // region color for moderate
        : regionDef.color + "60";  // dim for weak

    nodes.push({
      id: `n_${focusRegion}_${i}`,
      type: "neuron",
      regionId: focusRegion,
      label: `#${i} (${(activity * 100).toFixed(0)}%)`,
      color,
      val: 2 + activity * 6,  // size 2-8 based on activity
      activity,
      fx: x, fy: y, fz: z,
    });
  }

  // Connect top-10 most active neurons to each other
  const topActive = visibleNeurons.slice(0, 10);
  for (let a = 0; a < topActive.length; a++) {
    for (let b = a + 1; b < topActive.length; b++) {
      const strength = (topActive[a].activity + topActive[b].activity) / 2;
      if (strength > 0.3) {
        links.push({
          source: `n_${focusRegion}_${topActive[a].i}`,
          target: `n_${focusRegion}_${topActive[b].i}`,
          value: strength * 0.5,
          color: regionDef.color + "50",
          particles: strength > 0.6 ? 1 : 0,
          particleSpeed: 0.005,
        });
      }
    }
  }

  return { nodes, links };
}
