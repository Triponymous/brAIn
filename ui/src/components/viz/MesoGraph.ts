// ui/src/components/viz/MesoGraph.ts
// Builds a Meso-level graph: individual neurons within one focused region.

import { REGION_DEFS } from "./constants";
import type { VizNode, VizLink, VizGraph } from "./constants";

export type MesoState = {
  region_spikes?: Record<string, number[]>;
  concept_membrane?: number[];
  spike_counts?: Record<string, number>;
};

/**
 * Build a Meso graph for a single region.
 * Shows individual neurons as small spheres + co-activation edges.
 * Other regions appear as faded background context.
 */
export function buildMesoGraph(state: MesoState, focusRegion: string): VizGraph {
  const nodes: VizNode[] = [];
  const links: VizLink[] = [];

  const regionDef = REGION_DEFS.find((r) => r.id === focusRegion);
  if (!regionDef) return { nodes, links };

  // Background region nodes (faded, for context)
  for (const r of REGION_DEFS) {
    if (r.id === focusRegion) continue;
    nodes.push({
      id: `r_${r.id}`, type: "region", regionId: r.id,
      label: r.label, color: r.color + "30",
      val: 6, activity: 0,
      fx: r.target[0], fy: r.target[1], fz: r.target[2],
    });
  }

  // Individual neurons of the focus region
  const numNeurons = regionDef.neurons;
  const spikes = state.region_spikes?.[focusRegion] ?? [];
  const membrane = focusRegion === "concept" ? (state.concept_membrane ?? []) : [];

  const cx = regionDef.target[0];
  const cy = regionDef.target[1];
  const cz = regionDef.target[2];

  for (let i = 0; i < numNeurons; i++) {
    const spiked = (spikes[i] ?? 0) === 1;
    const mem = membrane[i] ?? 0;
    const activity = spiked ? 1.0 : Math.min(1, Math.abs(mem) * 2);

    // Spherical distribution around region center
    const phi = (i / numNeurons) * Math.PI * 2 * 5; // 5 spirals
    const theta = (i / numNeurons) * Math.PI;
    const r = 30 + (i % 7) * 5;
    const x = cx + r * Math.sin(theta) * Math.cos(phi);
    const y = cy + r * Math.sin(theta) * Math.sin(phi);
    const z = cz + r * Math.cos(theta);

    nodes.push({
      id: `n_${focusRegion}_${i}`, type: "neuron", regionId: focusRegion,
      label: `#${i}`, color: spiked ? "#ffffff" : regionDef.color,
      val: spiked ? 4 : 1.5 + activity * 2, activity,
      fx: x, fy: y, fz: z,
    });
  }

  // Co-activation edges between currently spiking neurons
  const activeIndices = spikes
    .map((v: number, i: number) => ({ i, v }))
    .filter((x: { v: number }) => x.v === 1)
    .slice(0, 10);

  for (let a = 0; a < activeIndices.length; a++) {
    for (let b = a + 1; b < activeIndices.length; b++) {
      links.push({
        source: `n_${focusRegion}_${activeIndices[a].i}`,
        target: `n_${focusRegion}_${activeIndices[b].i}`,
        value: 0.3, color: regionDef.color + "40",
        particles: 1, particleSpeed: 0.005,
      });
    }
  }

  return { nodes, links };
}
