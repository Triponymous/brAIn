// ui/src/components/viz/nodeRenderers.ts
// Custom Three.js node rendering: labels for regions/sensors, default spheres for neurons.

import SpriteText from "three-spritetext";
import type { VizNode } from "./constants";

/**
 * Custom Three.js object per node type.
 * Regions/sensors get text labels. Neurons use default sphere (performance).
 */
export function nodeThreeObject(node: VizNode): any {
  if (node.type === "region") {
    const s = new SpriteText(node.label, 5, node.color);
    s.fontWeight = "bold";
    s.backgroundColor = "rgba(0,0,0,0.6)";
    (s as any).padding = [1.5, 4];
    (s as any).borderRadius = 4;
    return s;
  }
  if (node.type === "sensor") {
    const s = new SpriteText(node.label, 3.5, node.color);
    s.backgroundColor = "rgba(0,0,0,0.4)";
    (s as any).padding = [0.5, 2];
    (s as any).borderRadius = 2;
    return s;
  }
  return undefined as any;
}

/** Node color: bright when active, dimmed when idle, highlight on hover. */
export function nodeColor(
  node: VizNode,
  hoverNodeId: string | null,
  highlightSet: Set<string>,
): string {
  if (!hoverNodeId) {
    return node.activity > 0.05 ? node.color : node.color + "40";
  }
  if (highlightSet.has(node.id)) {
    return node.id === hoverNodeId ? "#ffffff" : node.color;
  }
  return node.color + "15";
}

/** Rich HTML tooltip for node hover. */
export function nodeTooltip(node: VizNode): string {
  const actPct = (node.activity * 100).toFixed(0);
  return `<div style="background:rgba(0,0,0,0.85);padding:8px 12px;border-radius:8px;font-size:13px;color:${node.color};border:1px solid ${node.color}50">
    <b>${node.label}</b>
    ${node.regionId ? `<br/><span style="color:#888;font-size:11px">${node.regionId}</span>` : ""}
    ${node.activity > 0 ? `<br/><span style="color:#6ee7b7;font-size:11px">Aktivitaet: ${actPct}%</span>` : ""}
  </div>`;
}
