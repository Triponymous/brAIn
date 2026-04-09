// ui/src/components/viz/useBrainSubscription.ts
// Manages WebSocket subscriptions based on current zoom level.

import { useEffect, useRef } from "react";
import { sendWS } from "../../lib/ws";
import type { ZoomLevel } from "./constants";

/**
 * Sends subscription messages when zoom level or focus changes.
 * The backend uses these to decide which detail data to include in pushes.
 */
export function useBrainSubscription(
  zoomLevel: ZoomLevel,
  focusRegion: string | null,
  focusNeuron: number | null,
): void {
  const prevRef = useRef<string>("");

  useEffect(() => {
    const key = `${zoomLevel}:${focusRegion}:${focusNeuron}`;
    if (key === prevRef.current) return;
    prevRef.current = key;

    if (zoomLevel === "macro") {
      sendWS({ subscribe: "macro" });
    } else if (zoomLevel === "meso" && focusRegion) {
      sendWS({ subscribe: "meso", region: focusRegion });
    } else if (zoomLevel === "micro" && focusRegion && focusNeuron != null) {
      sendWS({ subscribe: "micro", region: focusRegion, neuron_id: focusNeuron });
    }
  }, [zoomLevel, focusRegion, focusNeuron]);
}
