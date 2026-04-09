export type BrainState = {
  tick: number;
  modulators: Record<string, number>;
  concept_membrane: number[];
  wm_membrane: number[];
  // Extended fields (present depending on subscription)
  sensors?: {
    app?: string; keys?: number; mouse?: number;
    idle?: number; mic_rms?: number;
    background_apps?: string[]; app_count?: number; app_switched?: boolean;
    switch_rate?: number;
  };
  spike_counts?: Record<string, number>;
  region_spikes?: Record<string, number[]>;
  synapse_activity?: Record<string, { mean_weight: number; active_connections: number[][] }>;
};

type Listener = (state: BrainState) => void;

let ws: WebSocket | null = null;
let listeners: Set<Listener> = new Set();
let latestState: BrainState | null = null;

export function connectWS() {
  const url = `ws://${window.location.host}/ws`;
  ws = new WebSocket(url);
  ws.onmessage = (ev) => {
    try {
      const state: BrainState = JSON.parse(ev.data);
      latestState = state;
      listeners.forEach((fn) => fn(state));
    } catch {}
  };
  ws.onclose = () => {
    setTimeout(connectWS, 2000);
  };
}

export function subscribe(fn: Listener): () => void {
  listeners.add(fn);
  if (latestState) fn(latestState);
  return () => listeners.delete(fn);
}

export function getLatest(): BrainState | null {
  return latestState;
}

/** Send a message to the WebSocket server (e.g. subscription changes). */
export function sendWS(msg: Record<string, unknown>): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}
