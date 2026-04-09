export type BrainState = {
  tick: number;
  modulators: Record<string, number>;
  concept_membrane: number[];
  wm_membrane: number[];
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
