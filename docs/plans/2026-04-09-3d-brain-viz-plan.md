# 3D Brain Visualization Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current BrainPanel3D with a Progressive Disclosure 3D force graph (Macro/Meso/Micro zoom levels) using react-force-graph-3d.

**Architecture:** Three zoom levels backed by separate graph builders. Backend WebSocket extended with subscription-based detail data. In-place mutation for property updates, graphData() only on topology changes. Semi-fixed region positions preserve mental map.

**Tech Stack:** react-force-graph-3d, Three.js (MeshStandardMaterial + UnrealBloomPass), TypeScript, FastAPI WebSocket, Python

---

### Task 1: Constants & Types Module

**Files:**
- Create: `ui/src/components/viz/constants.ts`

**Step 1: Create the constants file**

```typescript
// ui/src/components/viz/constants.ts

export const REGION_DEFS = [
  { id: "sensory",     label: "Sensorik",         color: "#6ee7b7", neurons: 200, target: [-120, -120, 0] },
  { id: "feature",     label: "Mustererkennung",   color: "#34d399", neurons: 200, target: [-60,  -60, 0] },
  { id: "association",  label: "Verknuepfung",     color: "#a78bfa", neurons: 500, target: [0,      0, 0] },
  { id: "concept",     label: "Konzeptbildung",    color: "#fbbf24", neurons: 200, target: [60,    60, 0] },
  { id: "wm",          label: "Gedaechtnis",       color: "#60a5fa", neurons: 100, target: [60,     0, -40] },
  { id: "motor",       label: "Motorik",           color: "#f87171", neurons: 50,  target: [120,  100, 0] },
  { id: "meta",        label: "Meta",              color: "#9ca3af", neurons: 10,  target: [0,    150, 0] },
] as const;

export const SENSOR_DEFS = [
  { id: "s_app",   label: "App",      color: "#60a5fa", offset: 0 },
  { id: "s_keys",  label: "Tastatur", color: "#34d399", offset: 1 },
  { id: "s_mouse", label: "Maus",     color: "#a78bfa", offset: 2 },
  { id: "s_idle",  label: "Idle",     color: "#fbbf24", offset: 3 },
  { id: "s_mic",   label: "Mikrofon", color: "#f87171", offset: 4 },
  { id: "s_time",  label: "Zeit",     color: "#818cf8", offset: 5 },
] as const;

// Sensor arc: positioned left of sensory region
export function sensorPosition(offset: number): [number, number, number] {
  return [-180, -120 + offset * 30, 0];
}

export const SYNAPSE_PIPELINE: [string, string][] = [
  ["sensory", "feature"],
  ["feature", "association"],
  ["association", "concept"],
  ["concept", "wm"],
  ["concept", "motor"],
];

export type ZoomLevel = "macro" | "meso" | "micro";

export type VizNode = {
  id: string;
  type: "sensor" | "region" | "neuron";
  regionId?: string;
  label: string;
  val: number;
  color: string;
  activity: number;
  fx?: number;
  fy?: number;
  fz?: number;
  // Three.js runtime fields (mutated in-place)
  x?: number;
  y?: number;
  z?: number;
};

export type VizLink = {
  source: string;
  target: string;
  value: number;
  color: string;
  particles: number;
  particleSpeed: number;
};

export type VizGraph = { nodes: VizNode[]; links: VizLink[] };
```

**Step 2: Verify it compiles**

Run: `cd ui && npx tsc --noEmit src/components/viz/constants.ts`
Expected: no errors (or run `npx tsc --noEmit` from ui/)

**Step 3: Commit**

```bash
git add ui/src/components/viz/constants.ts
git commit -m "feat(viz): add constants and types for 3D brain visualization"
```

---

### Task 2: Macro Graph Builder

**Files:**
- Create: `ui/src/components/viz/MacroGraph.ts`

**Step 1: Create the Macro graph builder**

```typescript
// ui/src/components/viz/MacroGraph.ts
import { REGION_DEFS, SENSOR_DEFS, SYNAPSE_PIPELINE, sensorPosition } from "./constants";
import type { VizNode, VizLink, VizGraph } from "./constants";

export type MacroState = {
  sensors?: {
    app?: string; keys?: number; mouse?: number;
    idle?: number; mic_rms?: number;
  };
  spike_counts?: Record<string, number>;
  modulators?: Record<string, number>;
};

const MAX_SPIKES: Record<string, number> = {
  sensory: 200, feature: 200, association: 500, concept: 200, wm: 100, motor: 50, meta: 10,
};

export function buildMacroGraph(state: MacroState): VizGraph {
  const nodes: VizNode[] = [];
  const links: VizLink[] = [];
  const sensors = state.sensors ?? {};
  const spikes = state.spike_counts ?? {};

  // Sensor nodes
  const sensorActive: Record<string, boolean> = {
    s_app: !!sensors.app,
    s_keys: (sensors.keys ?? 0) > 0,
    s_mouse: (sensors.mouse ?? 0) > 0,
    s_idle: (sensors.idle ?? 0) < 5,
    s_mic: (sensors.mic_rms ?? 0) > 0.005,
    s_time: true,
  };

  SENSOR_DEFS.forEach((s) => {
    const active = sensorActive[s.id] ?? false;
    const [fx, fy, fz] = sensorPosition(s.offset);
    nodes.push({
      id: s.id, type: "sensor", label: s.label, color: s.color,
      val: active ? 5 : 2, activity: active ? 0.8 : 0,
      fx, fy, fz,
    });
    if (active) {
      links.push({
        source: s.id, target: "r_sensory",
        value: 0.5, color: s.color,
        particles: 2, particleSpeed: 0.008,
      });
    }
  });

  // Region nodes — semi-fixed positions
  REGION_DEFS.forEach((r) => {
    const raw = spikes[r.id] ?? 0;
    const activity = Math.min(1, raw / (MAX_SPIKES[r.id] ?? 100));
    nodes.push({
      id: `r_${r.id}`, type: "region", regionId: r.id,
      label: r.label, color: r.color,
      val: 12 + activity * 10, activity,
      fx: r.target[0], fy: r.target[1], fz: r.target[2],
    });
  });

  // Synapse pipeline edges
  SYNAPSE_PIPELINE.forEach(([from, to]) => {
    const fromSpikes = spikes[from] ?? 0;
    const v = Math.min(1, fromSpikes / (MAX_SPIKES[from] ?? 100));
    links.push({
      source: `r_${from}`, target: `r_${to}`,
      value: 0.6 + v * 2.5,
      color: `rgba(120,200,180,${(0.2 + v * 0.5).toFixed(2)})`,
      particles: v > 0.05 ? Math.ceil(v * 4) : 0,
      particleSpeed: 0.003 + v * 0.01,
    });
  });

  return { nodes, links };
}

/** Update properties in-place on an existing graph. Returns true if topology changed. */
export function updateMacroGraph(graph: VizGraph, state: MacroState, nodeIndex: Map<string, VizNode>): boolean {
  const fresh = buildMacroGraph(state);

  // Topology change?
  if (fresh.nodes.length !== graph.nodes.length) return true;

  // In-place property update
  for (const fn of fresh.nodes) {
    const existing = nodeIndex.get(fn.id);
    if (!existing) return true; // new node = topology change
    existing.val = fn.val;
    existing.activity = fn.activity;
    existing.color = fn.color;
  }

  for (let i = 0; i < fresh.links.length && i < graph.links.length; i++) {
    const fl = fresh.links[i];
    const el = graph.links[i];
    el.value = fl.value;
    el.color = fl.color;
    el.particles = fl.particles;
    el.particleSpeed = fl.particleSpeed;
  }

  return false;
}
```

**Step 2: Verify it compiles**

Run: `cd ui && npx tsc --noEmit`
Expected: no errors

**Step 3: Commit**

```bash
git add ui/src/components/viz/MacroGraph.ts
git commit -m "feat(viz): add Macro-level graph builder with in-place updates"
```

---

### Task 3: Meso Graph Builder

**Files:**
- Create: `ui/src/components/viz/MesoGraph.ts`

**Step 1: Create the Meso graph builder**

```typescript
// ui/src/components/viz/MesoGraph.ts
import { REGION_DEFS } from "./constants";
import type { VizNode, VizLink, VizGraph } from "./constants";

export type MesoState = {
  region_spikes?: Record<string, number[]>;
  concept_membrane?: number[];
  spike_counts?: Record<string, number>;
};

/**
 * Build a Meso-level graph: individual neurons within a single region.
 * Other regions are shown as faded background nodes.
 */
export function buildMesoGraph(state: MesoState, focusRegion: string): VizGraph {
  const nodes: VizNode[] = [];
  const links: VizLink[] = [];

  const regionDef = REGION_DEFS.find((r) => r.id === focusRegion);
  if (!regionDef) return { nodes, links };

  // Background region nodes (faded)
  REGION_DEFS.forEach((r) => {
    if (r.id === focusRegion) return;
    nodes.push({
      id: `r_${r.id}`, type: "region", regionId: r.id,
      label: r.label, color: r.color + "30",
      val: 6, activity: 0,
      fx: r.target[0], fy: r.target[1], fz: r.target[2],
    });
  });

  // Individual neurons of the focus region
  const numNeurons = regionDef.neurons;
  const spikes = state.region_spikes?.[focusRegion] ?? [];
  const membrane = focusRegion === "concept" ? (state.concept_membrane ?? []) : [];

  const cx = regionDef.target[0];
  const cy = regionDef.target[1];
  const cz = regionDef.target[2];

  for (let i = 0; i < numNeurons; i++) {
    const spiked = spikes[i] === 1;
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

  // Top-K active neurons connect to each other (co-activation)
  const activeIndices = spikes
    .map((v, i) => ({ i, v }))
    .filter((x) => x.v === 1)
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
```

**Step 2: Verify it compiles**

Run: `cd ui && npx tsc --noEmit`
Expected: no errors

**Step 3: Commit**

```bash
git add ui/src/components/viz/MesoGraph.ts
git commit -m "feat(viz): add Meso-level graph builder for individual neurons"
```

---

### Task 4: Node Renderers (Three.js Materials)

**Files:**
- Create: `ui/src/components/viz/nodeRenderers.ts`

**Step 1: Create custom node renderers**

```typescript
// ui/src/components/viz/nodeRenderers.ts
import * as THREE from "three";
import SpriteText from "three-spritetext";
import type { VizNode } from "./constants";

/**
 * Create a Three.js object for a node based on its type.
 * - Regions: SpriteText label + emissive sphere
 * - Sensors: SpriteText label (smaller)
 * - Neurons: default sphere (no custom object needed for performance)
 */
export function nodeThreeObject(node: VizNode): THREE.Object3D | undefined {
  if (node.type === "region") {
    const sprite = new SpriteText(node.label, 5, node.color);
    sprite.fontWeight = "bold";
    sprite.backgroundColor = "rgba(0,0,0,0.6)";
    (sprite as any).padding = [1.5, 4];
    (sprite as any).borderRadius = 4;
    return sprite;
  }
  if (node.type === "sensor") {
    const sprite = new SpriteText(node.label, 3.5, node.color);
    sprite.backgroundColor = "rgba(0,0,0,0.4)";
    (sprite as any).padding = [0.5, 2];
    (sprite as any).borderRadius = 2;
    return sprite;
  }
  // Neurons: return undefined → library renders default sphere
  return undefined as any;
}

/** Color callback: active nodes are bright, inactive are dimmed. */
export function nodeColor(node: VizNode, hoverNodeId: string | null, highlightSet: Set<string>): string {
  if (!hoverNodeId) {
    return node.activity > 0.05 ? node.color : node.color + "40";
  }
  if (highlightSet.has(node.id)) {
    return node.id === hoverNodeId ? "#ffffff" : node.color;
  }
  return node.color + "15";
}

/** Tooltip HTML for node hover. */
export function nodeTooltip(node: VizNode): string {
  const actPct = (node.activity * 100).toFixed(0);
  return `<div style="background:rgba(0,0,0,0.85);padding:8px 12px;border-radius:8px;font-size:13px;color:${node.color};border:1px solid ${node.color}50">
    <b>${node.label}</b>
    ${node.regionId ? `<br/><span style="color:#888;font-size:11px">${node.regionId}</span>` : ""}
    ${node.activity > 0 ? `<br/><span style="color:#6ee7b7;font-size:11px">Aktivitaet: ${actPct}%</span>` : ""}
  </div>`;
}
```

**Step 2: Verify it compiles**

Run: `cd ui && npx tsc --noEmit`
Expected: no errors

**Step 3: Commit**

```bash
git add ui/src/components/viz/nodeRenderers.ts
git commit -m "feat(viz): add custom Three.js node renderers and tooltip"
```

---

### Task 5: WebSocket Subscription Hook

**Files:**
- Modify: `ui/src/lib/ws.ts`
- Create: `ui/src/components/viz/useBrainSubscription.ts`

**Step 1: Extend ws.ts with send capability**

Add to `ui/src/lib/ws.ts` after the existing `getLatest()`:

```typescript
// Add to the end of ws.ts
export function sendWS(msg: Record<string, unknown>): void {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(msg));
  }
}
```

And extend the BrainState type:

```typescript
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
  };
  spike_counts?: Record<string, number>;
  region_spikes?: Record<string, number[]>;
  synapse_activity?: Record<string, { mean_weight: number; active_connections: number[][] }>;
};
```

**Step 2: Create subscription hook**

```typescript
// ui/src/components/viz/useBrainSubscription.ts
import { useEffect, useRef } from "react";
import { sendWS } from "../../lib/ws";
import type { ZoomLevel } from "./constants";

/**
 * Manages WebSocket subscriptions based on current zoom level.
 * Sends subscribe messages when zoom/focus changes.
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
```

**Step 3: Verify it compiles**

Run: `cd ui && npx tsc --noEmit`
Expected: no errors

**Step 4: Commit**

```bash
git add ui/src/lib/ws.ts ui/src/components/viz/useBrainSubscription.ts
git commit -m "feat(viz): add WS send capability and subscription hook"
```

---

### Task 6: Backend — WebSocket Subscription Support

**Files:**
- Modify: `server/ws.py`
- Modify: `server/main.py:84-119` (push_loop function)

**Step 1: Add subscription tracking to WSPusher**

Replace `server/ws.py` entirely:

```python
"""WSPusher — broadcasts brain state to WebSocket clients with subscription support.

Each client can subscribe to different detail levels:
- "macro" (default): compact summary only
- "meso": adds region_spikes for the subscribed region
- "micro": adds synapse_activity for the subscribed neuron
"""
from __future__ import annotations
import asyncio
import json
from typing import Any


class ClientSubscription:
    __slots__ = ("level", "region", "neuron_id")

    def __init__(self) -> None:
        self.level: str = "macro"
        self.region: str | None = None
        self.neuron_id: int | None = None


class WSPusher:
    def __init__(self, rate_hz: float = 30.0) -> None:
        self.rate_hz = rate_hz
        self.clients: dict[Any, ClientSubscription] = {}
        self._lock = asyncio.Lock()

    async def register(self, client: Any) -> None:
        async with self._lock:
            self.clients[client] = ClientSubscription()

    async def unregister(self, client: Any) -> None:
        async with self._lock:
            self.clients.pop(client, None)

    def handle_client_message(self, client: Any, raw: str) -> None:
        """Process subscription messages from clients."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return
        sub = self.clients.get(client)
        if not sub:
            return
        if "subscribe" in msg:
            sub.level = msg["subscribe"]
            sub.region = msg.get("region")
            sub.neuron_id = msg.get("neuron_id")

    def get_subscriptions(self) -> dict[str, set[str | None]]:
        """Return which regions need detail data."""
        meso_regions: set[str | None] = set()
        micro_regions: set[str | None] = set()
        for sub in self.clients.values():
            if sub.level == "meso" and sub.region:
                meso_regions.add(sub.region)
            elif sub.level == "micro" and sub.region:
                micro_regions.add(sub.region)
        return {"meso": meso_regions, "micro": micro_regions}

    async def broadcast(self, base_state: dict[str, Any], detail_state: dict[str, Any] | None = None) -> None:
        """Send state to all clients. Macro clients get base_state only.
        Meso/micro clients get base_state + relevant detail fields."""
        base_text = json.dumps(base_state, default=_json_default)

        # Pre-build detail payload if needed
        detail_text: str | None = None
        if detail_state:
            merged = {**base_state, **detail_state}
            detail_text = json.dumps(merged, default=_json_default)

        async with self._lock:
            failed = []
            for client, sub in list(self.clients.items()):
                try:
                    if sub.level in ("meso", "micro") and detail_text:
                        await client.send_text(detail_text)
                    else:
                        await client.send_text(base_text)
                except Exception:
                    failed.append(client)
            for c in failed:
                self.clients.pop(c, None)


def _json_default(o: Any) -> Any:
    """Fallback JSON serializer for tensors etc."""
    try:
        import torch
        if isinstance(o, torch.Tensor):
            return o.tolist()
    except ImportError:
        pass
    if hasattr(o, "__iter__"):
        return list(o)
    return str(o)
```

**Step 2: Update the WebSocket endpoint in server/main.py to handle client messages**

In `server/main.py`, replace the `/ws` endpoint (lines 35-49):

```python
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        if pusher is not None:
            await pusher.register(ws)
        try:
            while True:
                raw = await ws.receive_text()
                # Forward subscription messages to pusher
                if pusher is not None:
                    pusher.handle_client_message(ws, raw)
        except WebSocketDisconnect:
            pass
        finally:
            if pusher is not None:
                await pusher.unregister(ws)
```

**Step 3: Update push_loop to include detail data**

In `server/main.py`, update `push_loop` (lines 84-119) to gather and send detail data:

```python
async def push_loop(brain: Any, pusher: WSPusher, exporter: Any = None, adapter: Any = None) -> None:
    """Periodically broadcast brain state to all WS clients."""
    period = 1.0 / pusher.rate_hz
    try:
        while True:
            # Sensor snapshot for dashboard display
            sensor_snap = adapter.bus.snapshot() if adapter else {}
            sensor_display = {}
            if "active_app" in sensor_snap:
                app_data = sensor_snap["active_app"]
                sensor_display["app"] = app_data.get("name", "?")
                sensor_display["background_apps"] = app_data.get("background_apps", [])
                sensor_display["app_count"] = app_data.get("app_count", 1)
                sensor_display["app_switched"] = app_data.get("switched", False)
            if "keystroke_rate" in sensor_snap:
                sensor_display["keys"] = sensor_snap["keystroke_rate"].get("count", 0)
            if "mouse_rate" in sensor_snap:
                sensor_display["mouse"] = sensor_snap["mouse_rate"].get("count", 0)
            if "idle" in sensor_snap:
                sensor_display["idle"] = round(sensor_snap["idle"].get("seconds", 0), 1)
            if "mic" in sensor_snap:
                sensor_display["mic_rms"] = round(sensor_snap["mic"].get("rms", 0), 4)

            base_state = {
                "tick": brain.tick_count,
                "modulators": brain.modulators.snapshot(),
                "concept_membrane": brain.concept_spike_accum.tolist(),
                "wm_membrane": brain.regions["wm"].membrane.tolist(),
                "sensors": sensor_display,
                "spike_counts": {
                    "sensory": int(brain._last_sensory_spikes) if hasattr(brain, '_last_sensory_spikes') else 0,
                    "feature": int(brain._last_feature_spikes) if hasattr(brain, '_last_feature_spikes') else 0,
                    "association": int(brain._last_association_spikes) if hasattr(brain, '_last_association_spikes') else 0,
                    "concept": int(brain._last_concept_spikes) if hasattr(brain, '_last_concept_spikes') else 0,
                },
            }

            # Gather detail data if any client subscribed to meso/micro
            detail_state = None
            subs = pusher.get_subscriptions()
            needed_regions = subs["meso"] | subs["micro"]

            if needed_regions:
                detail_state = {}
                region_spikes = {}
                for region_name in needed_regions:
                    region = brain.regions.get(region_name)
                    if region and hasattr(region, 'membrane'):
                        # Last spike state: 1 where membrane was reset (just spiked)
                        spikes = (region.membrane == 0).float().tolist()
                        region_spikes[region_name] = spikes
                detail_state["region_spikes"] = region_spikes

            await pusher.broadcast(base_state, detail_state)
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        return
```

**Step 4: Run backend tests**

Run: `bash scripts/run_tests.sh`
Expected: All batches PASSED

**Step 5: Commit**

```bash
git add server/ws.py server/main.py
git commit -m "feat(backend): add WebSocket subscription support for meso/micro detail data"
```

---

### Task 7: Main BrainViz3D Component

**Files:**
- Create: `ui/src/components/BrainViz3D.tsx`

**Step 1: Create the main visualization component**

```tsx
// ui/src/components/BrainViz3D.tsx
import { useRef, useEffect, useState, useMemo, useCallback, lazy, Suspense } from "react";
import type { BrainState } from "../lib/ws";
import { buildMacroGraph, updateMacroGraph } from "./viz/MacroGraph";
import { buildMesoGraph } from "./viz/MesoGraph";
import { useBrainSubscription } from "./viz/useBrainSubscription";
import { nodeThreeObject, nodeColor, nodeTooltip } from "./viz/nodeRenderers";
import { REGION_DEFS } from "./viz/constants";
import type { ZoomLevel, VizNode, VizLink, VizGraph } from "./viz/constants";

const ForceGraph3D = lazy(() => import("react-force-graph-3d"));

export function BrainViz3D({ state }: { state: BrainState | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [dims, setDims] = useState({ w: 800, h: 500 });

  // Zoom state
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>("macro");
  const [focusRegion, setFocusRegion] = useState<string | null>(null);
  const [focusNeuron, setFocusNeuron] = useState<number | null>(null);

  // Hover state
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null);
  const highlightSet = useRef(new Set<string>());

  // Graph data
  const graphRef = useRef<VizGraph>({ nodes: [], links: [] });
  const nodeIndexRef = useRef(new Map<string, VizNode>());
  const [graphVersion, setGraphVersion] = useState(0);

  // Subscribe to detail data based on zoom level
  useBrainSubscription(zoomLevel, focusRegion, focusNeuron);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([e]) =>
      setDims({ w: e.contentRect.width, h: e.contentRect.height })
    );
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Camera + auto-orbit setup
  const initDone = useRef(false);
  useEffect(() => {
    if (initDone.current) return;
    const tryInit = () => {
      const fg = fgRef.current;
      if (!fg) return setTimeout(tryInit, 200);
      initDone.current = true;
      fg.cameraPosition({ x: -60, y: 90, z: 350 }, { x: 0, y: 0, z: 0 }, 0);
      const controls = fg.controls();
      if (controls && "autoRotate" in controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.4;
      }
    };
    tryInit();
  }, []);

  // Build/update graph from state
  useEffect(() => {
    if (!state) return;

    if (zoomLevel === "macro") {
      const needsRebuild = updateMacroGraph(graphRef.current, state as any, nodeIndexRef.current);
      if (needsRebuild || graphRef.current.nodes.length === 0) {
        const fresh = buildMacroGraph(state as any);
        graphRef.current = fresh;
        nodeIndexRef.current = new Map(fresh.nodes.map((n) => [n.id, n]));
        setGraphVersion((v) => v + 1);
      }
    } else if (zoomLevel === "meso" && focusRegion) {
      const fresh = buildMesoGraph(state as any, focusRegion);
      graphRef.current = fresh;
      nodeIndexRef.current = new Map(fresh.nodes.map((n) => [n.id, n]));
      setGraphVersion((v) => v + 1);
    }
  }, [state, zoomLevel, focusRegion]);

  // Handle node click — zoom transitions
  const handleNodeClick = useCallback((node: any) => {
    if (!node) return;

    if (zoomLevel === "macro" && node.type === "region" && node.regionId) {
      // Macro → Meso: zoom into region
      setFocusRegion(node.regionId);
      setZoomLevel("meso");

      // Fly camera to region
      const fg = fgRef.current;
      if (fg && node.fx != null) {
        fg.cameraPosition(
          { x: node.fx - 30, y: node.fy + 20, z: node.fz + 80 },
          { x: node.fx, y: node.fy, z: node.fz },
          800
        );
      }
    } else if (zoomLevel === "macro" && node.type === "sensor") {
      // Clicking sensor in macro: no action (yet)
    } else if (zoomLevel === "meso" && node.type === "neuron") {
      // Meso: label concept neurons
      if (node.regionId === "concept") {
        const idx = parseInt(node.id.split("_").pop() ?? "0");
        const label = prompt(`Label fuer Neuron #${idx}:`);
        if (label) {
          fetch("/api/label", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_id: idx, label }),
          });
        }
      }
    }
  }, [zoomLevel]);

  // Handle Escape — zoom out
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (zoomLevel === "micro") {
          setFocusNeuron(null);
          setZoomLevel("meso");
        } else if (zoomLevel === "meso") {
          setFocusRegion(null);
          setZoomLevel("macro");
          // Reset camera
          const fg = fgRef.current;
          if (fg) {
            fg.cameraPosition({ x: -60, y: 90, z: 350 }, { x: 0, y: 0, z: 0 }, 800);
          }
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [zoomLevel]);

  // Hover
  const handleNodeHover = useCallback((node: any) => {
    highlightSet.current.clear();
    if (node) {
      highlightSet.current.add(node.id);
    }
    setHoverNodeId(node?.id ?? null);
  }, []);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Waiting for brain state...
      </div>
    );
  }

  return (
    <div ref={containerRef} className="h-full w-full rounded-lg overflow-hidden relative" style={{ background: "#000003" }}>
      <Suspense fallback={<div className="flex items-center justify-center h-full text-gray-500">Loading 3D brain...</div>}>
        <ForceGraph3D
          ref={fgRef}
          width={dims.w}
          height={dims.h}
          graphData={graphRef.current}
          backgroundColor="#000003"
          showNavInfo={false}
          nodeRelSize={4}
          nodeVal={(n: any) => n.val}
          nodeColor={(n: any) => nodeColor(n, hoverNodeId, highlightSet.current)}
          nodeOpacity={0.85}
          nodeResolution={8}
          nodeThreeObject={(n: any) => nodeThreeObject(n)}
          nodeThreeObjectExtend={true}
          nodeLabel={(n: any) => nodeTooltip(n)}
          linkColor={(l: any) => l.color}
          linkWidth={(l: any) => Math.max(0.2, l.value)}
          linkOpacity={0.4}
          linkCurvature={0.2}
          linkDirectionalParticles={(l: any) => l.particles}
          linkDirectionalParticleSpeed={(l: any) => l.particleSpeed}
          linkDirectionalParticleWidth={2.5}
          linkDirectionalParticleColor={() => "#34d399"}
          cooldownTicks={100}
          d3AlphaDecay={0.06}
          d3VelocityDecay={0.4}
          warmupTicks={30}
          enableNodeDrag={false}
          onNodeHover={handleNodeHover}
          onNodeClick={handleNodeClick}
        />
      </Suspense>

      {/* Zoom level indicator + back button */}
      <div className="absolute top-3 left-3 flex items-center gap-2">
        {zoomLevel !== "macro" && (
          <button
            onClick={() => {
              if (zoomLevel === "micro") {
                setFocusNeuron(null);
                setZoomLevel("meso");
              } else {
                setFocusRegion(null);
                setZoomLevel("macro");
                fgRef.current?.cameraPosition({ x: -60, y: 90, z: 350 }, { x: 0, y: 0, z: 0 }, 800);
              }
            }}
            className="bg-gray-800/80 hover:bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded backdrop-blur-sm"
          >
            ← Zurueck
          </button>
        )}
        <span className="bg-black/60 text-gray-400 text-[10px] px-2 py-0.5 rounded font-mono backdrop-blur-sm">
          {zoomLevel === "macro" ? "Uebersicht" : zoomLevel === "meso" ? `Region: ${focusRegion}` : `Neuron #${focusNeuron}`}
        </span>
      </div>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 bg-black/70 backdrop-blur-sm rounded-lg p-2.5 text-[10px] font-mono space-y-1 pointer-events-none">
        {REGION_DEFS.map((r) => (
          <div key={r.id} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: r.color, boxShadow: `0 0 6px ${r.color}` }} />
            <span className="font-bold" style={{ color: r.color }}>{r.label}</span>
          </div>
        ))}
      </div>

      {/* Keyboard hint */}
      {zoomLevel !== "macro" && (
        <div className="absolute bottom-3 right-3 text-[10px] text-gray-600 font-mono">
          ESC = zurueck
        </div>
      )}
    </div>
  );
}
```

**Step 2: Verify it compiles**

Run: `cd ui && npx tsc --noEmit`
Expected: no errors (or minor type issues to fix)

**Step 3: Commit**

```bash
git add ui/src/components/BrainViz3D.tsx
git commit -m "feat(viz): add BrainViz3D with Macro/Meso zoom levels"
```

---

### Task 8: Swap In App.tsx

**Files:**
- Modify: `ui/src/App.tsx`
- Rename: `ui/src/components/BrainPanel3D.tsx` → `ui/src/components/BrainPanel3D.legacy.tsx`

**Step 1: Rename old component as fallback**

```bash
cp ui/src/components/BrainPanel3D.tsx ui/src/components/BrainPanel3D.legacy.tsx
```

**Step 2: Update App.tsx import**

Replace in `ui/src/App.tsx`:

```typescript
import { BrainPanel3D } from "./components/BrainPanel3D";
```

with:

```typescript
import { BrainViz3D } from "./components/BrainViz3D";
```

And replace in the JSX:

```tsx
<BrainPanel3D state={state} />
```

with:

```tsx
<BrainViz3D state={state} />
```

**Step 3: Verify the UI builds**

Run: `cd ui && npx tsc --noEmit && npm run build`
Expected: Build succeeds

**Step 4: Commit**

```bash
git add ui/src/App.tsx ui/src/components/BrainPanel3D.legacy.tsx
git commit -m "feat(viz): swap BrainViz3D into main layout, keep legacy as fallback"
```

---

### Task 9: Bloom Post-Processing

**Files:**
- Modify: `ui/src/components/BrainViz3D.tsx` (init effect)

**Step 1: Add bloom in the camera init effect**

Add to the `tryInit` function inside the `useEffect` in BrainViz3D.tsx, after the auto-orbit setup:

```typescript
      // Bloom — only glows on emissive nodes (threshold 0.85)
      try {
        const { UnrealBloomPass } = await import("three/examples/jsm/postprocessing/UnrealBloomPass.js");
        const THREE = await import("three");
        const bloomPass = new UnrealBloomPass(
          new THREE.Vector2(dims.w, dims.h),
          0.8,   // strength
          0.4,   // radius
          0.85   // threshold — only bright emissive objects glow
        );
        fg.postProcessingComposer().addPass(bloomPass);
      } catch (e) {
        console.warn("Bloom not available:", e);
      }
```

Note: This requires `three` as a dependency (already present via react-force-graph-3d). The import is dynamic to avoid blocking initial render.

**Step 2: Verify it compiles and renders**

Run: `cd ui && npm run dev`
Open browser, verify bloom effect on active regions.

**Step 3: Commit**

```bash
git add ui/src/components/BrainViz3D.tsx
git commit -m "feat(viz): add UnrealBloomPass for emissive glow on active nodes"
```

---

### Task 10: Visual Verification & Polish

**Files:**
- Potentially: minor tweaks to any `viz/*.ts` files

**Step 1: Start full stack**

```bash
# Terminal 1: daemon
.venv/bin/python -m server.braind start --port 8765

# Terminal 2: UI dev server
cd ui && npm run dev
```

**Step 2: Verify Macro level**

Open `http://localhost:5174` in browser. Check:
- [ ] 7 region spheres visible with correct colors
- [ ] 6 sensor nodes in arc to the left
- [ ] Inter-region edges with directional particles
- [ ] Regions positioned along information flow (sensory bottom-left → motor top-right)
- [ ] Auto-orbit camera
- [ ] Hover highlighting works
- [ ] Legend in bottom-left corner

**Step 3: Verify Meso level**

Click on a region (e.g., "Konzeptbildung"). Check:
- [ ] Camera flies to region
- [ ] Other regions fade to transparent
- [ ] Individual neurons visible as small spheres
- [ ] Active neurons flash brighter
- [ ] "Zurueck" button visible in top-left
- [ ] ESC returns to Macro

**Step 4: Fix any issues found**

Address visual issues: adjust sizes, colors, camera positions as needed.

**Step 5: Commit**

```bash
git add -A
git commit -m "fix(viz): polish visual details after verification"
```

---

### Task 11: Run Full Test Suite

**Step 1: Run backend tests**

```bash
bash scripts/run_tests.sh
```
Expected: All batches PASSED

**Step 2: Run frontend build**

```bash
cd ui && npm run build
```
Expected: Build succeeds with no errors

**Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete 3D brain visualization overhaul with Progressive Disclosure"
```
