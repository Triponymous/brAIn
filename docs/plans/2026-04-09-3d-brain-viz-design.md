# 3D Brain Visualization Overhaul — Design Document

**Date:** 2026-04-09
**Status:** Approved
**Depends on:** Mini-OSCEN Design Doc (2026-04-08)

---

## Goal

Replace the current `BrainPanel3D.tsx` with a Progressive Disclosure visualization that renders the full SNN as an interactive 3D force graph. Three zoom levels — Macro (region blobs), Meso (individual neurons), Micro (single neuron + synapses) — let the user explore from overview to detail without sacrificing performance.

## Library

**react-force-graph-3d** (already installed, v1.29.1). No library change needed. The current implementation underuses the library — this redesign exploits its full API: directional particles, custom node objects, emissive materials, in-place graph mutation.

## Architecture

### Zoom Levels

| Level | What's visible | Node count | When |
|---|---|---|---|
| **Macro** | 7 region spheres + 6 sensor nodes + inter-region edges | ~15 | Default view |
| **Meso** | Individual neurons of one region + intra-region connections | 50-500 | Click a region |
| **Micro** | One neuron + all pre/post synaptic partners + weights | 10-50 | Click a neuron in Meso |

Transitions: click region (Macro -> Meso), click neuron (Meso -> Micro), click background or press Escape (back one level).

### Region Layout (Semi-Fixed)

Regions are positioned along the information flow direction with soft constraints. Each region has a **target position** (`fx`, `fy`, `fz`) that the force layout nudges toward via a custom cluster force. The layout may shift slightly for aesthetics but preserves the mental map across reloads.

Information flow: left-to-right, bottom-to-top.

```
Target positions (x, y, z):

                    Meta (0, 150, 0)
                        |
    Motor (120, 100, 0) |
         \              |
    Concept (60, 60, 0) --- WM (60, 0, -40)
         |
    Association (0, 0, 0)
         |
    Feature (-60, -60, 0)
         |
    Sensory (-120, -120, 0)
```

Sensors form an arc to the left of Sensory: `(-180, -120+i*30, 0)` for i in 0..5.

Implementation: a custom d3 force (`forceCluster`) that pulls each node toward its target with strength 0.3. Combined with charge repulsion (strength -100) and link force, this creates stable organic-looking layouts that respect the flow direction.

### Data Flow

```
WebSocket 30Hz
    |
    v
BrainState (existing format)
    |
    v
useGraphBuilder(state, zoomLevel, focusRegion, focusNeuron)
    |
    +-- Macro: buildMacroGraph(state)
    +-- Meso:  buildMesoGraph(state, region)
    +-- Micro: buildMicroGraph(state, neuronId)
    |
    v
graphRef (mutated in-place for property updates)
    |
    v
react-force-graph-3d
    |
    v
Three.js WebGL Canvas
```

**Critical rule:** `graphData()` is called ONLY when zoom level or focus changes (topology change). On every WebSocket tick, node properties (color, size, emissive intensity) are mutated in-place on the existing graph object. The force graph's render loop picks up property changes automatically.

### Backend Extension

The existing WebSocket push format is extended with optional detail fields. The client sends subscription messages to request detail data for specific regions/neurons.

**Client -> Server (new):**
```json
{"subscribe": "macro"}
{"subscribe": "meso", "region": "concept"}
{"subscribe": "micro", "neuron_id": 42, "region": "concept"}
```

**Server -> Client (extended):**
```json
{
  "tick": 12345,
  "modulators": {"DA": 0.03, "NE": 0.01, "ACh": 0.02, "5HT": 0.01},
  "spike_counts": {"sensory": 12, "feature": 8, "association": 3, "concept": 3},
  "sensors": {"app": "Claude", "keys": 5, "mouse": 12, "idle": 0.3, "mic_rms": 0.02,
              "background_apps": ["Terminal", "Spotify"], "app_count": 3},
  "concept_membrane": [0.0, 0.1, ...],
  "wm_membrane": [0.0, ...],

  "region_spikes": {
    "concept": [0, 0, 1, 0, 0, 1, ...]
  },
  "synapse_activity": {
    "association_concept": {
      "mean_weight": 0.28,
      "active_connections": [[12, 45, 0.89], [3, 22, 0.82]]
    }
  }
}
```

`region_spikes` and `synapse_activity` are only included when the client has subscribed to meso/micro for that region. Default (macro) sends only the existing compact format.

## Visual Design

### Macro Level

- **Region nodes:** Spheres with `MeshStandardMaterial`, size = `12 + activity * 10`, color per region (see below).
- **Region colors:** Sensory=#6ee7b7 (cyan-green), Feature=#34d399 (emerald), Association=#a78bfa (purple), Concept=#fbbf24 (gold), WM=#60a5fa (blue), Motor=#f87171 (red), Meta=#9ca3af (gray).
- **Emissive glow:** Active regions get `emissiveIntensity = activity * 0.8`. Bloom post-processing with threshold=0.85 makes only active regions glow.
- **Inter-region edges:** Width proportional to aggregate spike flow. Directional particles (count = spike_rate / 5, speed = 0.008) represent spike traffic between regions.
- **Sensor nodes:** Smaller spheres (size=6) in arc left of Sensory. Active sensors are bright, inactive are dim. Edges from active sensors to Sensory region.
- **Labels:** SpriteText labels below each region node (German: Sensorik, Mustererkennung, Verkn., Konzeptbildung, Ged., Motorik, Meta).

### Meso Level (click region)

- Camera flies to the clicked region (using `cameraPosition` + lookAt transition).
- Other regions become semi-transparent (opacity=0.15).
- Region blob is replaced by individual neuron nodes: small spheres (size=2-4), color = region color with brightness proportional to membrane potential.
- Active neurons (just spiked) flash white briefly.
- Top-K strongest intra-region weight connections shown as thin edges.
- Concept neurons show their label (if assigned) on hover.
- A "back" affordance (translucent shell around the region, or Escape key) returns to Macro.

### Micro Level (click neuron in Meso)

- Focus on one neuron, centered.
- All pre-synaptic and post-synaptic partners shown as connected nodes.
- Edge width = weight strength. Edge color = hot (red/orange) for strong, cold (blue) for weak.
- Directional particles show recent spike propagation direction.
- The focused neuron shows a pulsing ring proportional to its membrane potential.
- Escape or click background returns to Meso.

## Performance Strategy

1. **Node resolution:** `nodeResolution=8` (not 16). Halves triangle count per sphere.
2. **In-place mutation:** Never call `graphData()` for property-only updates. Mutate `node.val`, `node.color`, `link.value` directly.
3. **Map index:** `Map<string, GraphNode>` for O(1) lookup instead of `Array.find()` O(n).
4. **Physics freeze:** `cooldownTicks=100`, then pin all node positions via `fx/fy/fz`. Re-heat only on topology change (zoom level transition).
5. **Conditional data push:** Backend only sends detail data (region_spikes, synapse_activity) when client subscribes. Macro view = minimal payload.
6. **Bloom:** `UnrealBloomPass` with `strength=0.8, radius=0.4, threshold=0.85`. Only emissive nodes glow. Disabled if FPS drops below 30.
7. **Pointer interaction:** `enablePointerInteraction=true` only when mouse is over the graph (disable during idle to save GPU raycasts).

## Component Structure

```
ui/src/components/
  BrainViz3D.tsx          -- main component, manages zoom state
  viz/
    useGraphBuilder.ts    -- builds graph data for each zoom level
    useBrainSubscription.ts -- WebSocket subscription management
    MacroGraph.ts         -- Macro level graph builder
    MesoGraph.ts          -- Meso level graph builder
    MicroGraph.ts         -- Micro level graph builder
    nodeRenderers.ts      -- custom Three.js node materials
    constants.ts          -- colors, positions, sizes
```

## What Is NOT Built (YAGNI)

- No time scrubber (future phase)
- No spike raster panel (separate component, future)
- No InstancedMesh optimization (only needed at >5K neurons)
- No WebWorker for force simulation (only needed at >5K neurons)
- No VR/AR mode
- No screenshot/export feature

## Testing

- Unit tests for graph builders (MacroGraph, MesoGraph, MicroGraph) with mock BrainState
- Visual regression: manual verification at each zoom level
- Performance: measure FPS at Macro (target: 60fps) and Meso with 500 neurons (target: 30fps)
- WebSocket subscription: verify backend only sends detail data when subscribed

## Migration

1. Keep `BrainPanel3D.tsx` as `BrainPanel3D.legacy.tsx` (fallback)
2. Build `BrainViz3D.tsx` as replacement
3. Swap in `App.tsx` once verified
4. Delete legacy after 1 week of stable operation
