import { useRef, useEffect, useState, useMemo, useCallback, lazy, Suspense } from "react";
import type { BrainState } from "../lib/ws";
// No bloom – causes white-out at close zoom levels
import SpriteText from "three-spritetext";

const ForceGraph3D = lazy(() => import("react-force-graph-3d"));

/*
 * 3D Brain – Knowledge Graph with real brain anatomy layout.
 *
 * Positions follow a sagittal (side-view) human brain:
 *
 *              Motor Cortex (top-center)
 *             /                          \
 *   Prefrontal (front-top)          Parietal (top-back)
 *        |                                |
 *   Temporal (side-bottom)          Occipital (back)
 *        |                           /
 *   Hippocampus (deep-center)
 *        |
 *   Brainstem (bottom)
 *
 * z+ = anterior (front), z- = posterior (back)
 * y+ = superior (top),   y- = inferior (bottom)
 * x  = lateral (left/right)
 */

type ExtendedState = BrainState & {
  sensors?: { app?: string; keys?: number; mouse?: number; idle?: number; mic_rms?: number };
  spike_counts?: { sensory?: number; feature?: number; concept?: number };
};

/*
 * Brain layout (sagittal view, spread out for clarity):
 *
 *        Motorik (top-front)        Sensorik (top-back)
 *               \                      /
 *    Konzeptbildung (front)    Verknuepfung (back)
 *               \                   /
 *          Mustererkennung (side-low)
 *                    |
 *            Gedaechtnis (deep center)
 *
 * Scale factor ~4x for good spacing in 3D view
 */
const REGIONS = [
  { id: "r_sensory", label: "Sensorik", desc: "Sinneseindruecke empfangen",
    color: "#6ee7b7", pos: [0, 120, -50] as const, dataKey: "sensory" as const, maxSpikes: 200 },

  { id: "r_feature", label: "Mustererkennung", desc: "Muster & Features erkennen",
    color: "#67e8f9", pos: [-30, -10, 30] as const, dataKey: "feature" as const, maxSpikes: 300 },

  { id: "r_association", label: "Verknuepfung", desc: "Informationen verbinden",
    color: "#c084fc", pos: [10, 90, -100] as const, dataKey: "feature" as const, maxSpikes: 300 },

  { id: "r_concept", label: "Konzeptbildung", desc: "Abstrakte Konzepte bilden",
    color: "#34d399", pos: [0, 80, 100] as const, dataKey: "concept" as const, maxSpikes: 30 },

  { id: "r_wm", label: "Gedaechtnis", desc: "Aktive Erinnerungen halten",
    color: "#a78bfa", pos: [0, -30, 0] as const, dataKey: null, maxSpikes: 1 },

  { id: "r_motor", label: "Motorik", desc: "Aktionen & Output steuern",
    color: "#fbbf24", pos: [0, 130, 20] as const, dataKey: null, maxSpikes: 1 },
];

// Sensors – arc well outside the brain to the left
const SENSORS = [
  { id: "s_app",   label: "App",      color: "#60a5fa", pos: [-160, 130, -70] as const },
  { id: "s_keys",  label: "Tastatur", color: "#34d399", pos: [-170, 100, -55] as const },
  { id: "s_mouse", label: "Maus",     color: "#a78bfa", pos: [-170,  70, -40] as const },
  { id: "s_idle",  label: "Idle",     color: "#fbbf24", pos: [-160,  40, -25] as const },
  { id: "s_mic",   label: "Mikrofon", color: "#f87171", pos: [-150,  10, -10] as const },
  { id: "s_time",  label: "Zeit",     color: "#818cf8", pos: [-140, -20,   5] as const },
];

// Neural pathway: sensory input → processing → output
const PIPELINE: [string, string][] = [
  ["r_sensory", "r_feature"],      // sensory → temporal (pattern recognition)
  ["r_feature", "r_association"],   // temporal → parietal (integration)
  ["r_association", "r_concept"],   // parietal → prefrontal (conceptualization)
  ["r_concept", "r_wm"],           // prefrontal → hippocampus (memory)
  ["r_concept", "r_motor"],        // prefrontal → motor (action)
  ["r_wm", "r_association"],       // hippocampus → parietal (recall feeds back)
  ["r_sensory", "r_motor"],        // sensory → motor (reflexes, fast path)
];

type GNode = { id: string; type: "sensor"|"region"|"concept"; label: string; desc: string; val: number; color: string; active: boolean; activity: number; fx: number; fy: number; fz: number; neighbors?: GNode[]; links?: GLink[] };
type GLink = { source: string; target: string; value: number; color: string; particles: number; particleSpeed: number };

function buildGraph(state: ExtendedState): { nodes: GNode[]; links: GLink[] } {
  const nodes: GNode[] = [];
  const links: GLink[] = [];
  const sensors = state.sensors ?? {};
  const spikes = state.spike_counts ?? {};
  const conceptMem = state.concept_membrane ?? [];
  const maxC = Math.max(0.01, ...conceptMem.map(Math.abs));

  const sensorActive: Record<string, boolean> = {
    s_app: !!(sensors.app), s_keys: (sensors.keys ?? 0) > 0, s_mouse: (sensors.mouse ?? 0) > 0,
    s_idle: (sensors.idle ?? 0) < 5, s_mic: (sensors.mic_rms ?? 0) > 0.005, s_time: true,
  };

  // Sensor nodes – positioned outside the brain, curved arc
  SENSORS.forEach(s => {
    const active = sensorActive[s.id] ?? false;
    nodes.push({ id: s.id, type: "sensor", label: s.label, desc: `Sensor: ${s.label}`, color: s.color,
      val: active ? 5 : 2, active, activity: active ? 0.8 : 0,
      fx: s.pos[0], fy: s.pos[1], fz: s.pos[2] });
    if (active) {
      links.push({ source: s.id, target: "r_sensory", value: 0.4, color: s.color, particles: 2, particleSpeed: 0.008 });
    }
  });

  // Region nodes – anatomical brain positions
  REGIONS.forEach(r => {
    const raw = r.dataKey ? (spikes[r.dataKey] ?? 0) : 0;
    const activity = r.dataKey ? Math.min(1, raw / r.maxSpikes) : 0.15;
    nodes.push({ id: r.id, type: "region", label: r.label, desc: r.desc, color: r.color,
      val: 12 + activity * 10, active: activity > 0.05, activity,
      fx: r.pos[0], fy: r.pos[1], fz: r.pos[2] });
  });

  // Neural pipeline connections
  PIPELINE.forEach(([from, to]) => {
    const reg = REGIONS.find(r => r.id === from);
    const count = reg?.dataKey ? (spikes[reg.dataKey] ?? 0) : 0;
    const v = Math.min(1, count / 150);
    links.push({ source: from, target: to,
      value: 0.6 + v * 2.5,
      color: `rgba(120,200,180,${0.2 + v * 0.5})`,
      particles: v > 0.05 ? Math.ceil(v * 4) : 0,
      particleSpeed: 0.003 + v * 0.01 });
  });

  // Concept neurons – cluster around prefrontal cortex
  const indexed = conceptMem.map((v, i) => ({ i, v: Math.abs(v) / maxC }));
  indexed.sort((a, b) => b.v - a.v);
  const activeConcepts = indexed.filter(c => c.v > 0.05).slice(0, 20);
  const cpos = REGIONS.find(r => r.id === "r_concept")!.pos;

  activeConcepts.forEach((c, idx) => {
    const phi = (idx / Math.max(1, activeConcepts.length)) * Math.PI * 2;
    const layer = Math.floor(idx / 6);
    const rad = 30 + layer * 15;
    const ySpread = 15;
    nodes.push({ id: `c${c.i}`, type: "concept", label: `#${c.i}`, desc: `Konzept #${c.i}`,
      color: "#34d399", val: 1 + c.v * 5, active: c.v > 0.1, activity: c.v,
      fx: cpos[0] + Math.cos(phi) * rad,
      fy: cpos[1] + (layer - 1) * ySpread,
      fz: cpos[2] + Math.sin(phi) * rad });
    links.push({ source: "r_concept", target: `c${c.i}`,
      value: c.v, color: `rgba(52,211,153,${c.v * 0.35})`,
      particles: c.v > 0.3 ? 1 : 0, particleSpeed: 0.005 });
  });

  // Co-activation links between top concepts
  const top = activeConcepts.slice(0, 6);
  for (let i = 0; i < top.length; i++)
    for (let j = i + 1; j < top.length; j++) {
      const s = (top[i].v + top[j].v) / 2;
      if (s > 0.3)
        links.push({ source: `c${top[i].i}`, target: `c${top[j].i}`,
          value: s * 0.4, color: `rgba(52,211,153,${s * 0.15})`, particles: 0, particleSpeed: 0 });
    }

  // Build neighbor index for hover highlighting
  const byId = new Map(nodes.map(n => [n.id, n]));
  nodes.forEach(n => { n.neighbors = []; n.links = []; });
  links.forEach(l => {
    const s = byId.get(l.source as string), t = byId.get(l.target as string);
    if (s && t) { s.neighbors!.push(t); t.neighbors!.push(s); s.links!.push(l); t.links!.push(l); }
  });

  return { nodes, links };
}

export function BrainPanel3D({ state }: { state: BrainState | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [dims, setDims] = useState({ w: 800, h: 500 });
  const [hoverNode, setHoverNode] = useState<GNode | null>(null);
  const highlightNodes = useRef(new Set<string>());
  const highlightLinks = useRef(new Set<GLink>());
  const initDone = useRef(false);

  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([e]) => setDims({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Setup bloom, camera, auto-orbit. Retry until fgRef is ready.
  useEffect(() => {
    if (initDone.current) return;
    const tryInit = () => {
      const fg = fgRef.current;
      if (!fg) return setTimeout(tryInit, 200);
      initDone.current = true;

      // Camera – centered on brain network
      fg.cameraPosition(
        { x: -60, y: 90, z: 350 },
        { x: -40, y: 45, z: 0 },
        0
      );

      // Auto-orbit
      const controls = fg.controls();
      if (controls && 'autoRotate' in controls) {
        (controls as any).autoRotate = true;
        (controls as any).autoRotateSpeed = 0.4;
      }
    };
    tryInit();
  }, []);

  // Graph data with in-place update to prevent flicker
  const graphRef = useRef<{ nodes: GNode[]; links: GLink[] }>({ nodes: [], links: [] });
  const lastBuildRef = useRef(0);

  const graphData = useMemo(() => {
    if (!state) return { nodes: [], links: [] };
    const now = Date.now();
    const fresh = buildGraph(state as ExtendedState);
    const prev = graphRef.current;

    if (prev.nodes.length !== fresh.nodes.length || now - lastBuildRef.current > 2000) {
      graphRef.current = fresh;
      lastBuildRef.current = now;
      return fresh;
    }
    for (const nn of fresh.nodes) {
      const ex = prev.nodes.find(n => n.id === nn.id);
      if (ex) { ex.val = nn.val; ex.active = nn.active; ex.activity = nn.activity; ex.color = nn.color; ex.label = nn.label; }
    }
    for (let i = 0; i < fresh.links.length && i < prev.links.length; i++) {
      const fl = fresh.links[i], pl = prev.links[i];
      pl.value = fl.value; pl.color = fl.color; pl.particles = fl.particles; pl.particleSpeed = fl.particleSpeed;
    }
    return prev;
  }, [state]);

  // Hover highlighting
  const handleNodeHover = useCallback((node: any) => {
    highlightNodes.current.clear();
    highlightLinks.current.clear();
    if (node) {
      highlightNodes.current.add(node.id);
      (node.neighbors ?? []).forEach((n: GNode) => highlightNodes.current.add(n.id));
      (node.links ?? []).forEach((l: GLink) => highlightLinks.current.add(l));
    }
    setHoverNode(node || null);
  }, []);

  // SpriteText labels for regions and sensors
  const nodeThreeObject = useCallback((node: any) => {
    if (node.type === "region") {
      const s = new SpriteText(node.label, 4.5, node.color);
      s.fontWeight = "bold";
      s.backgroundColor = "rgba(0,0,0,0.5)";
      s.padding = [1, 3] as any;
      s.borderRadius = 3;
      return s;
    }
    if (node.type === "sensor") {
      const s = new SpriteText(node.label, 3, node.color);
      s.backgroundColor = "rgba(0,0,0,0.4)";
      s.padding = [0.5, 2] as any;
      s.borderRadius = 2;
      return s;
    }
    return undefined as any; // default sphere for concepts
  }, []);

  const nodeColor = useCallback((node: any) => {
    if (!hoverNode) return node.active ? node.color : node.color + "50";
    if (highlightNodes.current.has(node.id))
      return node.id === hoverNode.id ? "#ffffff" : node.color;
    return node.color + "18";
  }, [hoverNode]);

  const linkWidth = useCallback((link: any) =>
    highlightLinks.current.has(link) ? Math.max(1.5, link.value * 1.5) : Math.max(0.2, link.value)
  , [hoverNode]);

  const linkColor = useCallback((link: any) =>
    highlightLinks.current.has(link) ? "rgba(255,255,255,0.5)" : link.color
  , [hoverNode]);

  const nodeLabel = useCallback((node: any) =>
    `<div style="background:rgba(0,0,0,0.85);padding:8px 12px;border-radius:8px;font-size:13px;color:${node.color};border:1px solid ${node.color}50">
      <b>${node.label}</b><br/>
      <span style="color:#aaa;font-size:11px">${node.desc}</span>
      ${node.activity > 0 ? `<br/><span style="color:#6ee7b7;font-size:11px">Aktivitaet: ${(node.activity * 100).toFixed(0)}%</span>` : ""}
    </div>`
  , []);

  if (!state) {
    return <div className="h-full flex items-center justify-center text-gray-500">Waiting for brain state...</div>;
  }

  return (
    <div ref={containerRef} className="h-full w-full rounded-lg overflow-hidden relative" style={{ background: "#000003" }}>
      <Suspense fallback={<div className="flex items-center justify-center h-full text-gray-500">Loading 3D brain...</div>}>
        <ForceGraph3D
          ref={fgRef}
          width={dims.w}
          height={dims.h}
          graphData={graphData}
          backgroundColor="#000003"
          showNavInfo={false}
          nodeRelSize={4}
          nodeVal={(n: any) => n.val}
          nodeColor={nodeColor}
          nodeOpacity={0.85}
          nodeResolution={16}
          nodeThreeObject={nodeThreeObject}
          nodeThreeObjectExtend={true}
          nodeLabel={nodeLabel}
          linkColor={linkColor}
          linkWidth={linkWidth}
          linkOpacity={0.4}
          linkCurvature={0.25}
          linkCurveRotation={0.5}
          linkDirectionalParticles={(l: any) => l.particles}
          linkDirectionalParticleSpeed={(l: any) => l.particleSpeed}
          linkDirectionalParticleWidth={2.5}
          linkDirectionalParticleColor={() => "#34d399"}
          cooldownTicks={60}
          d3AlphaDecay={0.08}
          d3VelocityDecay={0.5}
          warmupTicks={20}
          enableNodeDrag={false}
          onNodeHover={handleNodeHover}
          onNodeClick={(node: any) => {
            if (node.type === "concept") {
              const label = prompt(`Label fuer ${node.id}:`);
              if (label) fetch("/api/label", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ concept_id: parseInt(node.id.replace("c", "")), label }) });
            }
            if (fgRef.current && node.x != null) {
              fgRef.current.cameraPosition(
                { x: node.x - 40, y: node.y + 30, z: node.z + 60 },
                { x: node.x, y: node.y, z: node.z }, 800);
            }
          }}
        />
      </Suspense>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 bg-black/70 backdrop-blur-sm rounded-lg p-2.5 text-[10px] font-mono space-y-1 pointer-events-none">
        {REGIONS.map(r => (
          <div key={r.id} className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: r.color, boxShadow: `0 0 6px ${r.color}` }} />
            <span className="font-bold" style={{ color: r.color }}>{r.label}</span>
            <span className="text-gray-500 ml-1">{r.desc}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
