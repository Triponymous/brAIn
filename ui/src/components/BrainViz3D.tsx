import { useRef, useEffect, useState, useCallback, lazy, Suspense } from "react";
import type { BrainState } from "../lib/ws";
import { buildMacroGraph, updateMacroGraph } from "./viz/MacroGraph";
import { buildMesoGraph } from "./viz/MesoGraph";
import { useBrainSubscription } from "./viz/useBrainSubscription";
import { nodeThreeObject, nodeColor, nodeTooltip } from "./viz/nodeRenderers";
import { REGION_DEFS } from "./viz/constants";
import type { ZoomLevel, VizNode, VizGraph } from "./viz/constants";

const ForceGraph3D = lazy(() => import("react-force-graph-3d"));

export function BrainViz3D({ state }: { state: BrainState | null }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const fgRef = useRef<any>(null);
  const [dims, setDims] = useState({ w: 800, h: 500 });

  // Zoom state
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>("macro");
  const [focusRegion, setFocusRegion] = useState<string | null>(null);
  const [focusNeuron, setFocusNeuron] = useState<number | null>(null);

  // Hover
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null);
  const highlightSet = useRef(new Set<string>());

  // Graph data + node index for O(1) lookups
  const graphRef = useRef<VizGraph>({ nodes: [], links: [] });
  const nodeIndexRef = useRef(new Map<string, VizNode>());
  // Bumping this forces react-force-graph to pick up new graphData
  const [graphVersion, setGraphVersion] = useState(0);

  // Subscribe to detail data based on zoom level
  useBrainSubscription(zoomLevel, focusRegion, focusNeuron);

  // ── Resize observer ──
  useEffect(() => {
    if (!containerRef.current) return;
    const ro = new ResizeObserver(([e]) =>
      setDims({ w: e.contentRect.width, h: e.contentRect.height })
    );
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // ── Camera + auto-orbit init ──
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

  // ── Build / update graph from state ──
  // CRITICAL: Never remount ForceGraph3D — it resets the force simulation.
  // Instead, mutate in-place for property changes, or call fg.graphData()
  // only for genuine topology changes (zoom level switch).
  const lastZoomRef = useRef<string>("");

  useEffect(() => {
    if (!state) return;
    const fg = fgRef.current;
    const zoomKey = `${zoomLevel}:${focusRegion}`;
    const isZoomChange = zoomKey !== lastZoomRef.current;

    if (zoomLevel === "macro") {
      if (isZoomChange || graphRef.current.nodes.length === 0) {
        // Zoom changed or first render: full rebuild
        const fresh = buildMacroGraph(state as any);
        graphRef.current = fresh;
        nodeIndexRef.current = new Map(fresh.nodes.map((n) => [n.id, n]));
        if (fg) fg.graphData(fresh);
        lastZoomRef.current = zoomKey;
      } else {
        // Same zoom level: in-place property update only (no topology change)
        updateMacroGraph(graphRef.current, state as any, nodeIndexRef.current);
        // Force a re-render without resetting physics
        if (fg) fg.refresh();
      }
    } else if (zoomLevel === "meso" && focusRegion) {
      // Meso: rebuild on zoom change AND periodically (every 2s) to pick up
      // new spike data. The first build after zoom often has empty region_spikes
      // because the subscription message hasn't been processed yet.
      const fresh = buildMesoGraph(state as any, focusRegion);
      if (isZoomChange || graphRef.current.nodes.length === 0) {
        graphRef.current = fresh;
        nodeIndexRef.current = new Map(fresh.nodes.map((n) => [n.id, n]));
        if (fg) fg.graphData(fresh);
        lastZoomRef.current = zoomKey;
      } else {
        // Update existing node properties in-place
        for (const fn of fresh.nodes) {
          const existing = nodeIndexRef.current.get(fn.id);
          if (existing) {
            existing.val = fn.val;
            existing.color = fn.color;
            existing.activity = fn.activity;
          }
        }
        if (fg) fg.refresh();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state, zoomLevel, focusRegion]);

  // ── Node click → zoom transitions ──
  const handleNodeClick = useCallback(
    (node: any) => {
      if (!node) return;

      if (node.type === "region" && node.regionId) {
        // Click region → fly camera to it
        const fg = fgRef.current;
        if (fg) {
          fg.cameraPosition(
            { x: (node.fx ?? 0), y: (node.fy ?? 0), z: (node.fz ?? 0) + 120 },
            { x: node.fx ?? 0, y: node.fy ?? 0, z: node.fz ?? 0 },
            800,
          );
        }
      } else if (node.type === "neuron" && node.regionId === "concept") {
        // Click concept neuron → label it
        const idx = parseInt(node.id.replace("c_", ""));
        const label = prompt(`Label fuer Concept #${idx}:`);
        if (label) {
          fetch("/api/label", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ concept_id: idx, label }),
          });
        }
      }
    },
    [zoomLevel],
  );

  // ── Escape → zoom out ──
  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (zoomLevel === "micro") {
          setFocusNeuron(null);
          setZoomLevel("meso");
        } else if (zoomLevel === "meso") {
          setFocusRegion(null);
          setZoomLevel("macro");
          const fg = fgRef.current;
          if (fg) {
            fg.cameraPosition({ x: -60, y: 90, z: 350 }, { x: 0, y: 0, z: 0 }, 800);
            const controls = fg.controls();
            if (controls && "autoRotate" in controls) controls.autoRotate = true;
          }
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [zoomLevel]);

  // ── Hover highlighting ──
  const handleNodeHover = useCallback((node: any) => {
    highlightSet.current.clear();
    if (node) highlightSet.current.add(node.id);
    setHoverNodeId(node?.id ?? null);
  }, []);

  // ── Render ──
  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Waiting for brain state...
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-full w-full rounded-lg overflow-hidden relative"
      style={{ background: "#000003" }}
    >
      <Suspense
        fallback={
          <div className="flex items-center justify-center h-full text-gray-500">
            Loading 3D brain...
          </div>
        }
      >
        <ForceGraph3D
          ref={fgRef}
          width={dims.w}
          height={dims.h}
          graphData={graphRef.current}
          backgroundColor="#000003"
          showNavInfo={false}
          nodeRelSize={4}
          nodeVal={(n: any) => n.val}
          nodeColor={(n: any) =>
            nodeColor(n, hoverNodeId, highlightSet.current)
          }
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

      {/* Zoom indicator + back button */}
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
                const fg = fgRef.current;
                if (fg) {
                  fg.cameraPosition(
                    { x: -60, y: 90, z: 350 },
                    { x: 0, y: 0, z: 0 },
                    800,
                  );
                  const controls = fg.controls();
                  if (controls && "autoRotate" in controls)
                    controls.autoRotate = true;
                }
              }
            }}
            className="bg-gray-800/80 hover:bg-gray-700 text-gray-300 text-xs px-2 py-1 rounded backdrop-blur-sm"
          >
            &larr; Zurueck
          </button>
        )}
        <span className="bg-black/60 text-gray-400 text-[10px] px-2 py-0.5 rounded font-mono backdrop-blur-sm">
          {zoomLevel === "macro"
            ? "Uebersicht"
            : zoomLevel === "meso"
              ? `Region: ${focusRegion}`
              : `Neuron #${focusNeuron}`}
        </span>
      </div>

      {/* Legend */}
      <div className="absolute bottom-3 left-3 bg-black/70 backdrop-blur-sm rounded-lg p-2.5 text-[10px] font-mono space-y-1 pointer-events-none">
        {REGION_DEFS.map((r) => (
          <div key={r.id} className="flex items-center gap-1.5">
            <span
              className="w-2 h-2 rounded-full shrink-0"
              style={{
                backgroundColor: r.color,
                boxShadow: `0 0 6px ${r.color}`,
              }}
            />
            <span className="font-bold" style={{ color: r.color }}>
              {r.label}
            </span>
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
