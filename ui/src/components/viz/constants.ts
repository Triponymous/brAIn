// ui/src/components/viz/constants.ts
// Central definitions for the 3D brain visualization.

export const REGION_DEFS = [
  { id: "sensory",     label: "Sensorik",         color: "#6ee7b7", neurons: 200, target: [-120, -120,   0] as [number,number,number] },
  { id: "feature",     label: "Mustererkennung",   color: "#34d399", neurons: 200, target: [ -60,  -60,   0] as [number,number,number] },
  { id: "association", label: "Verknuepfung",      color: "#a78bfa", neurons: 500, target: [   0,    0,   0] as [number,number,number] },
  { id: "concept",     label: "Konzeptbildung",    color: "#fbbf24", neurons: 200, target: [  60,   60,   0] as [number,number,number] },
  { id: "wm",          label: "Gedaechtnis",       color: "#60a5fa", neurons: 100, target: [  60,    0, -40] as [number,number,number] },
  { id: "motor",       label: "Motorik",           color: "#f87171", neurons:  50, target: [ 120,  100,   0] as [number,number,number] },
  { id: "meta",        label: "Meta",              color: "#9ca3af", neurons:  10, target: [   0,  150,   0] as [number,number,number] },
] as const;

export type RegionId = (typeof REGION_DEFS)[number]["id"];

export const SENSOR_DEFS = [
  { id: "s_app",   label: "App",      color: "#60a5fa", offset: 0 },
  { id: "s_keys",  label: "Tastatur", color: "#34d399", offset: 1 },
  { id: "s_mouse", label: "Maus",     color: "#a78bfa", offset: 2 },
  { id: "s_idle",  label: "Idle",     color: "#fbbf24", offset: 3 },
  { id: "s_mic",   label: "Mikrofon", color: "#f87171", offset: 4 },
  { id: "s_time",  label: "Zeit",     color: "#818cf8", offset: 5 },
] as const;

/** Sensor arc: positioned left of sensory region */
export function sensorPosition(offset: number): [number, number, number] {
  return [-180, -120 + offset * 30, 0];
}

/** Information-flow edges between regions */
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
  // Three.js runtime (mutated by force engine)
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
