import type { BrainState } from "../lib/ws";

type ExtendedState = BrainState & {
  sensors?: { app?: string; keys?: number; mouse?: number; idle?: number; mic_rms?: number };
  spike_counts?: { sensory?: number; feature?: number; concept?: number };
};

export function SensorPanel({ state }: { state: BrainState | null }) {
  if (!state) return <div className="text-gray-500 text-xs">Connecting...</div>;

  const s = state as ExtendedState;
  const mods = s.modulators;
  const sensors = s.sensors ?? {};
  const spikes = s.spike_counts ?? {};

  return (
    <div className="space-y-3 text-xs font-mono">
      <h3 className="text-gray-400 uppercase tracking-wider text-[10px]">Live Sensors</h3>

      {/* Active App */}
      <div className="bg-gray-900 rounded p-2">
        <div className="text-gray-500 text-[10px]">Active App</div>
        <div className="text-emerald-400 text-sm font-bold truncate">
          {sensors.app || "—"}
        </div>
      </div>

      {/* Input Activity */}
      <div className="bg-gray-900 rounded p-2 space-y-1">
        <div className="text-gray-500 text-[10px]">Input</div>
        <div className="flex justify-between">
          <span className="text-gray-400">⌨ Keys</span>
          <span className="text-gray-200">{sensors.keys ?? "—"}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">🖱 Mouse</span>
          <span className="text-gray-200">{sensors.mouse ?? "—"}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">💤 Idle</span>
          <span className="text-gray-200">{sensors.idle ?? "—"}s</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-400">🎤 Mic</span>
          <span className="text-gray-200">{(sensors.mic_rms ?? 0).toFixed(4)}</span>
        </div>
      </div>

      {/* Spike Pipeline */}
      <div className="bg-gray-900 rounded p-2 space-y-1">
        <div className="text-gray-500 text-[10px]">Spike Pipeline</div>
        {(["sensory", "feature", "concept"] as const).map((name) => {
          const count = spikes[name] ?? 0;
          const max = name === "concept" ? 200 : 200;
          return (
            <div key={name}>
              <div className="flex justify-between text-gray-400">
                <span className="capitalize">{name}</span>
                <span className="text-gray-200">{count}</span>
              </div>
              <div className="w-full bg-gray-800 rounded-full h-1.5 mt-0.5">
                <div
                  className="bg-emerald-500 h-1.5 rounded-full transition-all duration-300"
                  style={{ width: `${Math.min(100, (count / max) * 100)}%` }}
                />
              </div>
            </div>
          );
        })}
      </div>

      {/* Modulators */}
      <div className="bg-gray-900 rounded p-2">
        <div className="text-gray-500 text-[10px] mb-1">Modulators</div>
        {Object.entries(mods).map(([k, v]) => (
          <div key={k} className="flex items-center gap-2 mb-1">
            <span className="w-8 text-gray-400">{k}</span>
            <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all ${
                  k === "DA" ? "bg-yellow-500" :
                  k === "NE" ? "bg-red-500" :
                  k === "ACh" ? "bg-blue-400" :
                  "bg-green-500"
                }`}
                style={{ width: `${Math.max(0, Math.min(100, ((v as number) + 1) / 2 * 100))}%` }}
              />
            </div>
            <span className="w-10 text-right text-gray-500">{typeof v === 'number' ? v.toFixed(2) : '—'}</span>
          </div>
        ))}
      </div>

      <div className="text-gray-600 text-[10px]">
        Tick: {state.tick.toLocaleString()}
      </div>
    </div>
  );
}
