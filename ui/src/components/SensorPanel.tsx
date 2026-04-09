import type { BrainState } from "../lib/ws";

export function SensorPanel({ state }: { state: BrainState | null }) {
  if (!state) return <div className="text-gray-500 text-xs">Connecting...</div>;

  const mods = state.modulators;

  return (
    <div className="space-y-4 text-xs font-mono">
      <h3 className="text-gray-400 uppercase tracking-wider text-[10px]">Sensors</h3>

      <div>
        <div className="text-gray-500 mb-1">Modulators</div>
        {Object.entries(mods).map(([k, v]) => (
          <div key={k} className="flex items-center gap-2 mb-1">
            <span className="w-8 text-gray-400">{k}</span>
            <div className="flex-1 bg-gray-800 rounded-full h-2 overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  k === "DA" ? "bg-yellow-500" :
                  k === "NE" ? "bg-red-500" :
                  k === "ACh" ? "bg-blue-400" :
                  "bg-green-500"
                }`}
                style={{ width: `${Math.max(0, Math.min(100, ((v + 1) / 2) * 100))}%` }}
              />
            </div>
            <span className="w-12 text-right text-gray-500">{typeof v === 'number' ? v.toFixed(3) : '—'}</span>
          </div>
        ))}
      </div>

      <div>
        <div className="text-gray-500 mb-1">Working Memory</div>
        <div className="flex gap-0.5 h-8 items-end">
          {(state.wm_membrane ?? []).map((v, i) => (
            <div
              key={i}
              className="flex-1 bg-purple-500 rounded-t"
              style={{ height: `${Math.min(100, Math.max(4, Math.abs(v) * 100))}%`, opacity: 0.3 + Math.abs(v) * 0.7 }}
              title={`WM ${i}: ${v.toFixed(3)}`}
            />
          ))}
        </div>
      </div>

      <div className="text-gray-600 text-[10px]">
        Tick: {state.tick.toLocaleString()}
      </div>
    </div>
  );
}
