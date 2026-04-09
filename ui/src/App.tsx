import { useEffect, useState } from "react";
import { connectWS, subscribe, BrainState } from "./lib/ws";
import { SensorPanel } from "./components/SensorPanel";
import { BrainPanel3D } from "./components/BrainPanel3D";
import { ChatPanel } from "./components/ChatPanel";
import { WishBanner } from "./components/WishBanner";

export default function App() {
  const [state, setState] = useState<BrainState | null>(null);

  useEffect(() => {
    connectWS();
    return subscribe(setState);
  }, []);

  return (
    <div className="h-screen bg-gray-950 text-gray-100 flex flex-col">
      {/* Header */}
      <header className="h-10 px-4 flex items-center gap-4 bg-gray-900 border-b border-gray-800 text-sm font-mono">
        <span className="text-green-400">●</span>
        <span>braind</span>
        <span className="text-gray-500">
          tick {state?.tick ?? "—"}
        </span>
        <span className="text-gray-500">
          DA {state?.modulators?.DA?.toFixed(2) ?? "—"}
          {" "}NE {state?.modulators?.NE?.toFixed(2) ?? "—"}
          {" "}ACh {state?.modulators?.ACh?.toFixed(2) ?? "—"}
          {" "}5HT {state?.modulators?.["5HT"]?.toFixed(2) ?? "—"}
        </span>
      </header>

      <WishBanner />

      {/* Three columns */}
      <div className="flex-1 flex overflow-hidden">
        <div className="w-64 border-r border-gray-800 overflow-y-auto p-3">
          <SensorPanel state={state} />
        </div>
        <div className="flex-1 overflow-hidden p-3">
          <BrainPanel3D state={state} />
        </div>
        <div className="w-80 border-l border-gray-800 flex flex-col">
          <ChatPanel />
        </div>
      </div>
    </div>
  );
}
