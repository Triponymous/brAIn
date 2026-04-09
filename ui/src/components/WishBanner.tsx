import { useState, useEffect } from "react";

type Wish = {
  id: number;
  tool_name: string;
  trigger_concept_label: string;
  reason: string;
};

export function WishBanner() {
  const [wishes, setWishes] = useState<Wish[]>([]);

  useEffect(() => {
    const poll = async () => {
      try {
        const resp = await fetch("/api/wishes");
        const data = await resp.json();
        setWishes(data.wishes || []);
      } catch {}
    };
    poll();
    const interval = setInterval(poll, 10000); // poll every 10s
    return () => clearInterval(interval);
  }, []);

  const handleGrant = async (wish: Wish) => {
    await fetch("/api/wishes/grant", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ wish_id: wish.id, tool_name: wish.tool_name }),
    });
    setWishes((prev) => prev.filter((w) => w.id !== wish.id));
  };

  const handleDeny = async (wish: Wish) => {
    await fetch("/api/wishes/deny", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ wish_id: wish.id, tool_name: wish.tool_name }),
    });
    setWishes((prev) => prev.filter((w) => w.id !== wish.id));
  };

  if (wishes.length === 0) return null;

  return (
    <div className="bg-yellow-900/30 border border-yellow-700/50 rounded-lg p-3 mx-3 mt-2">
      <div className="text-yellow-400 text-xs font-mono uppercase tracking-wider mb-2">
        Capability Request
      </div>
      {wishes.map((w) => (
        <div key={w.id} className="mb-2 last:mb-0">
          <div className="text-sm text-gray-200 mb-1">
            I notice concept "{w.trigger_concept_label}" is very active.
            Should I learn <strong>{w.tool_name.replace("_", " ")}</strong>?
          </div>
          <div className="flex gap-2">
            <button
              className="px-2 py-1 bg-green-800 hover:bg-green-700 rounded text-xs text-green-200"
              onClick={() => handleGrant(w)}
            >
              Grant
            </button>
            <button
              className="px-2 py-1 bg-red-900 hover:bg-red-800 rounded text-xs text-red-300"
              onClick={() => handleDeny(w)}
            >
              Deny
            </button>
          </div>
        </div>
      ))}
    </div>
  );
}
