import { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "assistant";
  text: string;
  backend?: string;
  tool_results?: Array<{ tool: string; result: unknown }>;
};

function loadSavedMessages(): Message[] {
  try {
    const saved = localStorage.getItem("braintest_chat");
    return saved ? JSON.parse(saved) : [];
  } catch { return []; }
}

export function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>(loadSavedMessages);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Persist messages to localStorage
  useEffect(() => {
    localStorage.setItem("braintest_chat", JSON.stringify(messages.slice(-50)));
  }, [messages]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const voiceChat = async () => {
    if (loading) return;
    setLoading(true);
    setMessages((prev) => [...prev, { role: "user", text: "\ud83c\udfa4 (listening for 5s...)" }]);

    try {
      const resp = await fetch("/api/voice-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ duration: 5 }),
      });
      const blob = await resp.blob();
      if (blob.size > 0) {
        const audio = new Audio(URL.createObjectURL(blob));
        audio.play();
        setMessages((prev) => [...prev, { role: "assistant", text: "\ud83d\udd0a (speaking...)" }]);
      } else {
        setMessages((prev) => [...prev, { role: "assistant", text: "(no response)" }]);
      }
    } catch (err) {
      setMessages((prev) => [...prev, { role: "assistant", text: `Voice error: ${err}` }]);
    } finally {
      setLoading(false);
    }
  };

  const send = async () => {
    const msg = input.trim();
    if (!msg || loading) return;
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: msg }]);
    setLoading(true);

    try {
      // Send conversation history so the pet remembers what was said
      const history = messages.slice(-10).map((m) => ({
        role: m.role === "user" ? "user" : "assistant",
        content: m.text,
      }));
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, history }),
      });
      const data = await resp.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.text || "(no response)",
          backend: data.backend,
          tool_results: data.tool_results,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <h3 className="text-gray-400 uppercase tracking-wider text-[10px] p-3 pb-1">Chat</h3>

      <div className="flex-1 overflow-y-auto p-3 space-y-3 text-sm">
        {messages.length === 0 && (
          <div className="text-gray-600 text-xs mt-4">
            Ask the pet anything... try &quot;was siehst du?&quot; or &quot;which concepts are active?&quot;
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-blue-300" : "text-gray-200"}>
            <span className="text-gray-500 text-xs">{m.role === "user" ? "You" : "Pet"}</span>
            {m.backend && (
              <span className="text-gray-600 text-[10px] ml-1">[{m.backend}]</span>
            )}
            <div className="mt-0.5 whitespace-pre-wrap">{m.text}</div>
            {m.tool_results && m.tool_results.length > 0 && (
              <details className="mt-1 text-xs text-gray-500">
                <summary className="cursor-pointer hover:text-gray-400">
                  {m.tool_results.length} tool call(s)
                </summary>
                <pre className="mt-1 text-[10px] overflow-x-auto bg-gray-900 rounded p-2">
                  {JSON.stringify(m.tool_results, null, 2)}
                </pre>
              </details>
            )}
          </div>
        ))}
        {loading && <div className="text-gray-500 text-xs animate-pulse">Thinking...</div>}
        <div ref={bottomRef} />
      </div>

      <div className="p-3 border-t border-gray-800">
        <div className="flex gap-2">
          <input
            className="flex-1 bg-gray-900 border border-gray-700 rounded px-2 py-1.5 text-sm focus:outline-none focus:border-gray-500"
            placeholder="Ask the pet..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
            disabled={loading}
          />
          <button
            className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-sm disabled:opacity-50"
            onClick={send}
            disabled={loading || !input.trim()}
          >
            Send
          </button>
          <button
            className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded text-sm disabled:opacity-50"
            onClick={voiceChat}
            disabled={loading}
            title="Push to talk (5s recording)"
          >
            🎤
          </button>
        </div>
      </div>
    </div>
  );
}
