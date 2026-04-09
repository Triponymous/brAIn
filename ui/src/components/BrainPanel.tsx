import { useRef, useEffect } from "react";
import type { BrainState } from "../lib/ws";

export function BrainPanel({ state }: { state: BrainState | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!state || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    const W = rect.width;
    const H = rect.height;

    // Clear
    ctx.fillStyle = "#0a0a0f";
    ctx.fillRect(0, 0, W, H);

    const membrane = state.concept_membrane ?? [];
    const n = membrane.length;
    if (n === 0) return;

    // Draw concept neurons as circles in a ring
    const cx = W / 2;
    const cy = H / 2 - 20;
    const radius = Math.min(W, H) * 0.35;

    // Title
    ctx.fillStyle = "#6b7280";
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Concept Layer (${n} neurons)`, cx, 16);

    // Draw connections (simple: all-to-all, faint)
    ctx.strokeStyle = "rgba(100, 200, 150, 0.05)";
    ctx.lineWidth = 0.5;
    const positions: [number, number][] = [];
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * Math.PI * 2 - Math.PI / 2;
      positions.push([cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius]);
    }
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        ctx.beginPath();
        ctx.moveTo(positions[i][0], positions[i][1]);
        ctx.lineTo(positions[j][0], positions[j][1]);
        ctx.stroke();
      }
    }

    // Draw neurons
    for (let i = 0; i < n; i++) {
      const [x, y] = positions[i];
      const v = Math.abs(membrane[i]);
      const r = 8 + v * 20;
      const alpha = 0.2 + Math.min(0.8, v * 2);

      // Glow for active neurons
      if (v > 0.3) {
        ctx.beginPath();
        ctx.arc(x, y, r + 6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(52, 211, 153, ${alpha * 0.3})`;
        ctx.fill();
      }

      // Neuron circle
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(52, 211, 153, ${alpha})`;
      ctx.fill();

      // Label
      ctx.fillStyle = "#d1d5db";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(`${i}`, x, y + 3);
    }

    // Modulator bars at the bottom
    const mods = state.modulators;
    const modNames = ["DA", "NE", "ACh", "5HT"];
    const modColors = ["#eab308", "#ef4444", "#60a5fa", "#22c55e"];
    const barY = H - 30;
    const barW = 60;
    const barH = 8;
    const startX = cx - (modNames.length * (barW + 10)) / 2;

    for (let i = 0; i < modNames.length; i++) {
      const x = startX + i * (barW + 10);
      const val = mods[modNames[i]] ?? 0;
      const fillW = Math.max(0, Math.min(1, (val + 1) / 2)) * barW;

      // Background
      ctx.fillStyle = "#1f2937";
      ctx.fillRect(x, barY, barW, barH);
      // Fill
      ctx.fillStyle = modColors[i];
      ctx.fillRect(x, barY, fillW, barH);
      // Label
      ctx.fillStyle = "#9ca3af";
      ctx.font = "9px monospace";
      ctx.textAlign = "center";
      ctx.fillText(modNames[i], x + barW / 2, barY - 4);
    }
  }, [state]);

  if (!state) {
    return (
      <div className="h-full flex items-center justify-center text-gray-500">
        Waiting for brain state...
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col gap-2">
      <canvas
        ref={canvasRef}
        className="flex-1 w-full rounded-lg"
        style={{ background: "#0a0a0f" }}
      />
    </div>
  );
}
