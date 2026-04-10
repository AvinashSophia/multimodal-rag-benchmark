import { useEffect, useState } from "react";

const STEPS = [
  "Reading the documents…",
  "Retrieving relevant pages…",
  "Analysing visual evidence…",
  "Cross-referencing sources…",
  "Generating your answer…",
];

interface Props {
  visible: boolean;
}

export default function ThinkingOverlay({ visible }: Props) {
  const [stepIndex, setStepIndex] = useState(0);
  const [dots, setDots] = useState(0);

  useEffect(() => {
    if (!visible) { setStepIndex(0); setDots(0); return; }
    const id = setInterval(() => setStepIndex((i) => (i + 1) % STEPS.length), 1800);
    return () => clearInterval(id);
  }, [visible]);

  useEffect(() => {
    if (!visible) return;
    const id = setInterval(() => setDots((d) => (d + 1) % 4), 400);
    return () => clearInterval(id);
  }, [visible]);

  if (!visible) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col items-center justify-center"
      style={{ backdropFilter: "blur(16px)", background: "rgba(9,9,15,0.78)" }}
    >
      {/* ── Ring stack ── */}
      <div className="relative flex items-center justify-center" style={{ width: 300, height: 300 }}>

        {/* Outermost faint pulse ring */}
        <div
          className="absolute inset-0 rounded-full"
          style={{
            border: "1px solid rgba(237,203,105,0.12)",
            animation: "pulse-ring 2.4s ease-in-out infinite",
          }}
        />

        {/* Ring 1 — fast gold sweep */}
        <div
          className="absolute rounded-full"
          style={{
            inset: 12,
            background: "conic-gradient(from 0deg, transparent 55%, #EDCB69 85%, #FFE7A2 100%)",
            animation: "spin 1.8s linear infinite",
          }}
        />
        {/* Ring 1 mask — hides inner area */}
        <div
          className="absolute rounded-full"
          style={{ inset: 22, background: "#09090F" }}
        />

        {/* Ring 2 — slower reverse amber sweep */}
        <div
          className="absolute rounded-full"
          style={{
            inset: 26,
            background: "conic-gradient(from 120deg, transparent 60%, rgba(237,203,105,0.5) 90%, rgba(255,231,162,0.7) 100%)",
            animation: "spin 3.2s linear infinite reverse",
          }}
        />
        {/* Ring 2 mask */}
        <div
          className="absolute rounded-full"
          style={{ inset: 36, background: "#09090F" }}
        />

        {/* Ring 3 — slowest, subtle dashed-like effect */}
        <div
          className="absolute rounded-full"
          style={{
            inset: 40,
            background: "conic-gradient(from 240deg, transparent 70%, rgba(237,203,105,0.25) 100%)",
            animation: "spin 5s linear infinite",
          }}
        />
        {/* Ring 3 mask */}
        <div
          className="absolute rounded-full"
          style={{ inset: 50, background: "#09090F" }}
        />

        {/* Glow disc behind logo */}
        <div
          className="absolute rounded-full"
          style={{
            inset: 54,
            background: "radial-gradient(circle, rgba(237,203,105,0.12) 0%, transparent 70%)",
            animation: "breathe 2s ease-in-out infinite",
          }}
        />

        {/* Logo */}
        <img
          src="https://sophiaspatialai.com/wp-content/uploads/logo.webp"
          alt="SophiaSpatial AI"
          className="relative z-10"
          style={{
            width: 148,
            height: 148,
            objectFit: "contain",
            animation: "breathe 2s ease-in-out infinite",
            filter: "drop-shadow(0 0 28px rgba(237,203,105,0.55)) drop-shadow(0 0 8px rgba(237,203,105,0.3))",
          }}
        />
      </div>

      {/* ── Status text ── */}
      <div className="mt-10 text-center" style={{ minHeight: 56 }}>
        <p
          className="text-base font-semibold tracking-wide"
          style={{ color: "#EDCB69", textShadow: "0 0 20px rgba(237,203,105,0.4)" }}
        >
          {STEPS[stepIndex]}{"·".repeat(dots)}
        </p>
        <p className="mt-2 text-xs font-medium uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.2)" }}>
          DocuMind · FLVM Intelligence
        </p>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes breathe {
          0%, 100% { transform: scale(1);    opacity: 1;    }
          50%       { transform: scale(1.06); opacity: 0.88; }
        }
        @keyframes pulse-ring {
          0%, 100% { transform: scale(1);    opacity: 0.4; }
          50%       { transform: scale(1.04); opacity: 0.1; }
        }
      `}</style>
    </div>
  );
}
