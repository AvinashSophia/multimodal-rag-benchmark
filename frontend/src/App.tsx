import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { SlidersHorizontal, ChevronUp, ChevronDown } from "lucide-react";
import { submitQuery, fetchHealth, fetchConfigOptions } from "./api/client";
import type { QueryRequest, QueryResponse, ConfigOptions } from "./types";
import QueryInput from "./components/QueryInput";
import AnswerPanel from "./components/AnswerPanel";
import RetrievedChunks from "./components/RetrievedChunks";
import RetrievedImages from "./components/RetrievedImages";
import MetricsPanel from "./components/MetricsPanel";
import StatusBar from "./components/StatusBar";
import ConfigSelector from "./components/ConfigSelector";

interface SelectedConfig {
  model: string;
  text_method: string;
  image_method: string;
}

function defaultConfig(options: ConfigOptions): SelectedConfig {
  return {
    model: options.active_model,
    text_method: options.active_text_method,
    image_method: options.active_image_method,
  };
}

// SophiaSpatial AI hex logo mark
function LogoMark() {
  return (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      <polygon
        points="16,2 28,9 28,23 16,30 4,23 4,9"
        fill="url(#hexGrad)"
        stroke="rgba(255,255,255,0.15)"
        strokeWidth="0.5"
      />
      <path
        d="M11 13.5 C11 11.6 12.3 10.5 14.2 10.5 C15.4 10.5 16.4 11 17 11.9 L15.6 12.8 C15.3 12.3 14.8 12 14.2 12 C13.3 12 12.7 12.6 12.7 13.5 C12.7 14.4 13.3 15 14.2 15 L14.8 15 L14.8 16.5 L14 16.5 C13 16.5 12.3 17.1 12.3 18.1 C12.3 19.1 13.1 19.7 14.2 19.7 C15 19.7 15.7 19.3 16 18.7 L17.4 19.6 C16.8 20.6 15.6 21.2 14.2 21.2 C12.1 21.2 10.6 20 10.6 18.1 C10.6 17 11.2 16.1 12.1 15.7 C11.4 15.2 11 14.4 11 13.5Z"
        fill="white"
        opacity="0.9"
      />
      <circle cx="19.5" cy="15.8" r="3" fill="rgba(165,180,252,0.9)" />
      <circle cx="19.5" cy="15.8" r="1.5" fill="white" />
      <defs>
        <linearGradient id="hexGrad" x1="4" y1="2" x2="28" y2="30" gradientUnits="userSpaceOnUse">
          <stop offset="0%" stopColor="#4F46E5" />
          <stop offset="100%" stopColor="#7C3AED" />
        </linearGradient>
      </defs>
    </svg>
  );
}

export default function App() {
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedConfig, setSelectedConfig] = useState<SelectedConfig | null>(null);
  const [configOpen, setConfigOpen] = useState(false);

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    refetchInterval: 5000,
  });

  const { data: configOptions } = useQuery({
    queryKey: ["configOptions"],
    queryFn: fetchConfigOptions,
  });

  useEffect(() => {
    if (configOptions && !selectedConfig) {
      setSelectedConfig(defaultConfig(configOptions));
    }
  }, [configOptions]);

  const mutation = useMutation({
    mutationFn: (req: QueryRequest) => submitQuery(req),
    onSuccess: (data) => { setResult(data); setError(null); },
    onError: (err: Error) => { setError(err.message); setResult(null); },
  });

  function handleSubmit(base: QueryRequest) {
    mutation.mutate(selectedConfig ? { ...base, ...selectedConfig } : base);
  }

  function handleConfigChange(partial: Partial<SelectedConfig>) {
    setSelectedConfig((prev) => ({ ...(prev ?? {}), ...partial } as SelectedConfig));
  }

  return (
    <div className="min-h-screen" style={{ background: "#F0F2F7" }}>

      {/* ── Top Nav ─────────────────────────────────────────────── */}
      <header style={{ background: "linear-gradient(135deg, #0F0C29 0%, #1a1060 50%, #24243e 100%)" }}
        className="sticky top-0 z-40 shadow-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <LogoMark />
            <div>
              <div className="flex items-baseline gap-2">
                <span className="text-base font-bold tracking-tight text-white">SophiaSpatial</span>
                <span className="hidden text-sm font-light text-indigo-300 sm:block">AI</span>
              </div>
              <p className="text-[10px] font-medium uppercase tracking-widest text-indigo-400">
                FLVM Intelligence Platform
              </p>
            </div>
          </div>

          {/* Right: status + config toggle */}
          <div className="flex items-center gap-3">
            <StatusBar health={health} loading={mutation.isPending} />
            {configOptions && selectedConfig && (
              <button
                onClick={() => setConfigOpen((o) => !o)}
                className={`flex items-center gap-1.5 rounded-full border px-3 py-1.5 text-xs font-medium transition-all ${
                  configOpen
                    ? "border-indigo-400 bg-indigo-500/20 text-indigo-200"
                    : "border-white/10 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white"
                }`}
              >
                <SlidersHorizontal size={11} />
                Pipeline
                {configOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              </button>
            )}
          </div>
        </div>

        {/* Config panel — slides open under header */}
        {configOpen && configOptions && selectedConfig && (
          <div className="border-t border-white/10" style={{ background: "rgba(15,12,41,0.95)" }}>
            <div className="mx-auto max-w-7xl px-6 py-4">
              <ConfigSelector
                options={configOptions}
                selected={selectedConfig}
                onChange={handleConfigChange}
              />
            </div>
          </div>
        )}
      </header>

      {/* ── Main content ───────────────────────────────────────── */}
      <main className="mx-auto max-w-5xl px-4 pb-16 pt-8 sm:px-6">

        {/* Hero search card */}
        <div className="overflow-hidden rounded-2xl bg-white shadow-md">
          {/* Card top accent */}
          <div className="h-1 w-full" style={{ background: "linear-gradient(90deg, #4F46E5, #7C3AED, #06B6D4)" }} />
          <div className="px-6 py-6">
            <p className="mb-4 text-xs font-semibold uppercase tracking-widest text-indigo-400">
              Ask the FLVM Knowledge Base
            </p>
            <QueryInput onSubmit={handleSubmit} loading={mutation.isPending} />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 flex items-start gap-3 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            <span className="mt-0.5 text-base">⚠</span>
            {error}
          </div>
        )}

        {/* Loading skeleton */}
        {mutation.isPending && !result && (
          <div className="mt-6 space-y-4">
            {[80, 60, 40].map((w) => (
              <div key={w} className="h-4 animate-pulse rounded-full bg-gray-200" style={{ width: `${w}%` }} />
            ))}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-6 flex flex-col gap-5">
            {/* Answer + Metrics row */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <AnswerPanel result={result} />
              </div>
              {result.metrics && (
                <MetricsPanel metrics={result.metrics} />
              )}
            </div>

            {/* Evidence row */}
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
              <RetrievedChunks chunks={result.retrieved_text} />
              <RetrievedImages images={result.retrieved_images} />
            </div>
          </div>
        )}

        {/* Empty state */}
        {!result && !mutation.isPending && !error && (
          <div className="mt-6 flex flex-col items-center justify-center rounded-2xl border border-dashed border-gray-200 bg-white/60 py-20 text-center">
            <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-50">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#6366F1" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
              </svg>
            </div>
            <p className="text-sm font-medium text-gray-500">Ask a question to explore the FLVM documentation</p>
            <p className="mt-1 text-xs text-gray-400">
              Try: <span className="cursor-pointer text-indigo-500 hover:underline">"What wire gauge is used for BATT JUMPER RED?"</span>
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-4 text-center text-xs text-gray-400">
        SophiaSpatial AI · Multimodal RAG Benchmark · {new Date().getFullYear()}
      </footer>
    </div>
  );
}
