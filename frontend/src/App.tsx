import { useState, useEffect } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { SlidersHorizontal, ChevronUp, ChevronDown, Shield, LogOut } from "lucide-react";
import { submitQuery, fetchHealth, fetchConfigOptions } from "./api/client";
import type { QueryRequest, QueryResponse, ConfigOptions, User } from "./types";
import QueryInput from "./components/QueryInput";
import AnswerPanel from "./components/AnswerPanel";
import RetrievedChunks from "./components/RetrievedChunks";
import RetrievedImages from "./components/RetrievedImages";
import MetricsPanel from "./components/MetricsPanel";
import StatusBar from "./components/StatusBar";
import ConfigSelector from "./components/ConfigSelector";
import LoginPage from "./components/LoginPage";
import ThinkingOverlay from "./components/ThinkingOverlay";

interface SelectedConfig {
  dataset: string;
  model: string;
  text_method: string;
  image_method: string;
}

function defaultConfig(options: ConfigOptions): SelectedConfig {
  return {
    dataset: options.active_dataset,
    model: options.active_model,
    text_method: options.active_text_method,
    image_method: options.active_image_method,
  };
}

function LogoMark() {
  return (
    <img
      src="https://sophiaspatialai.com/wp-content/uploads/logo.webp"
      alt="SophiaSpatial AI"
      className="h-12 w-auto object-contain"
    />
  );
}

const USER_KEY = "sophiaspatial_user";

function loadUser(): User | null {
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch { return null; }
}

export default function App() {
  const [user, setUser] = useState<User | null>(loadUser);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedConfig, setSelectedConfig] = useState<SelectedConfig | null>(null);
  const [configOpen, setConfigOpen] = useState(false);

  function handleLogin(u: User) {
    localStorage.setItem(USER_KEY, JSON.stringify(u));
    setUser(u);
  }

  function handleLogout() {
    localStorage.removeItem(USER_KEY);
    setUser(null);
    setResult(null);
    setError(null);
  }

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

  if (!user) return <LoginPage onLogin={handleLogin} />;

  return (
    <div className="min-h-screen" style={{ background: "#09090F" }}>
      <ThinkingOverlay visible={mutation.isPending} />

      {/* Page content — blurred while query is running */}
      <div style={{ transition: "filter 0.3s", filter: mutation.isPending ? "blur(3px)" : "none", pointerEvents: mutation.isPending ? "none" : undefined }}>

      {/* ── Top Nav ─────────────────────────────────────────────── */}
      <header
        className="sticky top-0 z-40 border-b"
        style={{
          background: "rgba(9,9,15,0.85)",
          backdropFilter: "blur(20px)",
          borderColor: "rgba(237,203,105,0.12)",
        }}
      >
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">

          {/* Brand */}
          <div className="flex items-center gap-3">
            <LogoMark />
            <div>
              <div className="flex items-baseline gap-2.5">
                <span className="text-base font-bold tracking-tight text-white">SophiaSpatial</span>
                <span
                  className="hidden text-xs font-semibold sm:block px-1.5 py-0.5 rounded"
                  style={{ background: "rgba(237,203,105,0.12)", color: "#EDCB69" }}
                >
                  DocuMind
                </span>
              </div>
              <p className="text-[10px] font-medium uppercase tracking-widest" style={{ color: "rgba(237,203,105,0.5)" }}>
                Wisdom You Can See™
              </p>
            </div>
          </div>

          {/* Right: private badge + status + config toggle + user */}
          <div className="flex items-center gap-3">
            <div
              className="hidden sm:flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-medium"
              style={{ background: "rgba(237,203,105,0.08)", color: "rgba(237,203,105,0.6)", border: "1px solid rgba(237,203,105,0.15)" }}
            >
              <Shield size={9} />
              Private · On-Premises
            </div>
            <StatusBar health={health} loading={mutation.isPending} />
            {/* User avatar + logout */}
            <div className="flex items-center gap-2">
              <div
                className="flex items-center gap-2 rounded-full px-2.5 py-1.5"
                style={{ background: "rgba(237,203,105,0.08)", border: "1px solid rgba(237,203,105,0.15)" }}
              >
                <div
                  className="flex h-5 w-5 items-center justify-center rounded-full text-[10px] font-bold"
                  style={{ background: "rgba(237,203,105,0.2)", color: "#EDCB69" }}
                >
                  {user.name.charAt(0).toUpperCase()}
                </div>
                <span className="hidden text-xs font-medium sm:block" style={{ color: "rgba(237,203,105,0.8)" }}>
                  {user.name.split(" ")[0]}
                </span>
              </div>
              <button
                onClick={handleLogout}
                title="Sign out"
                className="flex h-7 w-7 items-center justify-center rounded-full transition"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.3)" }}
              >
                <LogOut size={12} />
              </button>
            </div>
            {configOptions && selectedConfig && (
              <button
                onClick={() => setConfigOpen((o) => !o)}
                className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium transition-all"
                style={configOpen
                  ? { background: "rgba(237,203,105,0.15)", border: "1px solid rgba(237,203,105,0.4)", color: "#EDCB69" }
                  : { background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.6)" }
                }
              >
                <SlidersHorizontal size={11} />
                Pipeline
                {configOpen ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
              </button>
            )}
          </div>
        </div>

        {/* Config panel */}
        {configOpen && configOptions && selectedConfig && (
          <div
            className="border-t"
            style={{ background: "rgba(9,9,15,0.98)", borderColor: "rgba(237,203,105,0.1)" }}
          >
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
      <main className="mx-auto max-w-7xl px-4 pb-16 pt-8 sm:px-6 lg:px-8">

        {/* Hero search card */}
        <div
          className="overflow-hidden rounded-2xl"
          style={{
            background: "rgba(255,255,255,0.03)",
            border: "1px solid rgba(237,203,105,0.15)",
            backdropFilter: "blur(10px)",
          }}
        >
          <div
            className="h-px w-full"
            style={{ background: "linear-gradient(90deg, transparent, #EDCB69, #FFE7A2, #EDCB69, transparent)" }}
          />
          <div className="px-6 py-6">
            <p className="mb-1 text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(237,203,105,0.7)" }}>
              DocuMind · FLVM Intelligence
            </p>
            <p className="mb-5 text-[11px]" style={{ color: "rgba(255,255,255,0.3)" }}>
              Ask anything about the FLVM technical documentation
            </p>
            <QueryInput onSubmit={handleSubmit} loading={mutation.isPending} />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            className="mt-4 flex items-start gap-3 rounded-xl px-4 py-3 text-sm"
            style={{ background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.2)", color: "#FCA5A5" }}
          >
            <span className="mt-0.5">⚠</span>
            {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <div className="mt-6 flex flex-col gap-5">
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-4">
              <div className={result.metrics ? "lg:col-span-3" : "lg:col-span-4"}>
                <AnswerPanel
                  result={result}
                  user={user}
                  config={selectedConfig ? { ...selectedConfig } : undefined}
                />
              </div>
              {result.metrics && (
                <div className="lg:col-span-1">
                  <MetricsPanel metrics={result.metrics} />
                </div>
              )}
            </div>
            <div className="grid grid-cols-1 gap-5 lg:grid-cols-3">
              <div className="lg:col-span-2">
                <RetrievedChunks chunks={result.retrieved_text} />
              </div>
              <RetrievedImages images={result.retrieved_images} dataset={selectedConfig?.dataset ?? "altumint"} query={result.query} />
            </div>
          </div>
        )}

        {/* Empty state */}
        {!result && !mutation.isPending && !error && (
          <div
            className="mt-6 flex flex-col items-center justify-center rounded-2xl py-20 text-center"
            style={{ border: "1px dashed rgba(237,203,105,0.12)", background: "rgba(237,203,105,0.02)" }}
          >
            <div
              className="mb-4 flex h-14 w-14 items-center justify-center rounded-2xl"
              style={{ background: "rgba(237,203,105,0.08)", border: "1px solid rgba(237,203,105,0.15)" }}
            >
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#EDCB69" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" opacity="0.7">
                <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
              </svg>
            </div>
            <p className="text-sm font-medium" style={{ color: "rgba(255,255,255,0.4)" }}>
              Ask a question to explore the FLVM documentation
            </p>
            <p className="mt-1 text-xs" style={{ color: "rgba(255,255,255,0.2)" }}>
              Try:{" "}
              <span
                className="cursor-pointer transition-colors hover:underline"
                style={{ color: "rgba(237,203,105,0.6)" }}
              >
                "What wire gauge is used for BATT JUMPER RED?"
              </span>
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer
        className="border-t py-4 text-center text-xs"
        style={{ borderColor: "rgba(237,203,105,0.08)", color: "rgba(255,255,255,0.2)" }}
      >
        SophiaSpatial AI · DocuMind · {new Date().getFullYear()} · Wisdom You Can See™
      </footer>

      </div>{/* end blur wrapper */}
    </div>
  );
}
