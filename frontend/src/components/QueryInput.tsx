import { useState } from "react";
import { ArrowRight, ChevronDown, ChevronUp, Loader } from "lucide-react";
import type { QueryRequest } from "../types";

interface Props {
  onSubmit: (req: QueryRequest) => void;
  loading: boolean;
}

const SUGGESTIONS = [
  "What wire gauge is used for BATT JUMPER RED?",
  "List all connectors in the main harness.",
  "What is the voltage rating of the power relay?",
];

export default function QueryInput({ onSubmit, loading }: Props) {
  const [query, setQuery] = useState("");
  const [groundTruth, setGroundTruth] = useState("");
  const [showGT, setShowGT] = useState(false);

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit({ query: query.trim(), ground_truth: groundTruth.trim() || undefined });
  }

  function useSuggestion(s: string) {
    setQuery(s);
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      {/* Main input row */}
      <div className="flex items-center gap-2 rounded-xl border border-gray-200 bg-gray-50 px-4 py-1 focus-within:border-indigo-400 focus-within:bg-white focus-within:ring-2 focus-within:ring-indigo-100 transition-all">
        <input
          className="flex-1 bg-transparent py-2.5 text-[15px] text-gray-900 placeholder-gray-400 outline-none disabled:opacity-50"
          placeholder="Ask anything about the FLVM system…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
          autoFocus
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          className="flex shrink-0 items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {loading ? (
            <Loader size={14} className="animate-spin" />
          ) : (
            <ArrowRight size={14} />
          )}
          {loading ? "Running…" : "Ask"}
        </button>
      </div>

      {/* Suggestion chips */}
      {!query && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => useSuggestion(s)}
              className="rounded-full border border-gray-200 bg-white px-3 py-1 text-xs text-gray-500 transition hover:border-indigo-300 hover:text-indigo-600"
            >
              {s}
            </button>
          ))}
        </div>
      )}

      {/* Ground truth toggle */}
      <button
        type="button"
        onClick={() => setShowGT(!showGT)}
        className="flex items-center gap-1 self-start text-xs text-gray-400 hover:text-indigo-500 transition-colors"
      >
        {showGT ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {showGT ? "Hide" : "Add ground truth"} for metric evaluation
      </button>

      {showGT && (
        <input
          className="rounded-xl border border-gray-200 bg-gray-50 px-4 py-2.5 text-sm text-gray-800 outline-none focus:border-indigo-400 focus:bg-white focus:ring-2 focus:ring-indigo-100 transition-all"
          placeholder="Expected answer — enables EM / F1 / ANLS scoring"
          value={groundTruth}
          onChange={(e) => setGroundTruth(e.target.value)}
        />
      )}
    </form>
  );
}
