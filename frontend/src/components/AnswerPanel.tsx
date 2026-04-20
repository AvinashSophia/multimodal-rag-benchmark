import React, { useState } from "react";
import { CheckCircle, Clock, Sparkles, ThumbsUp, ThumbsDown, Send } from "lucide-react";
import type { QueryResponse, User, FeedbackRequest } from "../types";
import { submitFeedback } from "../api/client";

interface Props {
  result: QueryResponse;
  user: User | null;
  config?: Record<string, string>;
}

function latencyMeta(ms: number): { style: React.CSSProperties; label: string } {
  if (ms < 500)   return { style: { background: "rgba(52,211,153,0.12)",  border: "1px solid rgba(52,211,153,0.35)",  color: "#34D399" }, label: "real-time capable" };
  if (ms < 2000)  return { style: { background: "rgba(237,203,105,0.12)", border: "1px solid rgba(237,203,105,0.35)", color: "#EDCB69" }, label: "interactive" };
  return           { style: { background: "rgba(248,113,113,0.12)",  border: "1px solid rgba(248,113,113,0.35)",  color: "#F87171" }, label: "too slow for AR/VR" };
}

export default function AnswerPanel({ result, user, config }: Props) {
  const [rating, setRating] = useState<"positive" | "negative" | null>(null);
  const [feedbackText, setFeedbackText] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  async function handleRate(r: "positive" | "negative") {
    if (submitted) return;
    setRating(r);
  }

  async function handleSubmitFeedback() {
    if (!rating || submitting) return;
    setSubmitting(true);
    const req: FeedbackRequest = {
      query: result.query,
      answer: result.answer,
      rating,
      feedback_text: feedbackText.trim() || undefined,
      sources: result.sources,
      config,
      user_name: user?.name,
      user_email: user?.email,
    };
    try {
      await submitFeedback(req);
    } finally {
      setSubmitted(true);
      setSubmitting(false);
    }
  }

  return (
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
        style={{ background: "linear-gradient(90deg, transparent, #EDCB69, transparent)" }}
      />

      <div className="p-5">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md" style={{ background: "rgba(237,203,105,0.1)" }}>
              <Sparkles size={13} style={{ color: "#EDCB69" }} />
            </div>
            <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.4)" }}>
              Answer
            </span>
          </div>
          {(() => {
            const { style, label } = latencyMeta(result.latency_ms);
            return (
              <span className="flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold" style={style}>
                <Clock size={10} />
                {result.latency_ms.toFixed(0)} ms
                <span className="opacity-70">· {label}</span>
              </span>
            );
          })()}
        </div>

        {/* Answer text */}
        <p className="text-[15px] leading-relaxed" style={{ color: "rgba(255,255,255,0.88)" }}>
          {result.answer}
        </p>

        {/* Sources */}
        {result.sources.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-1.5">
            <span className="mr-1 text-[11px] font-medium" style={{ color: "rgba(255,255,255,0.25)" }}>Sources:</span>
            {result.sources.map((src) => (
              <span
                key={src}
                className="flex items-center gap-1 rounded-full px-2.5 py-0.5 text-xs font-medium"
                style={{ background: "rgba(237,203,105,0.08)", border: "1px solid rgba(237,203,105,0.2)", color: "#EDCB69" }}
              >
                <CheckCircle size={9} />
                {src}
              </span>
            ))}
          </div>
        )}

        {/* Latency breakdown */}
        {result.latency_breakdown && (
          <div className="mt-4">
            <LatencyBreakdown breakdown={result.latency_breakdown} total={result.latency_ms} />
          </div>
        )}

        {/* Cost + token usage */}
        {(result.token_usage || result.cost_usd !== null) && (
          <div className="mt-2 flex flex-wrap items-center gap-3 rounded-xl px-3 py-2"
            style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
          >
            {result.token_usage && (
              <>
                <CostStat label="Input" value={`${result.token_usage.input_tokens.toLocaleString()} tokens`} />
                <CostStat label="Output" value={`${result.token_usage.output_tokens.toLocaleString()} tokens`} />
              </>
            )}
            {result.cost_usd !== null && result.cost_usd !== undefined && (
              <CostStat
                label="Cost"
                value={result.cost_usd === 0 ? "self-hosted" : `$${result.cost_usd.toFixed(4)}`}
                highlight={result.cost_usd > 0}
              />
            )}
          </div>
        )}

        {/* Feedback section */}
        <div className="mt-5 border-t pt-4" style={{ borderColor: "rgba(255,255,255,0.06)" }}>
          {submitted ? (
            <p className="text-xs font-medium" style={{ color: "rgba(237,203,105,0.7)" }}>
              ✓ Thanks for your feedback{user ? `, ${user.name}` : ""}!
            </p>
          ) : (
            <>
              <div className="flex items-center gap-3">
                <span className="text-xs" style={{ color: "rgba(255,255,255,0.3)" }}>Was this helpful?</span>
                <button
                  onClick={() => handleRate("positive")}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition"
                  style={rating === "positive"
                    ? { background: "rgba(16,185,129,0.15)", border: "1px solid rgba(16,185,129,0.4)", color: "#10B981" }
                    : { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.4)" }
                  }
                >
                  <ThumbsUp size={12} />
                  Yes
                </button>
                <button
                  onClick={() => handleRate("negative")}
                  className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium transition"
                  style={rating === "negative"
                    ? { background: "rgba(239,68,68,0.15)", border: "1px solid rgba(239,68,68,0.4)", color: "#EF4444" }
                    : { background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.4)" }
                  }
                >
                  <ThumbsDown size={12} />
                  No
                </button>
              </div>

              {/* Text feedback — slides in after rating */}
              {rating && (
                <div className="mt-3 flex gap-2">
                  <input
                    className="flex-1 rounded-xl px-3 py-2 text-xs outline-none transition-all"
                    style={{
                      background: "rgba(255,255,255,0.04)",
                      border: "1px solid rgba(237,203,105,0.15)",
                      color: "rgba(255,255,255,0.8)",
                    }}
                    placeholder="Tell us more (optional)…"
                    value={feedbackText}
                    onChange={(e) => setFeedbackText(e.target.value)}
                    onKeyDown={(e) => { if (e.key === "Enter") handleSubmitFeedback(); }}
                  />
                  <button
                    onClick={handleSubmitFeedback}
                    disabled={submitting}
                    className="flex shrink-0 items-center gap-1.5 rounded-xl px-3 py-2 text-xs font-semibold transition disabled:opacity-40"
                    style={{ background: "linear-gradient(135deg, #EDCB69, #C9A83C)", color: "#09090F" }}
                  >
                    <Send size={11} />
                    Submit
                  </button>
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Cost stat chip
// ---------------------------------------------------------------------------

function CostStat({ label, value, highlight = false }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.3)" }}>{label}</span>
      <span className="text-[10px] font-mono font-semibold" style={{ color: highlight ? "#EDCB69" : "rgba(255,255,255,0.6)" }}>{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Latency breakdown bar
// ---------------------------------------------------------------------------

const STAGES: { key: string; label: string; color: string }[] = [
  { key: "retrieval_ms",  label: "Retrieval",  color: "#38BDF8" }, // cyan
  { key: "generation_ms", label: "Generation", color: "#EDCB69" }, // brand gold
  { key: "evaluation_ms", label: "Evaluation", color: "#34D399" }, // emerald
];

function LatencyBreakdown({ breakdown, total }: { breakdown: Record<string, number>; total: number }) {
  return (
    <div
      className="mb-4 rounded-xl p-3"
      style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)" }}
    >
      <div className="mb-2 flex items-center gap-1.5">
        <Clock size={10} style={{ color: "rgba(255,255,255,0.3)" }} />
        <span className="text-[10px] font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.3)" }}>
          Latency Breakdown
        </span>
      </div>

      {/* Stacked bar */}
      <div className="mb-2 flex h-2 w-full overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.05)" }}>
        {STAGES.map(({ key, color }) => {
          const ms = breakdown[key] ?? 0;
          const pct = total > 0 ? (ms / total) * 100 : 0;
          return (
            <div
              key={key}
              style={{ width: `${pct}%`, background: color, opacity: 0.8 }}
              title={`${key}: ${ms.toFixed(0)} ms`}
            />
          );
        })}
      </div>

      {/* Labels */}
      <div className="flex flex-wrap gap-x-4 gap-y-1">
        {STAGES.filter(({ key }) => (breakdown[key] ?? 0) > 0).map(({ key, label, color }) => {
          const ms = breakdown[key] ?? 0;
          const pct = total > 0 ? ((ms / total) * 100).toFixed(0) : "0";
          return (
            <div key={key} className="flex items-center gap-1.5">
              <div className="h-2 w-2 rounded-full" style={{ background: color }} />
              <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.4)" }}>
                {label}
              </span>
              <span className="text-[10px] font-mono font-semibold" style={{ color: "rgba(255,255,255,0.7)" }}>
                {ms.toFixed(0)} ms
              </span>
              <span className="text-[9px] font-mono" style={{ color: "rgba(255,255,255,0.25)" }}>
                ({pct}%)
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
