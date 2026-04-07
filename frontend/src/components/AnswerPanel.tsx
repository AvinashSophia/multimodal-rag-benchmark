import { CheckCircle, Clock, Sparkles } from "lucide-react";
import type { QueryResponse } from "../types";

interface Props {
  result: QueryResponse;
}

export default function AnswerPanel({ result }: Props) {
  return (
    <div className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-gray-100">
      {/* Top accent bar */}
      <div className="h-0.5 w-full" style={{ background: "linear-gradient(90deg, #4F46E5, #7C3AED)" }} />

      <div className="p-5">
        {/* Header */}
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-50">
              <Sparkles size={13} className="text-indigo-500" />
            </div>
            <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
              Answer
            </span>
          </div>
          <span className="flex items-center gap-1 rounded-full bg-gray-50 px-2.5 py-1 text-xs text-gray-400">
            <Clock size={10} />
            {result.latency_ms.toFixed(0)} ms
          </span>
        </div>

        {/* Answer text */}
        <p className="text-[15px] leading-relaxed text-gray-800">{result.answer}</p>

        {/* Sources */}
        {result.sources.length > 0 && (
          <div className="mt-4 flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-medium text-gray-400 mr-1">Sources:</span>
            {result.sources.map((src) => (
              <span
                key={src}
                className="flex items-center gap-1 rounded-full bg-indigo-50 px-2.5 py-0.5 text-xs font-medium text-indigo-700"
              >
                <CheckCircle size={9} />
                {src}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
