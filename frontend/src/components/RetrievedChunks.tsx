import { useState } from "react";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";
import type { RetrievedTextChunk } from "../types";

interface Props {
  chunks: RetrievedTextChunk[];
}

function ChunkItem({ chunk, rank }: { chunk: RetrievedTextChunk; rank: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className="rounded-xl p-3 transition"
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.07)",
      }}
    >
      <div className="mb-1.5 flex items-center gap-2">
        <span
          className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[9px] font-bold"
          style={{ background: "rgba(237,203,105,0.15)", color: "#EDCB69" }}
        >
          {rank}
        </span>
        <span
          className="min-w-0 flex-1 truncate text-xs font-semibold"
          style={{ color: "rgba(237,203,105,0.8)" }}
          title={chunk.page_id}
        >
          {chunk.page_id}
        </span>
        <span
          className="shrink-0 rounded-full px-2 py-0.5 text-[10px] font-mono font-medium"
          style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "rgba(255,255,255,0.3)" }}
        >
          {chunk.score.toFixed(3)}
        </span>
      </div>
      <p className={`text-xs leading-relaxed ${expanded ? "" : "line-clamp-3"}`} style={{ color: "rgba(255,255,255,0.88)" }}>
        {chunk.text}
      </p>
      {chunk.text.length > 200 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1.5 flex items-center gap-0.5 text-[11px] font-medium transition-colors"
          style={{ color: "rgba(237,203,105,0.5)" }}
        >
          {expanded ? <><ChevronUp size={10} /> Show less</> : <><ChevronDown size={10} /> Show more</>}
        </button>
      )}
    </div>
  );
}

export default function RetrievedChunks({ chunks }: Props) {
  if (chunks.length === 0) return null;

  return (
    <div
      className="overflow-hidden rounded-2xl"
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(237,203,105,0.12)",
        backdropFilter: "blur(10px)",
      }}
    >
      <div
        className="h-px w-full"
        style={{ background: "linear-gradient(90deg, transparent, rgba(237,203,105,0.5), transparent)" }}
      />
      <div className="p-5">
        <div className="mb-3 flex items-center gap-2">
          <div
            className="flex h-6 w-6 items-center justify-center rounded-md"
            style={{ background: "rgba(237,203,105,0.1)" }}
          >
            <FileText size={13} style={{ color: "#EDCB69" }} />
          </div>
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.4)" }}>
            Text Evidence
          </span>
          <span
            className="ml-auto rounded-full px-2 py-0.5 text-[10px] font-medium"
            style={{ background: "rgba(237,203,105,0.08)", color: "rgba(237,203,105,0.7)" }}
          >
            {chunks.length} chunks
          </span>
        </div>
        <div className="flex flex-col gap-2">
          {chunks.map((chunk, i) => (
            <ChunkItem key={i} chunk={chunk} rank={i + 1} />
          ))}
        </div>
      </div>
    </div>
  );
}
