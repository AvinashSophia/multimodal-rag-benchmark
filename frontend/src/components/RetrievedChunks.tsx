import { useState } from "react";
import { FileText, ChevronDown, ChevronUp } from "lucide-react";
import type { RetrievedTextChunk } from "../types";

interface Props {
  chunks: RetrievedTextChunk[];
}

function ChunkItem({ chunk, rank }: { chunk: RetrievedTextChunk; rank: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-xl border border-gray-100 bg-gray-50/60 p-3 transition hover:border-indigo-100 hover:bg-indigo-50/30">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-[9px] font-bold text-indigo-600">
          {rank}
        </span>
        <span className="min-w-0 flex-1 truncate text-xs font-semibold text-indigo-600" title={chunk.page_id}>
          {chunk.page_id}
        </span>
        <span className="shrink-0 rounded-full bg-white px-2 py-0.5 text-[10px] font-mono font-medium text-gray-400 shadow-sm ring-1 ring-gray-100">
          {chunk.score.toFixed(3)}
        </span>
      </div>
      <p className={`text-xs leading-relaxed text-gray-600 ${expanded ? "" : "line-clamp-3"}`}>
        {chunk.text}
      </p>
      {chunk.text.length > 200 && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-1.5 flex items-center gap-0.5 text-[11px] font-medium text-indigo-400 hover:text-indigo-600 transition-colors"
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
    <div className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-gray-100">
      <div className="h-0.5 w-full" style={{ background: "linear-gradient(90deg, #6366F1, #8B5CF6)" }} />
      <div className="p-5">
        <div className="mb-3 flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-indigo-50">
            <FileText size={13} className="text-indigo-500" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
            Text Evidence
          </span>
          <span className="ml-auto rounded-full bg-indigo-50 px-2 py-0.5 text-[10px] font-medium text-indigo-500">
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
