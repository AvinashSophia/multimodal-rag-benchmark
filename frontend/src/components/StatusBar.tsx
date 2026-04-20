import { Loader } from "lucide-react";
import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | undefined;
  loading: boolean;
}

export default function StatusBar({ health, loading }: Props) {
  if (!health) {
    return (
      <div
        className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs"
        style={{ border: "1px solid rgba(237,203,105,0.2)", background: "rgba(237,203,105,0.06)", color: "rgba(237,203,105,0.6)" }}
      >
        <Loader size={10} className="animate-spin" />
        Connecting…
      </div>
    );
  }

  if (!health.initialized) {
    return (
      <div
        className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs"
        style={{ border: "1px solid rgba(237,203,105,0.2)", background: "rgba(237,203,105,0.06)", color: "rgba(237,203,105,0.6)" }}
      >
        <Loader size={10} className="animate-spin" />
        Initializing…
      </div>
    );
  }

  if (!health.index_ready) {
    return (
      <div
        className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium"
        style={{ border: "1px solid rgba(248,113,113,0.4)", background: "rgba(248,113,113,0.1)", color: "#F87171" }}
        title="Index not built — run the batch job before querying"
      >
        <span>⚠</span>
        Index not built
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {/* Live indicator */}
      <span className="relative flex h-2 w-2">
        {loading ? (
          <span className="h-2 w-2 rounded-full" style={{ background: "#EDCB69" }} />
        ) : (
          <>
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ background: "#EDCB69" }} />
            <span className="relative inline-flex h-2 w-2 rounded-full" style={{ background: "#EDCB69" }} />
          </>
        )}
      </span>
      <span className="hidden text-xs sm:block" style={{ color: "rgba(255,255,255,0.4)" }}>
        {health.dataset}
      </span>
      <span className="hidden sm:block" style={{ color: "rgba(255,255,255,0.15)" }}>·</span>
      <span className="hidden text-xs sm:block" style={{ color: "rgba(255,255,255,0.4)" }}>
        {health.model}
      </span>
    </div>
  );
}
