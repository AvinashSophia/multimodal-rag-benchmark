import { Loader } from "lucide-react";
import type { HealthResponse } from "../types";

interface Props {
  health: HealthResponse | undefined;
  loading: boolean;
}

export default function StatusBar({ health, loading }: Props) {
  if (!health) {
    return (
      <div className="flex items-center gap-1.5 rounded-full border border-yellow-400/30 bg-yellow-400/10 px-3 py-1.5 text-xs text-yellow-300">
        <Loader size={10} className="animate-spin" />
        Connecting…
      </div>
    );
  }

  if (!health.initialized) {
    return (
      <div className="flex items-center gap-1.5 rounded-full border border-yellow-400/30 bg-yellow-400/10 px-3 py-1.5 text-xs text-yellow-300">
        <Loader size={10} className="animate-spin" />
        Initializing…
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      {/* Live indicator */}
      <span className="relative flex h-2 w-2">
        {loading ? (
          <span className="h-2 w-2 rounded-full bg-indigo-400" />
        ) : (
          <>
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-60" />
            <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
          </>
        )}
      </span>
      <span className="hidden text-xs text-white/50 sm:block">
        {health.dataset}
      </span>
      <span className="hidden text-white/20 sm:block">·</span>
      <span className="hidden text-xs text-white/50 sm:block">{health.model}</span>
    </div>
  );
}
