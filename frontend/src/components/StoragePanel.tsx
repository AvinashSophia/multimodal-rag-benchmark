import { useQuery } from "@tanstack/react-query";
import { Database } from "lucide-react";
import { fetchStorageOverview } from "../api/client";

interface Props {
  refreshKey?: number; // changes when a new query completes → triggers refetch
}

export default function StoragePanel({ refreshKey = 0 }: Props) {
  const { data } = useQuery({
    queryKey: ["storage", refreshKey],
    queryFn: fetchStorageOverview,
    staleTime: 0,
  });

  if (!data || data.collections.length === 0) return null;

  const maxMb = Math.max(...data.collections.map((c) => c.estimated_mb), 0.01);

  return (
    <div
      className="overflow-hidden rounded-2xl"
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(237,203,105,0.12)",
        backdropFilter: "blur(10px)",
      }}
    >
      <div className="h-px w-full" style={{ background: "linear-gradient(90deg, transparent, rgba(237,203,105,0.5), transparent)" }} />
      <div className="p-5">
        {/* Header */}
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md" style={{ background: "rgba(237,203,105,0.1)" }}>
            <Database size={13} style={{ color: "#EDCB69" }} />
          </div>
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.4)" }}>
            Index Storage
          </span>
          <span
            className="ml-auto rounded-full px-2 py-0.5 text-[10px] font-medium"
            style={{ background: "rgba(237,203,105,0.08)", color: "rgba(237,203,105,0.7)" }}
          >
            {data.collections.length} collections
          </span>
        </div>

        {/* Collection bars */}
        <div className="flex flex-col gap-3">
          {data.collections.map((col) => {
            const pct = maxMb > 0 ? (col.estimated_mb / maxMb) * 100 : 0;
            const isActive = col.active;
            const barColor = isActive ? "#EDCB69" : "rgba(255,255,255,0.15)";
            const label = col.name.replace(/_/g, " ");

            return (
              <div key={col.name}>
                {/* Name row */}
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className="min-w-0 flex-1 truncate text-[10px] font-semibold"
                    style={{ color: isActive ? "#EDCB69" : "rgba(255,255,255,0.5)" }}
                    title={col.name}
                  >
                    {label}
                  </span>
                  {isActive && (
                    <span
                      className="shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider"
                      style={{ background: "rgba(237,203,105,0.12)", color: "#EDCB69", border: "1px solid rgba(237,203,105,0.3)" }}
                    >
                      active
                    </span>
                  )}
                </div>

                {/* Bar */}
                <div className="mb-1 h-1.5 w-full overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.05)" }}>
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{ width: `${Math.max(pct, 1)}%`, background: barColor, opacity: isActive ? 1 : 0.6 }}
                  />
                </div>

                {/* Stats row */}
                <div className="flex items-center gap-3">
                  <Stat label="vectors" value={col.vectors.toLocaleString()} dim={!isActive} />
                  <Stat label="dim" value={col.dimension > 0 ? String(col.dimension) : "—"} dim={!isActive} />
                  <Stat label="est. size" value={col.estimated_mb > 0 ? `${col.estimated_mb} MB` : "< 0.01 MB"} dim={!isActive} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value, dim }: { label: string; value: string; dim: boolean }) {
  return (
    <div className="flex items-center gap-1">
      <span className="text-[9px] uppercase tracking-wider" style={{ color: "rgba(255,255,255,0.25)" }}>{label}</span>
      <span className="text-[10px] font-mono font-semibold" style={{ color: dim ? "rgba(255,255,255,0.4)" : "rgba(255,255,255,0.75)" }}>{value}</span>
    </div>
  );
}
