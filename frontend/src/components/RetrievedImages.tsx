import { useState } from "react";
import { ImageIcon, X, ZoomIn, Flame } from "lucide-react";
import type { RetrievedImage } from "../types";
import { imageUrl, fetchHeatmap } from "../api/client";

interface Props {
  images: RetrievedImage[];
  dataset?: string;
  query?: string;
}

type ModalContent = { type: "image"; src: string } | { type: "heatmap"; src: string; pageId: string };

export default function RetrievedImages({ images, dataset = "altumint", query = "" }: Props) {
  const [modal, setModal] = useState<ModalContent | null>(null);
  const [loadingHeatmap, setLoadingHeatmap] = useState<string | null>(null);

  async function handleHeatmap(e: React.MouseEvent, img: RetrievedImage) {
    e.stopPropagation();
    if (!query) return;
    setLoadingHeatmap(img.page_id);
    try {
      const b64 = await fetchHeatmap(query, img.page_id);
      setModal({ type: "heatmap", src: `data:image/png;base64,${b64}`, pageId: img.page_id });
    } catch {
      // heatmap not available (non-colpali retriever) — silently ignore
    } finally {
      setLoadingHeatmap(null);
    }
  }

  if (images.length === 0) return null;

  return (
    <>
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
              <ImageIcon size={13} style={{ color: "#EDCB69" }} />
            </div>
            <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.4)" }}>
              Visual Evidence
            </span>
            <span
              className="ml-auto rounded-full px-2 py-0.5 text-[10px] font-medium"
              style={{ background: "rgba(237,203,105,0.08)", color: "rgba(237,203,105,0.7)" }}
            >
              {images.length} pages
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {images.map((img, i) => (
              <div
                key={i}
                className="group relative cursor-pointer overflow-hidden rounded-xl transition"
                style={{ border: "1px solid rgba(255,255,255,0.07)", background: "rgba(255,255,255,0.02)" }}
                onClick={() => setModal({ type: "image", src: imageUrl(img.page_id, dataset) })}
              >
                <img
                  src={imageUrl(img.page_id, dataset)}
                  alt={img.page_id}
                  className="h-40 w-full object-contain p-1"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />

                {/* Hover overlay */}
                <div
                  className="absolute inset-0 z-10 flex items-center justify-center gap-2 opacity-0 transition-all group-hover:opacity-100"
                  style={{ background: "rgba(9,9,15,0.5)" }}
                >
                  <ZoomIn size={16} style={{ color: "#EDCB69" }} />
                  {query && (
                    <button
                      onClick={(e) => handleHeatmap(e, img)}
                      title="ColPali attention heatmap"
                      className="flex items-center gap-1 rounded-lg px-2 py-1 text-[10px] font-semibold transition"
                      style={{ background: "rgba(237,203,105,0.15)", border: "1px solid rgba(237,203,105,0.4)", color: "#EDCB69" }}
                    >
                      {loadingHeatmap === img.page_id ? (
                        <span className="animate-pulse">…</span>
                      ) : (
                        <><Flame size={11} /> Heatmap</>
                      )}
                    </button>
                  )}
                </div>

                {/* Footer */}
                <div
                  className="flex items-center justify-between gap-1 px-2 py-1.5"
                  style={{ borderTop: "1px solid rgba(255,255,255,0.06)", background: "rgba(0,0,0,0.3)" }}
                >
                  <span
                    className="min-w-0 flex-1 truncate text-[10px] font-semibold"
                    style={{ color: "rgba(237,203,105,0.7)" }}
                  >
                    {img.page_id}
                  </span>
                  <span
                    className="shrink-0 rounded-full px-1.5 py-0.5 text-[9px] font-mono"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)", color: "rgba(255,255,255,0.3)" }}
                  >
                    {img.score.toFixed(3)}
                  </span>
                </div>

                {/* Full ID slide-up on hover */}
                <div
                  className="pointer-events-none absolute bottom-0 left-0 right-0 translate-y-full opacity-0 transition-all duration-200 group-hover:translate-y-0 group-hover:opacity-100"
                  style={{ background: "rgba(9,9,15,0.96)", borderTop: "1px solid rgba(237,203,105,0.2)", padding: "8px 10px" }}
                >
                  <p className="break-all text-[10px] font-medium leading-tight" style={{ color: "#EDCB69" }}>
                    {img.page_id}
                  </p>
                  <p className="mt-0.5 text-[9px] font-mono" style={{ color: "rgba(255,255,255,0.3)" }}>
                    score: {img.score.toFixed(4)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Modal — image lightbox or heatmap */}
      {modal && (
        <div
          className="fixed inset-0 z-50 flex flex-col items-center justify-center p-4"
          style={{ background: "rgba(0,0,0,0.92)", backdropFilter: "blur(8px)" }}
          onClick={() => setModal(null)}
        >
          <button
            className="absolute right-5 top-5 flex h-9 w-9 items-center justify-center rounded-full transition"
            style={{ background: "rgba(237,203,105,0.1)", border: "1px solid rgba(237,203,105,0.2)", color: "#EDCB69" }}
            onClick={() => setModal(null)}
          >
            <X size={18} />
          </button>

          {modal.type === "heatmap" && (
            <div className="mb-3 flex items-center gap-2">
              <Flame size={14} style={{ color: "#EDCB69" }} />
              <span className="text-xs font-semibold" style={{ color: "#EDCB69" }}>
                ColPali Attention Heatmap
              </span>
              <span className="text-[10px]" style={{ color: "rgba(255,255,255,0.3)" }}>
                — {modal.pageId}
              </span>
            </div>
          )}

          <img
            src={modal.src}
            alt={modal.type === "heatmap" ? "Heatmap" : "Page preview"}
            className="max-h-[85vh] max-w-[90vw] rounded-2xl shadow-2xl"
            style={{ border: "1px solid rgba(237,203,105,0.2)" }}
            onClick={(e) => e.stopPropagation()}
          />

          {modal.type === "heatmap" && (
            <p className="mt-3 text-center text-[11px]" style={{ color: "rgba(255,255,255,0.3)" }}>
              Warmer regions = more relevant to your query · ColPali MaxSim aggregated across all query tokens
            </p>
          )}
        </div>
      )}
    </>
  );
}
