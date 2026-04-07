import { useState } from "react";
import { ImageIcon, X, ZoomIn } from "lucide-react";
import type { RetrievedImage } from "../types";
import { imageUrl } from "../api/client";

interface Props {
  images: RetrievedImage[];
}

export default function RetrievedImages({ images }: Props) {
  const [lightbox, setLightbox] = useState<string | null>(null);

  if (images.length === 0) return null;

  return (
    <>
      <div className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-gray-100">
        <div className="h-0.5 w-full" style={{ background: "linear-gradient(90deg, #7C3AED, #EC4899)" }} />
        <div className="p-5">
          <div className="mb-3 flex items-center gap-2">
            <div className="flex h-6 w-6 items-center justify-center rounded-md bg-violet-50">
              <ImageIcon size={13} className="text-violet-500" />
            </div>
            <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
              Visual Evidence
            </span>
            <span className="ml-auto rounded-full bg-violet-50 px-2 py-0.5 text-[10px] font-medium text-violet-500">
              {images.length} pages
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {images.map((img, i) => (
              <div
                key={i}
                className="group relative cursor-pointer overflow-hidden rounded-xl border border-gray-100 bg-gray-50 transition hover:border-violet-300 hover:shadow-md"
                onClick={() => setLightbox(imageUrl(img.page_id))}
              >
                <img
                  src={imageUrl(img.page_id)}
                  alt={img.page_id}
                  className="h-28 w-full object-contain p-1"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />
                {/* Hover overlay */}
                <div className="absolute inset-0 flex items-center justify-center bg-violet-600/0 opacity-0 transition-all group-hover:bg-violet-600/10 group-hover:opacity-100">
                  <ZoomIn size={18} className="text-violet-700" />
                </div>
                {/* Footer */}
                <div className="border-t border-gray-100 bg-white px-2 py-1.5 flex items-center justify-between gap-1">
                  <span className="min-w-0 flex-1 truncate text-[10px] font-semibold text-violet-600" title={img.page_id}>
                    {img.page_id}
                  </span>
                  <span className="shrink-0 rounded-full bg-gray-50 px-1.5 py-0.5 text-[9px] font-mono text-gray-400 ring-1 ring-gray-100">
                    {img.score.toFixed(3)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4"
          style={{ background: "rgba(0,0,0,0.85)", backdropFilter: "blur(4px)" }}
          onClick={() => setLightbox(null)}
        >
          <button
            className="absolute right-5 top-5 flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white transition hover:bg-white/20"
            onClick={() => setLightbox(null)}
          >
            <X size={18} />
          </button>
          <img
            src={lightbox}
            alt="Page preview"
            className="max-h-[90vh] max-w-[90vw] rounded-2xl shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          />
        </div>
      )}
    </>
  );
}
