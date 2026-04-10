import { useState, useRef } from "react";
import { ArrowRight, ChevronDown, ChevronUp, Loader, ImagePlus, X } from "lucide-react";
import type { QueryRequest } from "../types";
import { uploadQueryImage } from "../api/client";

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
  const [queryImagePath, setQueryImagePath] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function handleImageSelect(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);
    setImagePreview(URL.createObjectURL(file));
    setUploading(true);
    try {
      const path = await uploadQueryImage(file);
      setQueryImagePath(path);
    } catch {
      setUploadError("Image upload failed. Check that the server is running.");
      setImagePreview(null);
      setQueryImagePath(null);
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  function clearImage() {
    setImagePreview(null);
    setQueryImagePath(null);
    setUploadError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit({
      query: query.trim(),
      ground_truth: groundTruth.trim() || undefined,
      query_image_path: queryImagePath ?? undefined,
    });
  }

  const busy = loading || uploading;

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-3">
      {/* Main input row */}
      <div
        className="flex items-center gap-2 rounded-xl px-4 py-1 transition-all"
        style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(237,203,105,0.2)",
        }}
        onFocus={() => {}}
      >
        <input
          className="flex-1 bg-transparent py-2.5 text-[15px] outline-none disabled:opacity-50"
          style={{ color: "rgba(255,255,255,0.9)", caretColor: "#EDCB69" }}
          placeholder="Ask anything about FLVM…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={busy}
          autoFocus
        />

        {/* Image upload button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={busy}
          title="Attach a query image"
          className="flex shrink-0 items-center justify-center rounded-lg p-2 transition disabled:cursor-not-allowed disabled:opacity-40"
          style={queryImagePath
            ? { background: "rgba(237,203,105,0.15)", border: "1px solid rgba(237,203,105,0.4)", color: "#EDCB69" }
            : { background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)", color: "rgba(255,255,255,0.4)" }
          }
        >
          {uploading ? <Loader size={15} className="animate-spin" /> : <ImagePlus size={15} />}
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/gif"
          className="hidden"
          onChange={handleImageSelect}
        />

        <button
          type="submit"
          disabled={busy || !query.trim()}
          className="shrink-0 rounded-lg py-2 text-sm font-semibold transition disabled:cursor-not-allowed disabled:opacity-40"
          style={{ background: "linear-gradient(135deg, #EDCB69, #C9A83C)", color: "#09090F", width: "8rem" }}
        >
          <span style={{ display: loading ? "none" : "flex", alignItems: "center", justifyContent: "center", gap: "8px" }}>
            <ArrowRight size={14} />
            Ask
          </span>
          <span style={{ display: loading ? "flex" : "none", alignItems: "center", justifyContent: "center", gap: "8px" }}>
            <Loader size={14} className="animate-spin" />
            Running…
          </span>
        </button>
      </div>

      {/* Image preview */}
      {imagePreview && (
        <div
          className="flex items-center gap-3 rounded-xl px-3 py-2"
          style={{ background: "rgba(237,203,105,0.06)", border: "1px solid rgba(237,203,105,0.2)" }}
        >
          <img
            src={imagePreview}
            alt="Query image"
            className="h-14 w-14 rounded-lg object-cover"
            style={{ border: "1px solid rgba(237,203,105,0.3)" }}
          />
          <div className="min-w-0 flex-1">
            <p className="text-xs font-medium" style={{ color: "#EDCB69" }}>Query image attached</p>
            <p className="text-[11px]" style={{ color: "rgba(237,203,105,0.4)" }}>Will be used as visual context for retrieval</p>
          </div>
          <button
            type="button"
            onClick={clearImage}
            className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full transition"
            style={{ background: "rgba(237,203,105,0.1)", color: "rgba(237,203,105,0.5)" }}
          >
            <X size={12} />
          </button>
        </div>
      )}

      {uploadError && (
        <p className="text-xs" style={{ color: "#FCA5A5" }}>{uploadError}</p>
      )}

      {/* Suggestion chips */}
      {!query && (
        <div className="flex flex-wrap gap-2">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => setQuery(s)}
              className="rounded-full px-3 py-1 text-xs transition"
              style={{ background: "rgba(237,203,105,0.06)", border: "1px solid rgba(237,203,105,0.15)", color: "rgba(255,255,255,0.4)" }}
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
        className="flex items-center gap-1 self-start text-xs transition-colors"
        style={{ color: "rgba(255,255,255,0.3)" }}
      >
        {showGT ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        {showGT ? "Hide" : "Add ground truth"} for metric evaluation
      </button>

      {showGT && (
        <input
          className="rounded-xl px-4 py-2.5 text-sm outline-none transition-all"
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(237,203,105,0.15)",
            color: "rgba(255,255,255,0.8)",
          }}
          placeholder="Expected answer — enables EM / F1 / ANLS scoring"
          value={groundTruth}
          onChange={(e) => setGroundTruth(e.target.value)}
        />
      )}
    </form>
  );
}
