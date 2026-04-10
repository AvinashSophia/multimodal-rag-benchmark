import { FileSearch, ImageIcon, Cpu, ChevronDown, Database } from "lucide-react";
import type { ConfigOptions } from "../types";

interface SelectedConfig {
  dataset: string;
  model: string;
  text_method: string;
  image_method: string;
}

interface Props {
  options: ConfigOptions;
  selected: SelectedConfig;
  onChange: (updated: Partial<SelectedConfig>) => void;
}

const LABELS: Record<string, string> = {
  altumint: "Altumint",
  docvqa: "DocVQA",
  hotpotqa: "HotpotQA",
  gqa: "GQA",
  bm25: "BM25",
  dense: "Dense",
  dense_qdrant: "Dense · Qdrant",
  bm25_elastic: "BM25 · Elastic",
  hybrid_elastic_qdrant: "Hybrid (Elastic + Qdrant)",
  clip: "CLIP",
  clip_qdrant: "CLIP · Qdrant",
  colpali_qdrant: "ColPali · Qdrant",
  gpt: "GPT-4o",
  gemini: "Gemini 2.5 Pro",
  gemini_vertex: "Gemini Vertex",
  qwen_vl: "Qwen-VL",
};

function friendlyLabel(key: string) {
  return LABELS[key] ?? key;
}

interface FieldProps {
  icon: React.ReactNode;
  label: string;
  value: string;
  choices: string[];
  onChange: (v: string) => void;
}

function Field({ icon, label, value, choices, onChange }: FieldProps) {
  return (
    <div className="flex flex-col gap-1.5">
      <div
        className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest"
        style={{ color: "rgba(237,203,105,0.6)" }}
      >
        {icon}
        {label}
      </div>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none rounded-lg py-2 pl-3 pr-8 text-sm font-medium transition cursor-pointer outline-none"
          style={{
            background: "rgba(255,255,255,0.05)",
            border: "1px solid rgba(237,203,105,0.2)",
            color: "rgba(255,255,255,0.85)",
          }}
        >
          {(choices ?? []).map((c) => (
            <option key={c} value={c} style={{ background: "#0A0A0F", color: "white" }}>
              {friendlyLabel(c)}
            </option>
          ))}
        </select>
        <ChevronDown size={12} className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2" style={{ color: "rgba(237,203,105,0.4)" }} />
      </div>
    </div>
  );
}

export default function ConfigSelector({ options, selected, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <Field
        icon={<Database size={10} />}
        label="Dataset"
        value={selected.dataset}
        choices={options.datasets}
        onChange={(v) => onChange({ dataset: v })}
      />
      <Field
        icon={<FileSearch size={10} />}
        label="Text Retriever"
        value={selected.text_method}
        choices={options.text_methods}
        onChange={(v) => onChange({ text_method: v })}
      />
      <Field
        icon={<ImageIcon size={10} />}
        label="Image Retriever"
        value={selected.image_method}
        choices={options.image_methods}
        onChange={(v) => onChange({ image_method: v })}
      />
      <Field
        icon={<Cpu size={10} />}
        label="Model"
        value={selected.model}
        choices={options.models}
        onChange={(v) => onChange({ model: v })}
      />
    </div>
  );
}
