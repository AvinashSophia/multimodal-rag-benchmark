import { FileSearch, ImageIcon, Cpu, ChevronDown } from "lucide-react";
import type { ConfigOptions } from "../types";

interface SelectedConfig {
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
      <div className="flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-widest text-indigo-300">
        {icon}
        {label}
      </div>
      <div className="relative">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none rounded-lg border border-white/10 bg-white/5 py-2 pl-3 pr-8 text-sm font-medium text-white/90 transition focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-400/50 hover:border-white/20 cursor-pointer"
        >
          {(choices ?? []).map((c) => (
            <option key={c} value={c} className="bg-gray-900 text-white">
              {friendlyLabel(c)}
            </option>
          ))}
        </select>
        <ChevronDown size={12} className="pointer-events-none absolute right-2.5 top-1/2 -translate-y-1/2 text-white/40" />
      </div>
    </div>
  );
}

export default function ConfigSelector({ options, selected, onChange }: Props) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
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
