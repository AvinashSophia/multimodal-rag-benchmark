import { BarChart3 } from "lucide-react";

interface Props {
  metrics: Record<string, number>;
}

const METRIC_LABELS: Record<string, string> = {
  exact_match: "Exact Match",
  f1: "F1 Score",
  anls: "ANLS",
  faithfulness: "Faithfulness",
  attribution_accuracy: "Attribution",
  vqa_accuracy: "VQA Accuracy",
  cross_modal_consistency: "Cross-modal",
  "text_recall@1": "Text Recall@1",
  "text_recall@3": "Text Recall@3",
  "text_recall@5": "Text Recall@5",
  "text_recall@10": "Text Recall@10",
  text_mrr: "Text MRR",
  text_ndcg: "Text nDCG",
  "image_recall@1": "Image Recall@1",
  "image_recall@3": "Image Recall@3",
  "image_recall@5": "Image Recall@5",
  image_mrr: "Image MRR",
  image_ndcg: "Image nDCG",
};

const GROUPS: { label: string; keys: string[] }[] = [
  { label: "Answer Quality", keys: ["exact_match", "f1", "anls", "vqa_accuracy"] },
  { label: "Grounding", keys: ["faithfulness", "attribution_accuracy", "cross_modal_consistency"] },
  { label: "Text Retrieval", keys: ["text_recall@1", "text_recall@3", "text_recall@5", "text_recall@10", "text_mrr", "text_ndcg"] },
  { label: "Image Retrieval", keys: ["image_recall@1", "image_recall@3", "image_recall@5", "image_mrr", "image_ndcg"] },
];

function barColor(value: number) {
  if (value >= 0.8) return "#10B981";
  if (value >= 0.5) return "#EDCB69";
  return "#EF4444";
}

function MetricBar({ label, value }: { label: string; value: number }) {
  const color = barColor(value);
  const pct = Math.round(value * 100);
  return (
    <div>
      <div className="mb-1 flex items-center justify-between">
        <span className="text-xs font-medium" style={{ color: "rgba(255,255,255,0.5)" }}>{label}</span>
        <span className="text-xs font-bold" style={{ color }}>{pct}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "rgba(255,255,255,0.06)" }}>
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  );
}

export default function MetricsPanel({ metrics }: Props) {
  const available = new Set(Object.keys(metrics).filter((k) => metrics[k] !== undefined));
  if (available.size === 0) return null;

  const knownKeys = new Set(GROUPS.flatMap((g) => g.keys));
  const ungrouped = [...available].filter((k) => !knownKeys.has(k));

  const sections = [
    ...GROUPS.map((g) => ({ label: g.label, entries: g.keys.filter((k) => available.has(k)) })),
    ...(ungrouped.length > 0 ? [{ label: "Other", entries: ungrouped }] : []),
  ].filter((s) => s.entries.length > 0);

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
        <div className="mb-4 flex items-center gap-2">
          <div
            className="flex h-6 w-6 items-center justify-center rounded-md"
            style={{ background: "rgba(237,203,105,0.1)" }}
          >
            <BarChart3 size={13} style={{ color: "#EDCB69" }} />
          </div>
          <span className="text-xs font-semibold uppercase tracking-widest" style={{ color: "rgba(255,255,255,0.4)" }}>
            Evaluation
          </span>
        </div>

        <div className="flex flex-col gap-5">
          {sections.map((section) => (
            <div key={section.label}>
              <p className="mb-2 text-[10px] font-semibold uppercase tracking-widest" style={{ color: "rgba(237,203,105,0.4)" }}>
                {section.label}
              </p>
              <div className="flex flex-col gap-3">
                {section.entries.map((key) => (
                  <MetricBar key={key} label={METRIC_LABELS[key] ?? key} value={metrics[key]} />
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
