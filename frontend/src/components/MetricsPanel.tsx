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
};

function barColor(value: number) {
  if (value >= 0.8) return { bar: "#10B981", bg: "#ECFDF5", text: "#065F46" };
  if (value >= 0.5) return { bar: "#F59E0B", bg: "#FFFBEB", text: "#78350F" };
  return { bar: "#EF4444", bg: "#FEF2F2", text: "#7F1D1D" };
}

export default function MetricsPanel({ metrics }: Props) {
  const entries = Object.entries(metrics).filter(([, v]) => v !== undefined);
  if (entries.length === 0) return null;

  return (
    <div className="overflow-hidden rounded-2xl bg-white shadow-sm ring-1 ring-gray-100">
      <div className="h-0.5 w-full" style={{ background: "linear-gradient(90deg, #10B981, #06B6D4)" }} />
      <div className="p-5">
        <div className="mb-4 flex items-center gap-2">
          <div className="flex h-6 w-6 items-center justify-center rounded-md bg-emerald-50">
            <BarChart3 size={13} className="text-emerald-500" />
          </div>
          <span className="text-xs font-semibold uppercase tracking-widest text-gray-500">
            Evaluation
          </span>
        </div>

        <div className="flex flex-col gap-3">
          {entries.map(([key, value]) => {
            const c = barColor(value);
            const pct = Math.round(value * 100);
            return (
              <div key={key}>
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-xs font-medium text-gray-600">
                    {METRIC_LABELS[key] ?? key}
                  </span>
                  <span className="text-xs font-bold" style={{ color: c.text }}>
                    {pct}%
                  </span>
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-gray-100">
                  <div
                    className="h-full rounded-full transition-all duration-700"
                    style={{ width: `${pct}%`, background: c.bar }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
