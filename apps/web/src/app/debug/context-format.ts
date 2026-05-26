export type ContextBudgetUsage = {
  estimated_input_tokens?: number;
  estimated_output_reserve?: number;
  context_window?: number;
  estimated_total_tokens?: number;
  usage_ratio?: number;
};

export type TargetCandidateDebug = {
  path?: string | null;
  score?: number | string | null;
  role?: string | null;
  sources?: string[];
  prior_reused?: boolean;
};

export function formatBudgetUsage(budget?: ContextBudgetUsage | null): string {
  if (!budget) return "-";
  const used = budget.estimated_total_tokens ?? (
    typeof budget.estimated_input_tokens === "number" || typeof budget.estimated_output_reserve === "number"
      ? Number(budget.estimated_input_tokens ?? 0) + Number(budget.estimated_output_reserve ?? 0)
      : undefined
  );
  const window = budget.context_window;
  const ratio = typeof budget.usage_ratio === "number" ? ` (${Math.round(budget.usage_ratio * 100)}%)` : "";
  if (typeof used === "number" && typeof window === "number") return `${used} / ${window} tokens${ratio}`;
  if (typeof budget.estimated_input_tokens === "number") return `${budget.estimated_input_tokens} input tokens`;
  return "-";
}

export function formatTargetCandidates(candidates?: TargetCandidateDebug[] | null): string {
  if (!candidates?.length) return "-";
  return candidates
    .slice(0, 6)
    .map((candidate) => {
      const score = candidate.score != null ? ` ${candidate.score}` : "";
      const prior = candidate.prior_reused ? " prior" : "";
      const role = candidate.role ? ` ${candidate.role}` : "";
      return `${candidate.path ?? "unknown"}${score}${role}${prior}`.trim();
    })
    .join(", ");
}

export function formatCarryoverSummary(memory?: {
  carryover_summaries?: Array<{ kind?: string; selected_target_files?: string[]; candidate_count?: number }>;
  target_candidate_summaries?: TargetCandidateDebug[];
} | null): string {
  const carryover = memory?.carryover_summaries ?? [];
  if (carryover.length) {
    return carryover
      .slice(0, 4)
      .map((item) => `${item.kind ?? "carryover"}: ${(item.selected_target_files ?? []).join(", ") || `${item.candidate_count ?? 0} candidate(s)`}`)
      .join(" | ");
  }
  return formatTargetCandidates(memory?.target_candidate_summaries ?? []);
}
