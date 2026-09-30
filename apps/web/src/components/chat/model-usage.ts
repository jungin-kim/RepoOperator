import type { ModelUsagePayload } from "../../lib/local-worker-client";

function compactTokens(value: number): string {
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return String(value);
}

/** One-line token summary for the trust-trace card, or null when nothing was recorded. */
export function formatModelUsage(usage: ModelUsagePayload | null | undefined): string | null {
  if (!usage || !usage.calls) return null;
  const parts = [`${usage.calls} call${usage.calls === 1 ? "" : "s"}`];
  if (usage.input_tokens) {
    const ratio = Math.round((usage.cache_hit_ratio ?? 0) * 100);
    parts.push(`in ${compactTokens(usage.input_tokens)} (${ratio}% cached)`);
  }
  if (usage.output_tokens) parts.push(`out ${compactTokens(usage.output_tokens)}`);
  if (usage.gate_feedback_retries) {
    parts.push(`${usage.gate_feedback_retries} policy retr${usage.gate_feedback_retries === 1 ? "y" : "ies"}`);
  }
  return parts.join(" · ");
}
