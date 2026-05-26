import type { AgentRunPayload } from "@/lib/local-worker-client";

const REAL_VALIDATION_KINDS = new Set(["change_set", "post_apply", "command", "git"]);

export function renderableValidationResult(metadata: AgentRunPayload) {
  const result = metadata.validation_result;
  const kind = normalizeValidationKind(result?.kind || result?.source);
  if (result && kind) {
    return { ...result, kind, source: result.source || kind };
  }
  if (metadata.post_apply_validation_status && hasExplicitPostApplyContext(metadata)) {
    return {
      kind: "post_apply",
      source: "post_apply",
      status: normalizeValidationStatus(metadata.post_apply_validation_status),
    };
  }
  return null;
}

export function normalizeValidationKind(value?: string | null): string | null {
  const kind = String(value || "").trim().replace(/-/g, "_");
  if (kind === "post_apply_validation") return "post_apply";
  return REAL_VALIDATION_KINDS.has(kind) ? kind : null;
}

export function normalizeValidationStatus(value?: string | null): string {
  const status = String(value || "").trim().replace(/-/g, "_");
  if (["success", "succeeded", "valid", "pass"].includes(status)) return "passed";
  if (["invalid", "error", "failure", "fail"].includes(status)) return "failed";
  if (["not_run", "selected", "skipped_no_validation_command", "skipped_no_safe_command_selected"].includes(status)) return "skipped";
  if (["approval_denied", "waiting_approval", "cancelled", "timed_out"].includes(status)) return "blocked";
  return status || "skipped";
}

function hasExplicitPostApplyContext(metadata: AgentRunPayload): boolean {
  return Boolean(
    metadata.validation_command_selection
      || metadata.validation_commands?.length
      || normalizeValidationKind(metadata.validation_result?.kind || metadata.validation_result?.source) === "post_apply",
  );
}
