import type { ProgressStep } from "./progress-types";

const LOW_VALUE_LABELS = new Set([
  ["Loaded", "context"].join(" "),
  ["Framed", "request"].join(" "),
  ["Updated", "plan"].join(" "),
  ["Created", "initial", "plan"].join(" "),
  ["Recorded", "observation"].join(" "),
  ["Chose", "next", "action"].join(" "),
  ["Inspect", "repository"].join(" "),
  ["Inspect", "repository", "tree"].join(" "),
]);

export function isLowValuePrimaryLabel(label?: string | null): boolean {
  return LOW_VALUE_LABELS.has(String(label || "").trim());
}

export function progressStepSummary(step: ProgressStep): string {
  return firstText(
    step.safeReasoningSummary,
    step.safetyNote,
    step.observation,
    step.currentAction,
    step.nextAction,
    step.label,
    step.message,
  ) || "Working";
}

export function hasTechnicalLogSteps(steps: ProgressStep[]): boolean {
  return steps.some((step) => {
    if (step.display === "hidden" || step.visibility === "internal") return true;
    if (step.display === "secondary" || step.visibility === "debug") return true;
    if (step.eventType === "action_result") return true;
    return isLowValuePrimaryLabel(step.label) && !step.safeReasoningSummary && !step.safetyNote;
  });
}

function firstText(...values: Array<string | null | undefined>): string | undefined {
  for (const value of values) {
    const text = String(value || "").trim();
    if (text) return text;
  }
  return undefined;
}
