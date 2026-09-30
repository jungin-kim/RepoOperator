import { describe, expect, it } from "vitest";
import { formatModelUsage } from "../model-usage";

describe("formatModelUsage", () => {
  it("summarizes calls, cache hit ratio and output", () => {
    expect(
      formatModelUsage({ calls: 7, input_tokens: 48210, cached_input_tokens: 42400, cache_hit_ratio: 0.879, output_tokens: 1320 }),
    ).toBe("7 calls · in 48.2k (88% cached) · out 1.3k");
  });

  it("mentions policy feedback retries", () => {
    expect(formatModelUsage({ calls: 1, gate_feedback_retries: 1 })).toBe("1 call · 1 policy retry");
  });

  it("hides the row when nothing was recorded", () => {
    expect(formatModelUsage(null)).toBeNull();
    expect(formatModelUsage({})).toBeNull();
  });
});
