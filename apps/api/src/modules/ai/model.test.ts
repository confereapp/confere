import { describe, expect, it } from "vitest";
import { reviewTextResult } from "./model";

describe("review inference metadata", () => {
  const common = {
    sha256: "a".repeat(64),
    step: 7,
    device: "cpu",
    context_size: 1024,
  };

  it("preserves decoder metadata and distinguishes classifier reservation", () => {
    expect(
      reviewTextResult.shape.model.parse({ ...common, response_tokens: 256 }),
    ).toEqual({ ...common, response_tokens: 256 });
    const classifier = {
      ...common,
      inference_kind: "unit_classifier",
      fragment_token_reserve: 256,
    };
    expect(reviewTextResult.shape.model.parse(classifier)).toEqual(classifier);
    expect(() => reviewTextResult.shape.model.parse(common)).toThrow();
    expect(() =>
      reviewTextResult.shape.model.parse({
        ...classifier,
        inference_kind: "unknown",
      }),
    ).toThrow();
  });
});
