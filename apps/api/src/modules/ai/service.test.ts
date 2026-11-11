import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  env: {
    AI_CHECKPOINT: "ai/runs/candidate/best.pt" as string | undefined,
    AI_DEVICE: "cpu",
    AI_PYTHON: undefined,
  },
  execFile: vi.fn(),
}));
vi.mock("@confere/env/server", () => ({ env: mocks.env }));
vi.mock("node:child_process", () => ({ execFile: mocks.execFile }));

import { aiRouter } from "./index";
import { reviewText } from "./service";

const input = {
  text: "O projeto inclui testes.",
  requirements: ["Incluir testes."],
};

describe("local AI boundary", () => {
  beforeEach(() => {
    mocks.env.AI_CHECKPOINT = "ai/runs/candidate/best.pt";
    mocks.execFile.mockReset();
  });

  it("requires authentication before inference", async () => {
    const caller = aiRouter.createCaller({ auth: null, session: null });
    await expect(caller.reviewText(input)).rejects.toMatchObject({
      code: "UNAUTHORIZED",
    });
    expect(mocks.execFile).not.toHaveBeenCalled();
  });

  it("requires explicit server checkpoint configuration", async () => {
    mocks.env.AI_CHECKPOINT = undefined;
    await expect(reviewText(input)).rejects.toMatchObject({
      code: "PRECONDITION_FAILED",
    });
    expect(mocks.execFile).not.toHaveBeenCalled();
  });

  it("rejects empty or oversized input before launching Python", async () => {
    await expect(reviewText({ ...input, text: " " })).rejects.toThrow();
    await expect(
      reviewText({ ...input, text: "a".repeat(8001) }),
    ).rejects.toThrow();
    expect(mocks.execFile).not.toHaveBeenCalled();
  });

  it("rejects concurrent work, hides process errors and releases the slot", async () => {
    let finish!: (error: Error | null, output: string) => void;
    const end = vi.fn();
    mocks.execFile.mockImplementation((_file, _args, options, callback) => {
      expect(options.timeout).toBe(120000);
      expect(options.killSignal).toBe("SIGKILL");
      finish = callback;
      return { stdin: { on: vi.fn(), end } };
    });
    const pending = reviewText(input);
    await expect(reviewText(input)).rejects.toMatchObject({
      code: "TOO_MANY_REQUESTS",
    });
    expect(end).toHaveBeenCalledWith(JSON.stringify(input));
    const failed = expect(pending).rejects.toMatchObject({
      code: "INTERNAL_SERVER_ERROR",
    });
    finish(new Error("private checkpoint path"), "");
    await failed;
    const retry = reviewText(input);
    const invalid = expect(retry).rejects.toMatchObject({
      code: "INTERNAL_SERVER_ERROR",
    });
    finish(null, '{"experimental":false}');
    await invalid;
    expect(mocks.execFile).toHaveBeenCalledTimes(2);
  });
});
