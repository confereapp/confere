import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => ({
  env: {
    AI_CHECKPOINT: "ai/runs/candidate/best.pt" as string | undefined,
    AI_CHAT_CHECKPOINT: undefined as string | undefined,
    AI_DEVICE: "cpu",
    AI_PYTHON: undefined,
  },
  execFile: vi.fn(),
}));
vi.mock("@confere/env/server", () => ({ env: mocks.env }));
vi.mock("node:child_process", () => ({ execFile: mocks.execFile }));

import { aiRouter } from "./index";
import { chat, reviewText } from "./service";

const input = {
  text: "O projeto inclui testes.",
  requirements: ["Incluir testes."],
};

describe("local AI boundary", () => {
  beforeEach(() => {
    mocks.env.AI_CHECKPOINT = "ai/runs/candidate/best.pt";
    mocks.env.AI_CHAT_CHECKPOINT = undefined;
    mocks.execFile.mockReset();
  });

  it("requires authentication before inference", async () => {
    const caller = aiRouter.createCaller({ auth: null, session: null });
    await expect(caller.reviewText(input)).rejects.toMatchObject({
      code: "UNAUTHORIZED",
    });
    await expect(caller.chat({ message: "oi" })).rejects.toMatchObject({
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
      expect(options.env.PYTHONPATH).toMatch(/ai\/src$/);
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

describe("local chat boundary", () => {
  beforeEach(() => {
    mocks.env.AI_CHECKPOINT = "ai/runs/candidate/best.pt";
    mocks.env.AI_CHAT_CHECKPOINT = undefined;
    mocks.execFile.mockReset();
  });

  it("validates messages, history and server configuration", async () => {
    for (const input of [
      { message: " " },
      { message: "x".repeat(2001) },
      {
        message: "oi",
        history: Array(9).fill({ message: "oi", response: "sim" }),
      },
    ]) {
      await expect(chat({ history: [], ...input })).rejects.toThrow();
    }
    mocks.env.AI_CHECKPOINT = undefined;
    await expect(chat({ message: "oi", history: [] })).rejects.toMatchObject({
      code: "PRECONDITION_FAILED",
    });
    expect(mocks.execFile).not.toHaveBeenCalled();
  });

  it("uses the chat checkpoint independently and falls back to the shared checkpoint", async () => {
    mocks.execFile.mockImplementation((_file, _args, _options, callback) => {
      callback(
        null,
        JSON.stringify({
          response: "oi",
          experimental: true,
          dropped_turns: 0,
        }),
      );
      return { stdin: { on: vi.fn(), end: vi.fn() } };
    });
    await chat({ message: "oi", history: [] });
    expect(mocks.execFile.mock.calls[0][1][3]).toMatch(
      /ai\/runs\/candidate\/best.pt$/,
    );
    mocks.env.AI_CHAT_CHECKPOINT = "ai/runs/conversation/best.pt";
    mocks.env.AI_CHECKPOINT = undefined;
    await chat({ message: "oi", history: [] });
    expect(mocks.execFile.mock.calls[1][1][3]).toMatch(
      /ai\/runs\/conversation\/best.pt$/,
    );
    await expect(reviewText(input)).rejects.toMatchObject({
      code: "PRECONDITION_FAILED",
    });
    expect(mocks.execFile).toHaveBeenCalledTimes(2);
  });

  it("shares the inference slot and returns only validated public data", async () => {
    let finish!: (error: Error | null, output: string) => void;
    const end = vi.fn();
    mocks.execFile.mockImplementation((_file, args, _options, callback) => {
      expect(args).toContain("confere_ai.chat_api");
      finish = callback;
      return { stdin: { on: vi.fn(), end } };
    });
    const input = {
      message: "olá 🙂",
      history: [{ message: "oi", response: "olá" }],
    };
    const pending = chat(input);
    await expect(
      reviewText({ text: "teste", requirements: ["teste"] }),
    ).rejects.toMatchObject({ code: "TOO_MANY_REQUESTS" });
    expect(end).toHaveBeenCalledWith(JSON.stringify(input));
    finish(
      null,
      JSON.stringify({
        response: "olá",
        experimental: true,
        dropped_turns: 1,
        checkpoint: "private",
      }),
    );
    await expect(pending).resolves.toEqual({
      response: "olá",
      experimental: true,
      dropped_turns: 1,
    });
    const oversized = chat(input);
    const rejected = expect(oversized).rejects.toMatchObject({
      code: "BAD_REQUEST",
    });
    finish(null, '{"error":"context_length"}');
    await rejected;
    const invalid = chat(input);
    const invalidResult = expect(invalid).rejects.toMatchObject({
      code: "INTERNAL_SERVER_ERROR",
    });
    finish(null, '{"response":42}');
    await invalidResult;
  });
});
