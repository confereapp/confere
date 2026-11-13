import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { env } from "@confere/env/server";
import { TRPCError } from "@trpc/server";
import {
  chatInput,
  chatResult,
  reviewTextInput,
  reviewTextResult,
} from "./model";

const root = fileURLToPath(new URL("../../../../../", import.meta.url));
let running = false;

export async function reviewText(input: typeof reviewTextInput._output) {
  const request = reviewTextInput.parse(input);
  return runInference("confere_ai.text_review", request, (value) =>
    reviewTextResult.parse(value),
  );
}

export async function chat(input: typeof chatInput._output) {
  const request = chatInput.parse(input);
  return runInference("confere_ai.chat_api", request, (value) => {
    if (
      value &&
      typeof value === "object" &&
      "error" in value &&
      value.error === "context_length"
    ) {
      throw new TRPCError({
        code: "BAD_REQUEST",
        message:
          "Mensagem longa demais para o contexto do modelo. Encurte o texto.",
      });
    }
    return chatResult.parse(value);
  });
}

async function runInference<T>(
  module: string,
  request: unknown,
  parse: (value: unknown) => T,
): Promise<T> {
  const checkpoint = env.AI_CHECKPOINT;
  if (!checkpoint) {
    throw new TRPCError({
      code: "PRECONDITION_FAILED",
      message: "Inferência local não configurada.",
    });
  }
  if (running) {
    throw new TRPCError({
      code: "TOO_MANY_REQUESTS",
      message: "Uma inferência está em andamento. Tente novamente ao concluir.",
    });
  }
  running = true;
  try {
    const output = await new Promise<string>((accept, reject) => {
      const child = execFile(
        env.AI_PYTHON ?? resolve(root, "ai/.venv/bin/python"),
        [
          "-m",
          module,
          "--checkpoint",
          resolve(root, checkpoint),
          "--device",
          env.AI_DEVICE,
        ],
        {
          cwd: root,
          timeout: 120000,
          maxBuffer: 4 * 1024 * 1024,
          killSignal: "SIGKILL",
        },
        (error, stdout) => (error ? reject(error) : accept(stdout)),
      );
      child.stdin?.on("error", () => child.kill("SIGKILL"));
      child.stdin?.end(JSON.stringify(request));
    });
    return parse(JSON.parse(output));
  } catch (error) {
    if (error instanceof TRPCError) throw error;
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message:
        "A inferência local não foi concluída. Nenhum resultado foi salvo.",
    });
  } finally {
    running = false;
  }
}
