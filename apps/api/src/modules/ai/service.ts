import { execFile } from "node:child_process";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { env } from "@confere/env/server";
import { TRPCError } from "@trpc/server";
import { reviewTextInput, reviewTextResult } from "./model";

const root = fileURLToPath(new URL("../../../../../", import.meta.url));
let running = false;

export async function reviewText(input: typeof reviewTextInput._output) {
  const request = reviewTextInput.parse(input);
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
      message: "Uma análise está em andamento. Tente novamente ao concluir.",
    });
  }
  running = true;
  try {
    const output = await new Promise<string>((accept, reject) => {
      const child = execFile(
        env.AI_PYTHON ?? resolve(root, "ai/.venv/bin/python"),
        [
          "-m",
          "confere_ai.text_review",
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
    return reviewTextResult.parse(JSON.parse(output));
  } catch {
    throw new TRPCError({
      code: "INTERNAL_SERVER_ERROR",
      message: "A análise local não foi concluída. Nenhum resultado foi salvo.",
    });
  } finally {
    running = false;
  }
}
