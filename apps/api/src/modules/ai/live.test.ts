import { fetchRequestHandler } from "@trpc/server/adapters/fetch";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import type { Context } from "../../context";
import { chatResult, reviewTextResult } from "./model";

describe.skipIf(process.env.CONFERE_AI_LIVE_TEST !== "1")(
  "real local inference through the HTTP adapter",
  () => {
    it("requires a session and preserves coverage, Unicode quotes and review flags", async () => {
      expect(process.env.AI_CHECKPOINT).toBeTruthy();
      const { aiRouter } = await import("./index");
      const now = new Date();
      // Sessão sintética: verifica middleware e transporte, sem acessar contas ou banco.
      const context: Context = {
        auth: null,
        session: {
          user: {
            id: "local-inference-test",
            name: "Local test",
            email: "test@example.invalid",
            emailVerified: true,
            image: null,
            createdAt: now,
            updatedAt: now,
          },
          session: {
            id: "local-inference-session",
            token: "local-inference-token",
            userId: "local-inference-test",
            expiresAt: new Date(now.getTime() + 60000),
            createdAt: now,
            updatedAt: now,
            ipAddress: null,
            userAgent: null,
          },
        },
      };
      const invoke = (text: string, ctx: Context) =>
        fetchRequestHandler({
          endpoint: "/trpc",
          router: aiRouter,
          req: new Request("http://localhost/trpc/reviewText", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              text,
              requirements: ["O projeto deve incluir testes."],
            }),
          }),
          createContext: () => ctx,
        });
      expect(
        (await invoke("Texto", { auth: null, session: null })).status,
      ).toBe(401);
      for (const [text, status, conflict] of [
        ["O projeto inclui testes.", "atendido", false],
        ["O projeto não inclui testes.", "nao_atendido", false],
        [
          "🙂 O projeto inclui testes. O projeto não inclui testes.",
          "evidencia_insuficiente",
          true,
        ],
      ] as const) {
        const response = await invoke(text, context);
        expect(response.status).toBe(200);
        const raw = await response.json();
        const body = z
          .object({ result: z.object({ data: reviewTextResult }) })
          .parse(raw);
        const result = body.result.data;
        expect(result.requirements).toHaveLength(1);
        const row = result.requirements[0];
        if (!row) throw new Error("Missing requirement result");
        expect(row.status).toBe(status);
        expect(row.conflicting_evidence).toBe(conflict);
        expect(row.model_outputs_valid).toBe(true);
        expect(row.evidence).toHaveLength(conflict ? 2 : 1);
        expect(row.coverage.processed_characters).toBe(Array.from(text).length);
        expect(row.coverage.text_characters).toBe(Array.from(text).length);
        for (const evidence of row.evidence) {
          expect(evidence.quote).toBe(
            Array.from(text)
              .slice(evidence.char_start, evidence.char_end)
              .join(""),
          );
        }
        if (row.conflicting_evidence || !row.model_outputs_valid) {
          expect(row.status).toBe("evidencia_insuficiente");
        }
        expect(JSON.stringify(raw)).not.toContain('"checkpoint":');
      }
      const invokeChat = (
        ctx: Context,
        method = "POST",
        history: { message: string; response: string }[] = [],
      ) =>
        fetchRequestHandler({
          endpoint: "/trpc",
          router: aiRouter,
          req: new Request("http://localhost/trpc/chat", {
            method,
            headers: { "content-type": "application/json" },
            ...(method === "POST"
              ? { body: JSON.stringify({ message: "Olá", history }) }
              : {}),
          }),
          createContext: () => ctx,
        });
      expect((await invokeChat({ auth: null, session: null })).status).toBe(
        401,
      );
      expect((await invokeChat(context, "GET")).status).toBe(405);
      const first = await invokeChat(context);
      expect(first.status).toBe(200);
      const firstRaw = await first.json();
      const result = chatResult.parse(firstRaw.result.data);
      expect(result.response.trim().length).toBeGreaterThan(0);
      expect(result.dropped_turns).toBe(0);
      expect(JSON.stringify(firstRaw)).not.toContain('"checkpoint":');
      const next = await invokeChat(context, "POST", [
        { message: "Olá", response: result.response },
      ]);
      expect(next.status).toBe(200);
      chatResult.parse((await next.json()).result.data);
    }, 120000);
  },
);
