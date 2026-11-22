import { beforeEach, describe, expect, it, vi } from "vitest";

const mocks = vi.hoisted(() => {
  const updates: Record<string, unknown>[] = [];
  const inserted: Record<string, unknown>[][] = [];
  const update = () => ({
    set: (value: Record<string, unknown>) => {
      updates.push(value);
      return { where: async () => {} };
    },
  });
  const tx = {
    delete: () => ({ where: async () => {} }),
    insert: () => ({
      values: async (rows: Record<string, unknown>[]) => {
        inserted.push(rows);
      },
    }),
    update,
  };
  return {
    env: { AI_CHECKPOINT: "best.pt" as string | undefined },
    updates,
    inserted,
    submission: {} as Record<string, unknown> | undefined,
    runInference: vi.fn(),
    download: vi.fn(),
    db: {
      query: {
        submission: { findFirst: vi.fn() },
        requirement: { findMany: vi.fn() },
      },
      update,
      transaction: async (fn: (t: typeof tx) => Promise<void>) => fn(tx),
    },
  };
});
vi.mock("@confere/env/server", () => ({ env: mocks.env }));
vi.mock("@confere/db", () => ({
  db: mocks.db,
  and: vi.fn(),
  eq: vi.fn(),
  inArray: vi.fn(),
}));
vi.mock("@confere/db/schema/project", () => ({
  checkResult: {},
  requirement: {},
  submission: {},
}));
vi.mock("../project/service", () => ({
  getOptionalS3Bucket: async () => ({ download: mocks.download }),
}));
vi.mock("./service", () => ({ runInference: mocks.runInference }));

import {
  analyzeSubmission,
  NOT_CONFIGURED_MESSAGE,
  NOT_STORED_MESSAGE,
  toCheckResult,
} from "./analysis";

const evidence = (source_unit: string, model_status: string, quote = "x") => ({
  source_unit,
  quote,
  model_status: model_status as "atendido",
});

describe("toCheckResult", () => {
  it("maps model status, page and quote without ever claiming absence", () => {
    expect(
      toCheckResult({
        id: "r1",
        status: "atendido",
        evidence: [
          evidence("page:2", "nao_atendido"),
          evidence("page:5", "atendido", "Há testes."),
        ],
      }),
    ).toEqual({
      status: "atendido",
      feedback: "Trecho encontrado na p. 5.",
      quote: "Há testes.",
      pageNumber: 5,
    });
    expect(
      toCheckResult({
        id: "r2",
        status: "evidencia_insuficiente",
        evidence: [],
      }),
    ).toEqual({
      status: "revisar",
      feedback: "Sem evidência suficiente; revise manualmente.",
      quote: null,
      pageNumber: null,
    });
    const docx = toCheckResult({
      id: "r3",
      status: "nao_atendido",
      evidence: [evidence("paragraph:7", "nao_atendido", "a".repeat(900))],
    });
    expect(docx.status).toBe("revisar");
    expect(docx.pageNumber).toBeNull();
    expect(docx.feedback).toContain("paragraph:7");
    expect(docx.quote).toHaveLength(801);
  });
});

describe("analyzeSubmission", () => {
  beforeEach(() => {
    mocks.env.AI_CHECKPOINT = "best.pt";
    mocks.updates.length = 0;
    mocks.inserted.length = 0;
    mocks.runInference.mockReset();
    mocks.db.query.submission.findFirst.mockResolvedValue({
      id: "s1",
      projectId: "p1",
      status: "aguardando_analise",
      fileUrl: "submissions/p1/1-tcc.pdf",
    });
    mocks.db.query.requirement.findMany.mockResolvedValue([
      { id: "r1", title: "Testes", description: null },
      { id: "r2", title: "Diagrama", description: "UML" },
    ]);
    mocks.download.mockResolvedValue({
      Body: { transformToByteArray: async () => new Uint8Array([1]) },
    });
  });

  it("writes one check_result per requirement and the coverage", async () => {
    mocks.runInference.mockImplementation(async (module, request, parse) => {
      expect(module).toBe("confere_ai.submission_review");
      expect(request.path).toMatch(/entrega\.pdf$/);
      expect(request.requirements).toEqual([
        { id: "r1", title: "Testes", description: null },
        { id: "r2", title: "Diagrama", description: "UML" },
      ]);
      return parse({
        requirements: [
          {
            id: "r1",
            status: "atendido",
            evidence: [evidence("page:3", "atendido", "Há testes.")],
          },
          { id: "r2", status: "evidencia_insuficiente", evidence: [] },
        ],
        warnings: [],
        model: { sha256: "a", step: 2000 },
      });
    });
    await analyzeSubmission("s1");
    expect(mocks.inserted[0]).toMatchObject([
      {
        submissionId: "s1",
        requirementId: "r1",
        status: "atendido",
        pageNumber: 3,
        quote: "Há testes.",
      },
      { submissionId: "s1", requirementId: "r2", status: "revisar" },
    ]);
    expect(mocks.updates).toEqual([
      { status: "processando", analysisMessage: null },
      { status: "concluido", coverage: 50, analysisMessage: null },
    ]);
  });

  it("fails local uploads and waits when the checkpoint is missing", async () => {
    mocks.db.query.submission.findFirst.mockResolvedValueOnce({
      id: "s1",
      status: "aguardando_analise",
      fileUrl: "local://submissions/p1/1-tcc.pdf",
    });
    await analyzeSubmission("s1");
    mocks.env.AI_CHECKPOINT = undefined;
    await analyzeSubmission("s1");
    expect(mocks.updates).toEqual([
      { status: "erro", analysisMessage: NOT_STORED_MESSAGE },
      { status: "aguardando_analise", analysisMessage: NOT_CONFIGURED_MESSAGE },
    ]);
    expect(mocks.runInference).not.toHaveBeenCalled();
  });

  it("surfaces the bridge error message", async () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    mocks.runInference.mockRejectedValue(
      Object.assign(new Error("x"), {
        cause: {
          stderr: "Erro: PDF protegido: forneça uma cópia sem senha.\n",
        },
      }),
    );
    await analyzeSubmission("s1");
    expect(mocks.updates.at(-1)).toEqual({
      status: "erro",
      analysisMessage:
        "Falha na análise: PDF protegido: forneça uma cópia sem senha.",
    });
    expect(mocks.inserted).toHaveLength(0);
  });
});
