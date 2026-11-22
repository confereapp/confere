import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { extname, join } from "node:path";
import { and, db, eq, inArray } from "@confere/db";
import {
  checkResult,
  requirement,
  submission,
} from "@confere/db/schema/project";
import { env } from "@confere/env/server";
import { TRPCError } from "@trpc/server";
import { getOptionalS3Bucket } from "../project/service";
import { submissionReviewResult } from "./model";
import { runInference } from "./service";

type ReviewedRequirement =
  (typeof submissionReviewResult._output)["requirements"][number];

const QUOTE_LIMIT = 800;
const RETRY_DELAY = 5000;
const ANALYSIS_TIMEOUT = 10 * 60 * 1000;
export const NOT_CONFIGURED_MESSAGE =
  "Análise automática indisponível: o servidor não tem AI_CHECKPOINT configurado.";
export const NOT_STORED_MESSAGE = "Arquivo não armazenado: configure o bucket.";

// O modelo é classificador: nunca afirma ausência ("faltando") e não gera justificativa.
export function toCheckResult(item: ReviewedRequirement) {
  const met = item.status === "atendido";
  const evidence =
    item.evidence.find((e) => e.model_status === item.status) ??
    (met ? undefined : item.evidence[0]);
  if (!evidence) {
    return {
      status: "revisar",
      feedback: "Sem evidência suficiente; revise manualmente.",
      quote: null,
      pageNumber: null,
    };
  }
  const page = /^page:(\d+)$/.exec(evidence.source_unit)?.[1];
  const where = page ? `na p. ${page}` : `em ${evidence.source_unit}`;
  return {
    status: met ? "atendido" : "revisar",
    feedback: met
      ? `Trecho encontrado ${where}.`
      : `O modelo não confirmou o requisito; trecho analisado ${where}. Revise manualmente.`,
    quote:
      evidence.quote.length > QUOTE_LIMIT
        ? `${evidence.quote.slice(0, QUOTE_LIMIT)}…`
        : evidence.quote,
    pageNumber: page ? Number(page) : null,
  };
}

function setStatus(
  id: string,
  status: string,
  analysisMessage: string | null = null,
) {
  return db
    .update(submission)
    .set({ status, analysisMessage })
    .where(eq(submission.id, id));
}

function failureMessage(error: unknown) {
  const stderr = (error as { cause?: { stderr?: string } }).cause?.stderr;
  const detail = stderr?.match(/^Erro: (.+)$/m)?.[1];
  return detail
    ? `Falha na análise: ${detail}`
    : "Falha na análise automática. Envie a entrega novamente.";
}

export async function analyzeSubmission(id: string) {
  const sub = await db.query.submission.findFirst({
    where: eq(submission.id, id),
  });
  if (!sub || sub.status === "concluido" || sub.status === "erro") return;
  const key = sub.fileUrl;
  if (!key || key.startsWith("local://")) {
    await setStatus(id, "erro", NOT_STORED_MESSAGE);
    return;
  }
  if (!env.AI_CHECKPOINT) {
    await setStatus(id, "aguardando_analise", NOT_CONFIGURED_MESSAGE);
    return;
  }
  const requirements = await db.query.requirement.findMany({
    where: and(
      eq(requirement.projectId, sub.projectId),
      eq(requirement.checked, true),
    ),
    orderBy: [requirement.order, requirement.createdAt],
  });
  if (requirements.length === 0) {
    await setStatus(id, "erro", "Nenhum requisito selecionado para conferir.");
    return;
  }

  await setStatus(id, "processando");
  const dir = await mkdtemp(join(tmpdir(), "confere-"));
  try {
    const bucket = await getOptionalS3Bucket();
    if (!bucket) {
      await setStatus(id, "erro", NOT_STORED_MESSAGE);
      return;
    }
    const object = await bucket.download(key);
    const bytes = await object.Body?.transformToByteArray();
    if (!bytes) throw new Error("Download vazio do bucket.");
    const path = join(dir, `entrega${extname(key).toLowerCase()}`);
    await writeFile(path, bytes);

    const review = await runInference(
      "confere_ai.submission_review",
      {
        path,
        requirements: requirements.map((r) => ({
          id: r.id,
          title: r.title,
          description: r.description,
        })),
      },
      (value) => submissionReviewResult.parse(value),
      { timeout: ANALYSIS_TIMEOUT },
    );
    const rows = review.requirements.map((item) => ({
      id: crypto.randomUUID(),
      submissionId: id,
      requirementId: item.id,
      ...toCheckResult(item),
    }));
    const met = rows.filter((row) => row.status === "atendido").length;
    await db.transaction(async (tx) => {
      await tx.delete(checkResult).where(eq(checkResult.submissionId, id));
      await tx.insert(checkResult).values(rows);
      await tx
        .update(submission)
        .set({
          status: "concluido",
          coverage: Math.round((100 * met) / rows.length),
          analysisMessage: null,
        })
        .where(eq(submission.id, id));
    });
  } catch (error) {
    // Chat ou revisão textual ocupando a inferência: volta para a fila.
    if (error instanceof TRPCError && error.code === "TOO_MANY_REQUESTS") {
      await setStatus(id, "aguardando_analise");
      setTimeout(() => enqueueAnalysis(id), RETRY_DELAY);
      return;
    }
    console.error("[ai] análise da entrega falhou", id, error);
    await setStatus(id, "erro", failureMessage(error));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
}

// ponytail: fila em memória e serial no processo; trocar por fila persistente se houver mais de uma instância da API.
let queue: Promise<void> = Promise.resolve();

export function enqueueAnalysis(id: string) {
  queue = queue
    .then(() => analyzeSubmission(id))
    .catch((error) => console.error("[ai] fila de análise", id, error));
  return queue;
}

export async function resumePendingAnalyses() {
  const pending = await db
    .select({ id: submission.id })
    .from(submission)
    .where(inArray(submission.status, ["aguardando_analise", "processando"]));
  for (const { id } of pending) enqueueAnalysis(id);
}
