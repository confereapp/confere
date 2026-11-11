import { z } from "zod";

export const reviewTextInput = z.object({
  text: z
    .string()
    .max(8000)
    .refine((text) => text.trim().length > 0),
  requirements: z.array(z.string().trim().min(1).max(600)).min(1).max(4),
});

const status = z.enum(["atendido", "nao_atendido", "evidencia_insuficiente"]);

export const reviewTextResult = z.object({
  schema_version: z.union([z.literal(1), z.literal(2)]),
  experimental: z.literal(true),
  offset_unit: z.literal("unicode_codepoints"),
  scope_note: z.string(),
  sources: z.array(
    z.object({
      id: z.string(),
      name: z.string(),
      sha256: z.string(),
      format: z.string(),
      warnings: z.array(z.string()),
      units: z.array(
        z.object({
          id: z.string(),
          characters: z.number().int().nonnegative(),
        }),
      ),
    }),
  ),
  requirements: z.array(
    z.object({
      id: z.string(),
      requirement: z.string(),
      status,
      requires_human_review: z.literal(true),
      conflicting_evidence: z.boolean(),
      model_outputs_valid: z.boolean(),
      evidence: z.array(
        z.object({
          document_id: z.string(),
          document: z.string(),
          source_sha256: z.string(),
          source_unit: z.string(),
          char_start: z.number().int().nonnegative(),
          char_end: z.number().int().nonnegative(),
          quote: z.string(),
          model_status: status,
        }),
      ),
      coverage: z.object({
        scope: z.literal("all_extracted_text"),
        all_extracted_text_processed: z.literal(true),
        text_characters: z.number().int().nonnegative(),
        processed_characters: z.number().int().nonnegative(),
        batches: z.number().int().positive(),
        units_without_text: z.array(
          z.object({ document_id: z.string(), unit: z.string() }),
        ),
      }),
    }),
  ),
  model: z.object({
    sha256: z.string().regex(/^[a-f0-9]{64}$/),
    step: z.number().int().nonnegative(),
    device: z.string(),
    context_size: z.number().int().positive(),
    response_tokens: z.number().int().positive(),
  }),
});
