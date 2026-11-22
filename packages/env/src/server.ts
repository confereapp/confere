import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createEnv } from "@t3-oss/env-core";
import { config } from "dotenv";
import { z } from "zod";

// Load monorepo package env relative to this file (not process.cwd()).
// Turbo/`bun run` from the workspace root would otherwise miss apps/*/ .env.
const envFilePath = resolve(
  fileURLToPath(new URL(".", import.meta.url)),
  "../../../apps/server/.env",
);
config({ path: envFilePath });

export const env = createEnv({
  server: {
    DATABASE_URL: z.string().min(1),
    BETTER_AUTH_SECRET: z.string().min(32),
    BETTER_AUTH_URL: z.url(),

    CORS_ORIGIN: z.url(),

    AI_CHECKPOINT: z.string().min(1).optional(),
    AI_CHAT_CHECKPOINT: z.string().min(1).optional(),
    AI_PYTHON: z.string().min(1).optional(),
    AI_DEVICE: z.enum(["auto", "cpu", "mps"]).default("auto"),

    GMAIL_USER: z.email().optional(),
    GMAIL_APP_PASSWORD: z.string().min(1).optional(),
    GOOGLE_CLIENT_ID: z.string().min(1).optional(),
    GOOGLE_CLIENT_SECRET: z.string().min(1).optional(),
    NODE_ENV: z
      .enum(["development", "production", "test"])
      .default("development"),
  },
  runtimeEnv: process.env,
  skipValidation: !!process.env.SKIP_ENV_VALIDATION,
  emptyStringAsUndefined: true,
});
