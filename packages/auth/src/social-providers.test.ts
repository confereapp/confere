import { expect, test } from "vitest";
import { isGoogleEnabled, socialProviders } from "./social-providers";

test("registra o Google só com client id e secret", () => {
  const env = { GOOGLE_CLIENT_ID: "id", GOOGLE_CLIENT_SECRET: "secret" };
  expect(socialProviders(env)).toEqual({
    google: { clientId: "id", clientSecret: "secret" },
  });
  expect(isGoogleEnabled(env)).toBe(true);
});

test("não registra o Google sem alguma das envs", () => {
  for (const env of [
    {},
    { GOOGLE_CLIENT_ID: "id" },
    { GOOGLE_CLIENT_SECRET: "secret" },
  ]) {
    expect(socialProviders(env)).toEqual({});
    expect(isGoogleEnabled(env)).toBe(false);
  }
});
