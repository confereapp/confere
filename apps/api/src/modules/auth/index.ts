import { isGoogleEnabled } from "@confere/auth/social-providers";
import { env } from "@confere/env/server";
import { protectedProcedure, publicProcedure } from "../../index";
import { AuthService } from "./service";

export const authModule = protectedProcedure.query(({ ctx }) =>
  AuthService.privateData(ctx),
);

// Público: o app mostra o botão do Google só quando o servidor tem as credenciais.
export const authProvidersModule = publicProcedure.query(() => ({
  google: isGoogleEnabled(env),
}));
