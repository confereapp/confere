import { router } from "./index";
import { authModule, authProvidersModule } from "./modules/auth";
import { healthModule } from "./modules/health";
import { projectRouter } from "./modules/project";

export const appRouter = router({
  healthCheck: healthModule,
  privateData: authModule,
  authProviders: authProvidersModule,
  project: projectRouter,
});

export type AppRouter = typeof appRouter;
