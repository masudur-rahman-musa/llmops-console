import { apiIsConfigured } from "@/lib/console-api";
import { secretIsUsable } from "@/lib/session";

/**
 * Readiness is configuration only: the four values the console cannot serve
 * without. console-api being down is not unreadiness, because the pages degrade to a
 * notice and a Traefik 503 would replace it. The body never names the
 * failing value and never carries one. The session-secret length rule lives in
 * `secretIsUsable` and the API-url check in `apiIsConfigured`, so there is one home
 * for each. Pinned by tests/probes.test.ts and tests/screens.test.tsx.
 */
export const dynamic = "force-dynamic";

function configured(): boolean {
  const env = process.env;
  return (
    Boolean(env.CONSOLE_PASSWORD_SHA256) &&
    secretIsUsable(env.CONSOLE_SESSION_SECRET) &&
    apiIsConfigured(env)
  );
}

export function GET(): Response {
  if (configured()) {
    return Response.json({ status: "ready" });
  }
  return Response.json({ status: "unconfigured" }, { status: 503 });
}
