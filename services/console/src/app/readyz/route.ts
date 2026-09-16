/**
 * Readiness is configuration only: the four values the console cannot serve
 * without. console-api being down is not unreadiness, because the pages degrade to a
 * notice and a Traefik 503 would replace it. The body never names the
 * failing value and never carries one. Pinned by tests/probes.test.ts.
 */
export const dynamic = "force-dynamic";

const MIN_SESSION_SECRET_LENGTH = 32;

function apiUrlIsHttp(value: string | undefined): boolean {
  if (!value) return false;
  try {
    const { protocol } = new URL(value);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
}

function configured(): boolean {
  const env = process.env;
  return (
    Boolean(env.CONSOLE_PASSWORD_SHA256) &&
    (env.CONSOLE_SESSION_SECRET ?? "").length >= MIN_SESSION_SECRET_LENGTH &&
    apiUrlIsHttp(env.CONSOLE_API_URL) &&
    Boolean(env.CONSOLE_ADMIN_KEY)
  );
}

export function GET(): Response {
  if (configured()) {
    return Response.json({ status: "ready" });
  }
  return Response.json({ status: "unconfigured" }, { status: 503 });
}
