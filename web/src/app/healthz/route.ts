/**
 * Liveness: the process answers. Reads no configuration and makes no fetch, so an
 * unconfigured pod is held out of the Service by readiness, never restarted in a loop.
 * Pinned by tests/probes.test.ts.
 */
export const dynamic = "force-dynamic";

export function GET(): Response {
  return Response.json({ status: "ok" });
}
