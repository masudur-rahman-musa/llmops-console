import { costGuard, type CostBody } from "@/features/cost/guards";
import { homeGuard, type HomeBody } from "@/features/home/guards";
import { incidentsGuard, type IncidentsBody } from "@/features/incidents/guards";
import { qualityGuard, type QualityBody } from "@/features/quality/guards";
import { releasesGuard, type ReleasesBody } from "@/features/releases/guards";
import { reliabilityGuard, type ReliabilityBody } from "@/features/reliability/guards";
import { residencyGuard, type ResidencyBody } from "@/features/residency/guards";

/**
 * The console's only read client: server-side, one fixed path per page, never a
 * query string (decision F). A failure is a reason, never a throw, so a page
 * degrades to a notice (decision D, a design note). The admin key is sent and never
 * logged. The seven per-page guards (one feature folder each) are assembled
 * here into PageBodies and GUARDS. Pinned by tests/console-api.test.ts.
 */

export const CONSOLE_PATHS = {
  home: "/console/home",
  cost: "/console/cost",
  quality: "/console/quality",
  residency: "/console/residency",
  reliability: "/console/reliability",
  incidents: "/console/incidents",
  releases: "/console/releases",
} as const;

export type PageName = keyof typeof CONSOLE_PATHS;

export type PageBodies = {
  home: HomeBody;
  cost: CostBody;
  quality: QualityBody;
  residency: ResidencyBody;
  reliability: ReliabilityBody;
  incidents: IncidentsBody;
  releases: ReleasesBody;
};

export const GUARDS: { [N in PageName]: (value: unknown) => value is PageBodies[N] } = {
  home: homeGuard,
  cost: costGuard,
  quality: qualityGuard,
  residency: residencyGuard,
  reliability: reliabilityGuard,
  incidents: incidentsGuard,
  releases: releasesGuard,
};

export const FETCH_TIMEOUT_MS = 5000;

export type Reason = "unconfigured" | "unauthorized" | "unreachable" | "invalid";

export type ConsoleResult<T> = { ok: true; body: T } | { ok: false; reason: Reason };

export type ConsoleDeps = {
  fetch?: typeof fetch;
  logger?: { error: (message: string) => void };
  env?: Record<string, string | undefined>;
};

/** The base's origin and path without trailing slashes; its query and fragment are dropped. */
function baseUrl(raw: string | undefined): string | null {
  if (!raw) {
    return null;
  }
  try {
    const url = new URL(raw);
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      return null;
    }
    return `${url.origin}${url.pathname.replace(/\/+$/, "")}`;
  } catch {
    return null;
  }
}

export async function getConsolePage<N extends PageName>(
  name: N,
  deps: ConsoleDeps = {},
): Promise<ConsoleResult<PageBodies[N]>> {
  const env = deps.env ?? process.env;
  const logger = deps.logger ?? console;
  const doFetch = deps.fetch ?? globalThis.fetch;
  const fail = (reason: Reason, detail: string): ConsoleResult<PageBodies[N]> => {
    logger.error(`console ${name}: ${reason}: ${detail}`);
    return { ok: false, reason };
  };

  const base = baseUrl(env.CONSOLE_API_URL);
  const key = env.CONSOLE_ADMIN_KEY;
  if (base === null || !key) {
    return fail("unconfigured", "CONSOLE_API_URL must be an http or https URL and CONSOLE_ADMIN_KEY must be set");
  }

  let response: Response;
  try {
    response = await doFetch(`${base}${CONSOLE_PATHS[name]}`, {
      method: "GET",
      headers: { Authorization: `Bearer ${key}` },
      cache: "no-store",
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
    });
  } catch (exc) {
    // Only the error's name: a message could echo request details.
    return fail("unreachable", exc instanceof Error ? exc.name : "fetch failed");
  }
  if (response.status === 401) {
    return fail("unauthorized", "status 401");
  }
  if (response.status !== 200) {
    return fail("unreachable", `status ${response.status}`);
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return fail("invalid", "the body is not JSON");
  }
  if (!GUARDS[name](body)) {
    return fail("invalid", "the body does not match the page's shape");
  }
  return { ok: true, body };
}
