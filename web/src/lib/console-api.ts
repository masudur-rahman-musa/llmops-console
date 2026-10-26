import { costGuard, type CostBody } from "@/features/cost/guards";
import { homeGuard, type HomeBody } from "@/features/home/guards";
import { qualityGuard, type QualityBody } from "@/features/quality/guards";
import { requestsGuard, type RequestsBody } from "@/features/requests/guards";
import type { RequestFilters } from "@/lib/filters";

/**
 * The console's only read client: server-side, one fixed path per page. The one
 * query string is on the requests call: `?page=N` from a whole number, then the filters
 * that are set, each already checked by `readRequestFilters` and encoded here. A failure is a reason, never a throw, so a page
 * degrades to a notice. The admin key is sent and never
 * logged. The four per-page guards (one feature folder each) are assembled
 * here into PageBodies and GUARDS.
 */

export const CONSOLE_PATHS = {
  home: "/console/home",
  cost: "/console/cost",
  quality: "/console/quality",
  requests: "/console/requests",
} as const;

export type PageName = keyof typeof CONSOLE_PATHS;

export type PageBodies = {
  home: HomeBody;
  cost: CostBody;
  quality: QualityBody;
  requests: RequestsBody;
};

export const GUARDS: { [N in PageName]: (value: unknown) => value is PageBodies[N] } = {
  home: homeGuard,
  cost: costGuard,
  quality: qualityGuard,
  requests: requestsGuard,
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

/**
 * True when console-api is reachable in principle: a parseable http(s) URL and a
 * set admin key. Reuses `baseUrl`, so what readyz accepts matches what this
 * client accepts. Pinned by tests/screens.test.tsx.
 */
export function apiIsConfigured(env: Record<string, string | undefined> = process.env): boolean {
  return baseUrl(env.CONSOLE_API_URL) !== null && Boolean(env.CONSOLE_ADMIN_KEY);
}

/** The requests page's query: which page of the ledger to read, and the filters that narrow it. */
export type PageQuery = { page: number; filters?: RequestFilters };

/**
 * `?page=N` and then each filter that is set, for the requests call when asked, else nothing;
 * a page that is not a whole number above 0 reads as 1. Booleans are sent only when true.
 */
function queryString(name: PageName, query: PageQuery | undefined): string {
  if (name !== "requests" || query === undefined) {
    return "";
  }
  const page = Number.isSafeInteger(query.page) && query.page >= 1 ? query.page : 1;
  const params = new URLSearchParams({ page: String(page) });
  const filters = query.filters;
  if (filters?.day) params.set("day", filters.day);
  if (filters?.tenant) params.set("tenant", filters.tenant);
  if (filters?.model) params.set("model", filters.model);
  if (filters?.errors) params.set("errors", "true");
  if (filters?.slow) params.set("slow", "true");
  if (filters?.cached) params.set("cached", "true");
  return `?${params.toString()}`;
}

export async function getConsolePage<N extends PageName>(
  name: N,
  deps: ConsoleDeps = {},
  query?: PageQuery,
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
    response = await doFetch(`${base}${CONSOLE_PATHS[name]}${queryString(name, query)}`, {
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
