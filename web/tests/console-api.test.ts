import { readFileSync } from "node:fs";

import { afterEach, describe, expect, it, vi, type Mock } from "vitest";

import {
  CONSOLE_PATHS,
  FETCH_TIMEOUT_MS,
  getConsolePage,
  type ConsoleDeps,
  type PageName,
} from "@/lib/console-api";

const BASE = "http://console-api.test";
const KEY = "adminkey-sentinel-7c41e2d9f0";
const PAGES: PageName[] = ["home", "cost", "quality", "reliability", "incidents", "releases", "models", "requests"];

function fixture(name: PageName): unknown {
  return JSON.parse(readFileSync(new URL(`./fixtures/${name}.json`, import.meta.url), "utf8"));
}

type Harness = { deps: ConsoleDeps; fetch: Mock; error: Mock };

// Every call gets a fresh Response, because a body can be read once.
function harness(
  respond: () => Promise<Response> | Response,
  env: Record<string, string | undefined> = { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: KEY },
): Harness {
  const fetch = vi.fn(async () => respond());
  const error = vi.fn();
  return { deps: { fetch: fetch as unknown as typeof globalThis.fetch, logger: { error }, env }, fetch, error };
}

function okResponse(body: unknown): () => Response {
  return () => new Response(JSON.stringify(body), { status: 200 });
}

function fetchedUrl(fetch: Mock): string {
  expect(fetch).toHaveBeenCalledTimes(1);
  const url: unknown = fetch.mock.calls[0][0];
  expect(typeof url).toBe("string");
  return url as string;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getConsolePage", () => {
  it("the eight paths are pinned", async () => {
    // Catches a path map with a missing, extra or misspelt page, or a client that fetches a path other than the map's own.
    expect(CONSOLE_PATHS).toEqual({
      home: "/console/home",
      cost: "/console/cost",
      quality: "/console/quality",
      reliability: "/console/reliability",
      incidents: "/console/incidents",
      releases: "/console/releases",
      models: "/console/models",
      requests: "/console/requests",
    });
    for (const name of PAGES) {
      const { deps, fetch } = harness(okResponse(fixture(name)));
      const result = await getConsolePage(name, deps);
      expect(fetchedUrl(fetch), name).toBe(`${BASE}/console/${name}`);
      expect(result, name).toEqual({ ok: true, body: fixture(name) });
    }
  });

  it("a call sends the bearer key, no-store and a 5000 ms timeout", async () => {
    // Catches a request missing the Bearer key, sent cacheable, sent with a method other than GET, or bounded by a timeout other than 5000 ms.
    expect(FETCH_TIMEOUT_MS).toBe(5000);
    const signal = new AbortController().signal;
    const timeout = vi.spyOn(AbortSignal, "timeout").mockReturnValue(signal);
    const { deps, fetch } = harness(okResponse(fixture("home")));

    const result = await getConsolePage("home", deps);

    expect(result.ok).toBe(true);
    expect(timeout).toHaveBeenCalledWith(5000);
    fetchedUrl(fetch);
    const init = fetch.mock.calls[0][1] as RequestInit;
    expect(init.method === undefined || init.method === "GET").toBe(true);
    expect(init.headers).not.toBeInstanceOf(Headers);
    expect((init.headers as Record<string, string>).Authorization).toBe(`Bearer ${KEY}`);
    expect(init.cache).toBe("no-store");
    expect(init.signal).toBe(signal);
  });

  it("the url carries no query string", async () => {
    // Catches a URL built by concatenating the raw base, which carries the base's query, fragment or trailing slash into the request.
    const bases = [
      [`${BASE}/`, `${BASE}/console/cost`],
      [`${BASE}/api//`, `${BASE}/api/console/cost`],
      [`${BASE}/api?days=90`, `${BASE}/api/console/cost`],
      [`${BASE}/api/?tenant=acme-eu#top`, `${BASE}/api/console/cost`],
    ];
    for (const [base, expected] of bases) {
      const { deps, fetch } = harness(okResponse(fixture("cost")), { CONSOLE_API_URL: base, CONSOLE_ADMIN_KEY: KEY });
      await getConsolePage("cost", deps);
      const url = fetchedUrl(fetch);
      expect(url, base).toBe(expected);
      expect(url, base).not.toContain("?");
    }
  });

  it("an unset url or key makes no fetch and is unconfigured with an error log", async () => {
    // Catches a client that fetches, or answers anything but unconfigured, or stays silent, when either variable is unset or empty.
    const envs: Array<Record<string, string | undefined>> = [
      { CONSOLE_ADMIN_KEY: KEY },
      { CONSOLE_API_URL: "", CONSOLE_ADMIN_KEY: KEY },
      { CONSOLE_API_URL: BASE },
      { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: "" },
    ];
    for (const env of envs) {
      const { deps, fetch, error } = harness(okResponse(fixture("home")), env);
      const result = await getConsolePage("home", deps);
      expect(result, JSON.stringify(env)).toEqual({ ok: false, reason: "unconfigured" });
      expect(fetch).not.toHaveBeenCalled();
      expect(error).toHaveBeenCalled();
    }
  });

  it("a non http url is unconfigured", async () => {
    // Catches a client that fetches any URL that parses, or passes an unparseable base on to fetch, instead of treating it as unconfigured.
    for (const base of ["ftp://x", "not a url"]) {
      const { deps, fetch, error } = harness(okResponse(fixture("home")), { CONSOLE_API_URL: base, CONSOLE_ADMIN_KEY: KEY });
      const result = await getConsolePage("home", deps);
      expect(result, base).toEqual({ ok: false, reason: "unconfigured" });
      expect(fetch).not.toHaveBeenCalled();
      expect(error).toHaveBeenCalled();
    }
    const { deps, fetch } = harness(okResponse(fixture("home")), { CONSOLE_API_URL: "https://console-api.test", CONSOLE_ADMIN_KEY: KEY });
    expect((await getConsolePage("home", deps)).ok).toBe(true);
    expect(fetchedUrl(fetch)).toBe("https://console-api.test/console/home");
  });

  it("a 401 is unauthorized", async () => {
    // Catches a 401 folded into unreachable, or reported without an error log.
    const { deps, error } = harness(() => new Response("", { status: 401 }));
    expect(await getConsolePage("home", deps)).toEqual({ ok: false, reason: "unauthorized" });
    expect(error).toHaveBeenCalled();
  });

  it("a 500, a timeout and a network error are unreachable", async () => {
    // Catches a client that throws on a rejected fetch, or accepts any 2xx (201 carries a valid body) as success.
    const home = JSON.stringify(fixture("home"));
    const failures: Array<[string, () => Promise<Response> | Response]> = [
      ["500", () => new Response("", { status: 500 })],
      ["403", () => new Response("", { status: 403 })],
      ["201", () => new Response(home, { status: 201 })],
      ["timeout", () => Promise.reject(new DOMException("The operation was aborted due to timeout", "TimeoutError"))],
      ["network", () => Promise.reject(new TypeError("fetch failed"))],
    ];
    for (const [label, respond] of failures) {
      const { deps, error } = harness(respond);
      expect(await getConsolePage("home", deps), label).toEqual({ ok: false, reason: "unreachable" });
      expect(error, label).toHaveBeenCalled();
    }
  });

  it("a non json body and a guard failure are invalid", async () => {
    // Catches a client that throws on a non JSON body, or returns a 200 body without running the page's guard.
    const withoutField = fixture("home") as Record<string, unknown>;
    delete withoutField.requests_today;
    const bodies: Array<[string, () => Response]> = [
      ["not json", () => new Response("not json", { status: 200 })],
      ["guard", okResponse(withoutField)],
      ["other page", okResponse(fixture("releases"))],
    ];
    for (const [label, respond] of bodies) {
      const { deps, error } = harness(respond);
      expect(await getConsolePage("home", deps), label).toEqual({ ok: false, reason: "invalid" });
      expect(error, label).toHaveBeenCalled();
    }
  });

  it("the admin key never appears in a log line or a thrown message", async () => {
    // Catches a client that logs the request headers or init, or throws with them, on any path.
    const home = fixture("home");
    const cases: Array<[string, () => Promise<Response> | Response, Record<string, string | undefined>]> = [
      ["ok", okResponse(home), { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: KEY }],
      ["unconfigured", okResponse(home), { CONSOLE_API_URL: "ftp://x", CONSOLE_ADMIN_KEY: KEY }],
      ["unauthorized", () => new Response("", { status: 401 }), { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: KEY }],
      ["unreachable", () => new Response("", { status: 500 }), { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: KEY }],
      ["timeout", () => Promise.reject(new DOMException("timed out", "TimeoutError")), { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: KEY }],
      ["invalid", () => new Response("not json", { status: 200 }), { CONSOLE_API_URL: BASE, CONSOLE_ADMIN_KEY: KEY }],
    ];
    for (const [label, respond, env] of cases) {
      const { deps, error } = harness(respond, env);
      let thrown = "";
      try {
        await getConsolePage("home", deps);
      } catch (exc) {
        thrown = exc instanceof Error ? `${exc.message}\n${exc.stack ?? ""}` : String(exc);
      }
      expect(thrown, label).toBe("");
      const logged = error.mock.calls
        .map((call) => call.map((arg) => (typeof arg === "string" ? arg : `${String(arg)} ${JSON.stringify(arg)}`)).join(" "))
        .join("\n");
      expect(logged, label).not.toContain(KEY);
    }
  });
});
