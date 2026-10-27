import { describe, expect, it } from "vitest";

import { SESSION_COOKIE, verifySession } from "@/lib/session";

import { checkConsole } from "./live/check";

const BASE = "http://127.0.0.1:3999";
const SECRET = "test-session-secret-0123456789abcdefghij";
const TABS = ["/", "/requests", "/cost", "/quality"];
const LOGIN_PAGE = "<h1>Sign in</h1><p>Password</p>";

type Page = { status: number; body?: string; location?: string };
type Call = { path: string; cookie: string | null; redirect: RequestRedirect | undefined };

function clean(path: string): Page {
  return { status: 200, body: `<h1>${path}</h1><p>Answered 3 of 10, 48.48%</p>` };
}

/**
 * A console served from a table. A 3xx is handed back as is only under
 * `redirect: "manual"`; otherwise it is followed to the login page, as a real
 * fetch would, so a check that lets fetch follow it sees a clean page.
 */
function fakeConsole(pages: Record<string, Page>, calls: Call[]): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = new URL(input instanceof Request ? input.url : String(input));
    calls.push({ path: url.pathname, cookie: new Headers(init?.headers).get("cookie"), redirect: init?.redirect });
    const page = pages[url.pathname] ?? { status: 404, body: "<h1>Not found</h1>" };
    if (page.status >= 300 && page.status < 400) {
      if (init?.redirect === "manual") {
        return new Response(null, { status: page.status, headers: { Location: page.location ?? "/login" } });
      }
      return new Response(LOGIN_PAGE, { status: 200, headers: { "Content-Type": "text/html" } });
    }
    return new Response(page.body ?? "", { status: page.status, headers: { "Content-Type": "text/html" } });
  };
}

function cleanConsole(): Record<string, Page> {
  return Object.fromEntries(TABS.map((path) => [path, clean(path)]));
}

describe("checkConsole", () => {
  it("a clean console passes", async () => {
    // Catches a check that skips a tab, sends no session cookie, or signs one the console would refuse (wrong name, wrong secret, already expired).
    const calls: Call[] = [];
    const failures = await checkConsole({ baseUrl: BASE, secret: SECRET, fetchFn: fakeConsole(cleanConsole(), calls) });
    expect(failures).toEqual([]);
    expect([...new Set(calls.map((call) => call.path))].sort()).toEqual([...TABS].sort());
    const now = Math.floor(Date.now() / 1000);
    for (const call of calls) {
      const value = call.cookie?.match(new RegExp(`(?:^|;\\s*)${SESSION_COOKIE}=([^;]+)`))?.[1];
      expect(value, call.path).toBeDefined();
      expect(await verifySession(value, SECRET, now), call.path).toBe(true);
    }
  });

  it("junk is named with its page", async () => {
    // Catches a check that finds the junk but drops which page it was on, or names it on every page.
    const pages = cleanConsole();
    pages["/requests"] = { status: 200, body: "<h1>Requests</h1><p>Failed 13.0128054</p>" };
    const failures = await checkConsole({ baseUrl: BASE, secret: SECRET, fetchFn: fakeConsole(pages, []) });
    expect(failures).toEqual(["/requests: long decimal: 13.0128054"]);
  });

  it.each([303, 307])("a redirect to login fails: %i", async (status) => {
    // Catches a check that lets fetch follow the redirect and then passes the clean login page, so a bad cookie looks like a clean console.
    const pages = cleanConsole();
    pages["/quality"] = { status, location: "/login?next=%2Fquality" };
    const failures = await checkConsole({ baseUrl: BASE, secret: SECRET, fetchFn: fakeConsole(pages, []) });
    expect(failures).toHaveLength(1);
    expect(failures[0].startsWith("/quality: ")).toBe(true);
  });

  it("a page that answers 500 fails", async () => {
    // Catches a check that reads only the body, so a broken tab whose error page holds no junk passes.
    const pages = cleanConsole();
    pages["/cost"] = { status: 500, body: "<h1>Something went wrong</h1>" };
    const failures = await checkConsole({ baseUrl: BASE, secret: SECRET, fetchFn: fakeConsole(pages, []) });
    expect(failures).toHaveLength(1);
    expect(failures[0].startsWith("/cost: ")).toBe(true);
  });

  it("an unreachable console fails", async () => {
    // Catches a check that throws on a refused connection, or reports nothing, instead of a failure naming where it tried to reach.
    const refused: typeof fetch = async () => {
      throw new TypeError("fetch failed");
    };
    const failures = await checkConsole({ baseUrl: BASE, secret: SECRET, fetchFn: refused });
    expect(failures.length).toBeGreaterThan(0);
    expect(failures.some((failure) => failure.includes(BASE))).toBe(true);
  });
});
