import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import { config, middleware } from "@/middleware";

// Edge middleware cannot load node:crypto. Importing it anywhere in the
// middleware's module graph fails this file, so verification must use Web Crypto.
vi.mock("node:crypto", () => {
  throw new Error("edge middleware may not import node:crypto");
});

const ORIGIN = "http://console.test";
// 32 characters exactly: the smallest usable secret.
const SECRET = "0123456789abcdef0123456789abcdef";
const NOW_MS = Date.UTC(2026, 8, 16, 12, 0, 0);
const NOW = Math.floor(NOW_MS / 1000);

// Web Crypto HMAC, independent of the module under test.
async function signed(expires: number): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const payload = `v1.${expires}`;
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(payload));
  const hex = Array.from(new Uint8Array(mac), (byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${payload}.${hex}`;
}

function request(path: string, session?: string): NextRequest {
  const headers: Record<string, string> = session === undefined ? {} : { cookie: `opsconsole_session=${session}` };
  return new NextRequest(`${ORIGIN}${path}`, { headers });
}

function expectPassed(response: Response, label: string): void {
  expect(response.headers.get("x-middleware-next"), label).toBe("1");
  expect(response.headers.get("location"), label).toBeNull();
}

function expectRedirected(response: Response, path: string): void {
  expect(response.status, path).toBe(307);
  expect(response.headers.get("location"), path).toBe(`${ORIGIN}/login?next=${encodeURIComponent(path)}`);
  expect(response.headers.get("x-middleware-next"), path).toBeNull();
}

let errorSpy: MockInstance;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW_MS);
  vi.stubEnv("CONSOLE_SESSION_SECRET", SECRET);
  errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  errorSpy.mockRestore();
});

describe("middleware", () => {
  it("no cookie redirects with next encoded", async () => {
    // Catches a middleware that lets an unauthenticated request through, redirects elsewhere, or puts the path in next unencoded.
    for (const path of ["/", "/cost"]) {
      expectRedirected(await middleware(request(path)), path);
    }
    expect(encodeURIComponent("/")).toBe("%2F");
  });

  it("a valid cookie passes", async () => {
    // Catches a middleware that never verifies (redirects everyone), reads the secret at import time, or passes the clock in milliseconds.
    const value = await signed(NOW + 43_200);
    for (const path of ["/", "/cost", "/releases"]) {
      expectPassed(await middleware(request(path, value)), path);
    }
    // Expires one second from now: a millisecond clock would refuse it.
    expectPassed(await middleware(request("/cost", await signed(NOW + 1))), "expires minus one");
  });

  it("an expired cookie redirects", async () => {
    // Catches a middleware that trusts a correctly signed cookie without checking it against the current clock.
    for (const expires of [NOW, NOW - 1, NOW - 43_200]) {
      expectRedirected(await middleware(request("/cost", await signed(expires))), "/cost");
    }
    // The same signing passes one second later in the other direction, so only expiry refuses the rows above.
    expectPassed(await middleware(request("/cost", await signed(NOW + 1))), "not yet expired");
  });

  it("the exempt paths pass without a cookie", async () => {
    // Catches a middleware that guards the login routes or Next assets (a redirect loop), or exempts by prefix so /login/x or /api/loginx slip through.
    vi.stubEnv("CONSOLE_SESSION_SECRET", undefined);
    for (const path of [
      "/login",
      "/api/login",
      "/api/logout",
      "/healthz",
      "/readyz",
      "/_next/static/chunks/main.js",
      "/_next/image",
    ]) {
      expectPassed(await middleware(request(path)), path);
    }

    vi.stubEnv("CONSOLE_SESSION_SECRET", SECRET);
    for (const path of ["/login/x", "/loginx", "/api/login2", "/api/logout/x", "/_nextx", "/api", "/cost/login"]) {
      expectRedirected(await middleware(request(path)), path);
    }
  });

  it("the matcher is pinned", () => {
    // Catches a matcher that drops paths from the middleware (for example excluding /api) or stops excluding Next assets.
    expect(config).toEqual({ matcher: ["/((?!_next/).*)"] });
  });
});
