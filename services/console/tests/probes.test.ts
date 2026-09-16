import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import { GET as healthzRoute } from "@/app/healthz/route";
import { GET as readyzRoute } from "@/app/readyz/route";
import { middleware } from "@/middleware";

// Edge middleware cannot load node:crypto; the same guard middleware.test.ts carries.
vi.mock("node:crypto", () => {
  throw new Error("edge middleware may not import node:crypto");
});

const ORIGIN = "http://console.test";
const KEYS = ["CONSOLE_PASSWORD_SHA256", "CONSOLE_SESSION_SECRET", "CONSOLE_API_URL", "CONSOLE_ADMIN_KEY"] as const;
type Key = (typeof KEYS)[number];
type Env = Record<Key, string | undefined>;

// 48 characters: comfortably above the minimum, so the boundary row below is the only 32.
const LONG_SECRET = "0123456789abcdef0123456789abcdef0123456789abcdef";
const EXACT_SECRET = "0123456789abcdef0123456789abcdef";
const SHORT_SECRET = "0123456789abcdef0123456789abcde";

const CONFIGURED: Env = {
  CONSOLE_PASSWORD_SHA256: "a".repeat(64),
  CONSOLE_SESSION_SECRET: LONG_SECRET,
  CONSOLE_API_URL: "http://gateway.apps.svc.cluster.local:8080",
  CONSOLE_ADMIN_KEY: "admin-key-for-tests",
};

// Values no correct body or header could contain by accident.
const SENTINELS: Env = {
  CONSOLE_PASSWORD_SHA256: "sentinel-hash-4f1c9e",
  CONSOLE_SESSION_SECRET: "sentinel-secret-7b2d-0123456789abcdef0123456789",
  CONSOLE_API_URL: "https://sentinel-url-9a3e.example",
  CONSOLE_ADMIN_KEY: "sentinel-admin-5c8f",
};

type Handler = (request: NextRequest) => Response | Promise<Response>;

// Whether a route takes the request or not, it is called the way Next calls it.
const healthz = healthzRoute as unknown as Handler;
const readyz = readyzRoute as unknown as Handler;

function probe(path: string): NextRequest {
  return new NextRequest(`${ORIGIN}${path}`);
}

function setEnv(env: Env): void {
  for (const key of KEYS) {
    vi.stubEnv(key, env[key]);
  }
}

async function expectReady(label: string): Promise<void> {
  const response = await readyz(probe("/readyz"));
  expect(response.status, label).toBe(200);
  expect(await response.json(), label).toEqual({ status: "ready" });
}

async function expectUnconfigured(label: string): Promise<void> {
  const response = await readyz(probe("/readyz"));
  expect(response.status, label).toBe(503);
  expect(await response.json(), label).toEqual({ status: "unconfigured" });
}

let fetchSpy: MockInstance;
let errorSpy: MockInstance;

beforeEach(() => {
  fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(() => {
    throw new Error("a probe may not fetch");
  });
  errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  fetchSpy.mockRestore();
  errorSpy.mockRestore();
});

describe("probes", () => {
  it("healthz is 200 ok with no environment", async () => {
    // Catches a healthz that validates configuration or dependencies, so an unconfigured pod is restarted by liveness forever.
    setEnv({ CONSOLE_PASSWORD_SHA256: undefined, CONSOLE_SESSION_SECRET: undefined, CONSOLE_API_URL: undefined, CONSOLE_ADMIN_KEY: undefined });
    const response = await healthz(probe("/healthz"));
    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({ status: "ok" });
  });

  it("readyz is 200 when all four are configured", async () => {
    // Catches a readyz that refuses a valid configuration, rejects https, or answers with another body.
    setEnv(CONFIGURED);
    await expectReady("http url");
    setEnv({ ...CONFIGURED, CONSOLE_API_URL: "https://console-api.example:8443" });
    await expectReady("https url");
  });

  it("readyz is 503 unconfigured for each missing value", async () => {
    // Catches a readyz that skips any one of the four checks, accepts a 31-character secret, accepts a non-http url, treats empty as set, throws on an unparseable url, or reads the environment once at import.
    const rows: [string, Partial<Env>][] = [
      ["hash unset", { CONSOLE_PASSWORD_SHA256: undefined }],
      ["hash empty", { CONSOLE_PASSWORD_SHA256: "" }],
      ["secret unset", { CONSOLE_SESSION_SECRET: undefined }],
      ["secret 31 characters", { CONSOLE_SESSION_SECRET: SHORT_SECRET }],
      ["url unset", { CONSOLE_API_URL: undefined }],
      ["url ftp", { CONSOLE_API_URL: "ftp://x" }],
      ["url unparseable", { CONSOLE_API_URL: "not a url" }],
      ["admin key unset", { CONSOLE_ADMIN_KEY: undefined }],
      ["admin key empty", { CONSOLE_ADMIN_KEY: "" }],
    ];
    expect(SHORT_SECRET).toHaveLength(31);
    for (const [label, override] of rows) {
      // The same process answers 200 first, so only the override can make the row 503.
      setEnv(CONFIGURED);
      await expectReady(`${label}: baseline`);
      setEnv({ ...CONFIGURED, ...override });
      await expectUnconfigured(label);
    }
  });

  it("readyz accepts a secret of exactly 32 characters", async () => {
    // Catches a length check written as greater than 32 instead of at least 32.
    expect(EXACT_SECRET).toHaveLength(32);
    setEnv({ ...CONFIGURED, CONSOLE_SESSION_SECRET: EXACT_SECRET });
    await expectReady("32 characters");
  });

  it("neither probe body carries a configured value", async () => {
    // Catches a probe that echoes a value, names the failing variable, or puts either in a header.
    const failing: [string, Env][] = [
      ["all set", SENTINELS],
      ["secret short", { ...SENTINELS, CONSOLE_SESSION_SECRET: "sentinel-short-2e6a" }],
      ["url ftp", { ...SENTINELS, CONSOLE_API_URL: "ftp://sentinel-ftp-8d1b" }],
      ["hash unset", { ...SENTINELS, CONSOLE_PASSWORD_SHA256: undefined }],
    ];
    for (const [label, env] of failing) {
      setEnv(env);
      for (const [name, handler, path] of [
        ["healthz", healthz, "/healthz"],
        ["readyz", readyz, "/readyz"],
      ] as const) {
        const response = await handler(probe(path));
        const text = await response.text();
        const headers = [...response.headers.entries()].map(([key, value]) => `${key}: ${value}`).join("\n");
        for (const value of Object.values(env)) {
          if (value === undefined) continue;
          expect(text, `${label} ${name} body`).not.toContain(value);
          expect(headers, `${label} ${name} headers`).not.toContain(value);
        }
        expect(text, `${label} ${name} body`).not.toContain("CONSOLE_");
        expect(headers, `${label} ${name} headers`).not.toContain("CONSOLE_");
      }
    }
  });

  it("neither probe makes a fetch", async () => {
    // Catches a readyz that pings console-api, or a healthz that checks anything over the network.
    for (const env of [CONFIGURED, { ...CONFIGURED, CONSOLE_ADMIN_KEY: undefined }]) {
      setEnv(env);
      await healthz(probe("/healthz"));
      await readyz(probe("/readyz"));
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("healthz and readyz pass the middleware without a cookie and healthz x redirects", async () => {
    // Catches probes left behind the session check (the kubelet reads a 307 and never marks the pod ready) or exempted by prefix.
    vi.stubEnv("CONSOLE_SESSION_SECRET", undefined);
    for (const path of ["/healthz", "/readyz"]) {
      const response = await middleware(probe(path));
      expect(response.headers.get("x-middleware-next"), path).toBe("1");
      expect(response.headers.get("location"), path).toBeNull();
    }

    vi.stubEnv("CONSOLE_SESSION_SECRET", EXACT_SECRET);
    for (const path of ["/healthz/x", "/readyz/x", "/healthzx"]) {
      const response = await middleware(probe(path));
      expect(response.status, path).toBe(307);
      expect(response.headers.get("location"), path).toBe(`${ORIGIN}/login?next=${encodeURIComponent(path)}`);
    }
  });
});
