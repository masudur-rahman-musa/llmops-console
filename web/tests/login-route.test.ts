import { createHash, createHmac } from "node:crypto";

import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest";

import { POST as login } from "@/app/api/login/route";
import { POST as logout } from "@/app/api/logout/route";
import { MAX_PASSWORD_LENGTH, MIN_PASSWORD_LENGTH } from "@/lib/login";
import { safeNext } from "@/lib/nav";
import { readSessionSecret } from "@/lib/session";

// createHash stays real; the spy only records whether the handler reached hashing.
vi.mock("node:crypto", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:crypto")>();
  return { ...actual, createHash: vi.fn(actual.createHash) };
});

const ORIGIN = "http://console.test";
// 32 characters exactly: the smallest usable secret.
const SECRET = "0123456789abcdef0123456789abcdef";
const PASSWORD = "Pw-marker-7f3a9c-correct-horse";
const NOW_MS = Date.UTC(2026, 8, 16, 12, 0, 0);
const NOW = Math.floor(NOW_MS / 1000);
const NAV_PATHS = ["/", "/cost", "/quality", "/residency", "/reliability", "/incidents", "/releases"];

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function post(fields: Record<string, string>): Request {
  return new Request(`${ORIGIN}/api/login`, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(fields).toString(),
  });
}

// Resolves a Location header the way a browser would, so an absolute same-origin
// redirect and a relative one compare equal while an off-origin one does not.
function target(response: Response): string {
  const location = response.headers.get("location");
  expect(location).not.toBeNull();
  return new URL(location ?? "", `${ORIGIN}/api/login`).href;
}

type ParsedCookie = { name: string; value: string; attributes: Map<string, string> };

function onlyCookie(response: Response): ParsedCookie {
  const cookies = response.headers.getSetCookie();
  expect(cookies).toHaveLength(1);
  const [pair, ...rest] = cookies[0].split(";").map((part) => part.trim());
  const eq = pair.indexOf("=");
  const attributes = new Map<string, string>();
  for (const attribute of rest) {
    const at = attribute.indexOf("=");
    const key = (at === -1 ? attribute : attribute.slice(0, at)).toLowerCase();
    attributes.set(key, at === -1 ? "" : attribute.slice(at + 1));
  }
  return { name: pair.slice(0, eq), value: pair.slice(eq + 1), attributes };
}

function expectRefused(response: Response): void {
  expect(response.status).toBe(303);
  expect(target(response)).toBe(`${ORIGIN}/login?error=1`);
  expect(response.headers.getSetCookie()).toEqual([]);
  expect(response.headers.get("set-cookie")).toBeNull();
}

function expectSignedIn(response: Response, path: string): void {
  expect(response.status).toBe(303);
  expect(target(response)).toBe(`${ORIGIN}${path}`);
  expect(onlyCookie(response).name).toBe("opsconsole_session");
}

let hashSpy: MockInstance;
let errorSpy: MockInstance;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(NOW_MS);
  vi.stubEnv("CONSOLE_SESSION_SECRET", SECRET);
  vi.stubEnv("CONSOLE_PASSWORD_SHA256", sha256(PASSWORD));
  hashSpy = vi.mocked(createHash);
  hashSpy.mockClear();
  errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  errorSpy.mockRestore();
});

describe("POST /api/login", () => {
  it("the right password is 303 with the session cookie attributes", async () => {
    // Catches a cookie missing HttpOnly, Secure, SameSite=Strict, Path=/ or Max-Age=43200, or a value not signed as v1.<now+43200>.<hmac>.
    const response = await login(post({ password: PASSWORD, next: "/cost" }));
    expectSignedIn(response, "/cost");

    const expires = NOW + 43_200;
    const session = onlyCookie(response);
    expect(session.value).toBe(`v1.${expires}.${createHmac("sha256", SECRET).update(`v1.${expires}`).digest("hex")}`);
    expect(session.attributes.get("path")).toBe("/");
    expect(session.attributes.get("max-age")).toBe("43200");
    expect(session.attributes.has("httponly")).toBe(true);
    expect(session.attributes.has("secure")).toBe(true);
    expect(session.attributes.get("samesite")?.toLowerCase()).toBe("strict");

    expectSignedIn(await login(post({ password: PASSWORD })), "/");
  });

  it("a wrong password is 303 to login error and sets no cookie", async () => {
    // Catches a handler that signs in without comparing the password, or refuses with a cookie still set.
    for (const wrong of ["Pw-marker-7f3a9c-correct-hors", `${PASSWORD} `, PASSWORD.toUpperCase(), sha256(PASSWORD)]) {
      expectRefused(await login(post({ password: wrong, next: "/cost" })));
    }
  });

  it("an unset password hash refuses every login", async () => {
    // Catches a handler that fails open when CONSOLE_PASSWORD_SHA256 is unset or empty.
    for (const unset of [undefined, ""]) {
      vi.stubEnv("CONSOLE_PASSWORD_SHA256", unset);
      for (const password of [PASSWORD, "x", "a".repeat(256)]) {
        expectRefused(await login(post({ password, next: "/" })));
      }
    }
  });

  it("an unset or short secret refuses every login and logs an error", async () => {
    // Catches a handler that signs with a missing or 31-character secret, or refuses without logging an ERROR; the password here is correct.
    for (const secret of [undefined, "", SECRET.slice(0, 31)]) {
      // The ERROR logs once per bad spell, so a good read first makes each case a new spell.
      readSessionSecret({ CONSOLE_SESSION_SECRET: SECRET });
      vi.stubEnv("CONSOLE_SESSION_SECRET", secret);
      errorSpy.mockClear();
      const response = await login(post({ password: PASSWORD, next: "/cost" }));
      expectRefused(response);
      expect(errorSpy).toHaveBeenCalled();
      const logged = errorSpy.mock.calls.map((call) => call.map(String).join(" "));
      expect(logged.some((line) => line.startsWith("ERROR"))).toBe(true);
      if (secret) {
        expect(logged.join("\n")).not.toContain(secret);
      }
    }
  });

  it("a password of 256 is accepted and 257 is refused before hashing", async () => {
    // Catches a maximum password length other than 256, or a length check placed after hashing; the 257 row's hash is configured, so only the cap refuses it.
    expect(MIN_PASSWORD_LENGTH).toBe(1);
    expect(MAX_PASSWORD_LENGTH).toBe(256);

    const at = "a".repeat(256);
    vi.stubEnv("CONSOLE_PASSWORD_SHA256", sha256(at));
    hashSpy.mockClear();
    expectSignedIn(await login(post({ password: at, next: "/" })), "/");
    expect(hashSpy).toHaveBeenCalled();

    const over = "a".repeat(257);
    vi.stubEnv("CONSOLE_PASSWORD_SHA256", sha256(over));
    hashSpy.mockClear();
    expectRefused(await login(post({ password: over, next: "/" })));
    expect(hashSpy).not.toHaveBeenCalled();
  });

  it("an empty password is refused", async () => {
    // Catches a missing minimum length; the configured hash is sha256 of the empty string, so only the length check refuses.
    vi.stubEnv("CONSOLE_PASSWORD_SHA256", sha256(""));
    const rows: Array<Record<string, string>> = [{ password: "", next: "/" }, { next: "/" }];
    for (const fields of rows) {
      hashSpy.mockClear();
      expectRefused(await login(post(fields)));
      expect(hashSpy).not.toHaveBeenCalled();
    }
  });

  it("next accepts each of the seven nav paths", async () => {
    // Catches an allowlist missing one of the seven nav paths, or a handler that always redirects to root.
    for (const path of NAV_PATHS) {
      expect(safeNext(path)).toBe(path);
      expectSignedIn(await login(post({ password: PASSWORD, next: path })), path);
    }
  });

  it("the next rejection table falls back to root", async () => {
    // Catches an open redirect: a prefix, normalising or URL-parsing check that lets any value other than the seven exact paths through.
    const rows = [
      "//evil.example",
      "https://evil.example",
      "/\\evil.example",
      "/cost\n",
      "/cost ",
      "/unknown",
      "/cost\r\n",
      " /cost",
      "/cost/",
      "/COST",
      "/cost?x=1",
      "/cost#x",
      "%2Fcost",
      "/%63ost",
      "cost",
      "",
    ];
    for (const next of rows) {
      expect(safeNext(next), JSON.stringify(next)).toBe("/");
      expectSignedIn(await login(post({ password: PASSWORD, next })), "/");
    }
    expect(safeNext(null)).toBe("/");
    expect(safeNext(undefined)).toBe("/");
  });

  it("the password never appears in the response or a log line", async () => {
    // Catches a handler that logs the form, or echoes the presented password into a header, the Location or the body.
    const spies = (["log", "info", "warn", "debug"] as const).map((level) =>
      vi.spyOn(console, level).mockImplementation(() => undefined),
    );
    const allSpies: MockInstance[] = [...spies, errorSpy];
    const long = `${PASSWORD}${"z".repeat(300)}`;
    const cases: Array<{ password: string; secret: string | undefined }> = [
      { password: PASSWORD, secret: SECRET },
      { password: `${PASSWORD}-wrong`, secret: SECRET },
      { password: long, secret: SECRET },
      { password: PASSWORD, secret: undefined },
    ];
    try {
      for (const { password, secret } of cases) {
        vi.stubEnv("CONSOLE_SESSION_SECRET", secret);
        const response = await login(post({ password, next: "/cost" }));
        const seen: string[] = [await response.text()];
        response.headers.forEach((value: string, name: string) => seen.push(`${name}: ${value}`));
        seen.push(...response.headers.getSetCookie());
        for (const spy of allSpies) {
          for (const call of spy.mock.calls) {
            seen.push(call.map((arg) => (typeof arg === "string" ? arg : JSON.stringify(arg) ?? String(arg))).join(" "));
          }
        }
        const joined = seen.join("\n");
        expect(joined).not.toContain(PASSWORD);
        expect(joined).not.toContain(encodeURIComponent(PASSWORD));
      }
    } finally {
      for (const spy of spies) spy.mockRestore();
    }
  });

  it("logout is 303 to login and clears the cookie", async () => {
    // Catches a logout that leaves the session cookie alive: wrong name, a Path other than /, or a Max-Age other than 0.
    const response = await logout();
    expect(response.status).toBe(303);
    expect(target(response)).toBe(`${ORIGIN}/login`);
    const cleared = onlyCookie(response);
    expect(cleared.name).toBe("opsconsole_session");
    expect(cleared.value).toBe("");
    expect(cleared.attributes.get("path")).toBe("/");
    expect(cleared.attributes.get("max-age")).toBe("0");
    expect(cleared.attributes.has("httponly")).toBe(true);
    expect(cleared.attributes.has("secure")).toBe(true);
    expect(cleared.attributes.get("samesite")?.toLowerCase()).toBe("strict");
  });
});
