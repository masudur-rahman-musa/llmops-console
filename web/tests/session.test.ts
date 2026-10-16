import { createHmac } from "node:crypto";

import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MIN_SECRET_LENGTH,
  SESSION_COOKIE,
  SESSION_TTL_SECONDS,
  readSessionSecret,
  secretIsUsable,
  signSession,
  verifySession,
} from "@/lib/session";

// 32 characters exactly: the smallest secret the contract accepts.
const SECRET = "0123456789abcdef0123456789abcdef";
const EXPIRES = 1_900_000_000;

// Computed with node:crypto, independently of the Web Crypto implementation.
function mac(secret: string, payload: string): string {
  return createHmac("sha256", secret).update(payload).digest("hex");
}

function cookie(secret: string, expires: string, version = "v1"): string {
  return `${version}.${expires}.${mac(secret, `${version}.${expires}`)}`;
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("session cookie", () => {
  it("signs and verifies a round trip", async () => {
    // Catches a signer whose MAC is not HMAC-SHA256 over "v1.<expires>" in lowercase hex, or a verifier that refuses its own signer's output.
    const value = await signSession(SECRET, EXPIRES);
    expect(value).toBe(`v1.${EXPIRES}.${mac(SECRET, `v1.${EXPIRES}`)}`);
    expect(await verifySession(value, SECRET, EXPIRES - 60)).toBe(true);
  });

  it("refuses a tampered mac", async () => {
    // Catches a verifier that checks shape and expiry but never compares the MAC; every row is well-formed hex of the right length.
    const good = cookie(SECRET, String(EXPIRES));
    expect(await verifySession(good, SECRET, EXPIRES - 60)).toBe(true);

    const lastHex = good.slice(-1);
    const flipped = good.slice(0, -1) + (lastHex === "0" ? "1" : "0");
    const otherSecret = cookie("ffffffffffffffffffffffffffffffff", String(EXPIRES));
    const zeros = `v1.${EXPIRES}.${"0".repeat(64)}`;
    for (const tampered of [flipped, otherSecret, zeros]) {
      expect(await verifySession(tampered, SECRET, EXPIRES - 60), tampered).toBe(false);
    }
  });

  it("refuses a tampered expiry", async () => {
    // Catches a MAC computed over something other than the expiry, so a client could extend its own session.
    const good = cookie(SECRET, String(EXPIRES));
    const [version, , signature] = good.split(".");
    const extended = `${version}.${EXPIRES + 86_400}.${signature}`;
    expect(await verifySession(extended, SECRET, EXPIRES - 60)).toBe(false);
    // The same MAC on an expiry that has not been tampered with is still accepted.
    expect(await verifySession(good, SECRET, EXPIRES - 60)).toBe(true);
  });

  it("a cookie at expires minus one is valid and at expires is not", async () => {
    // Catches `now <= expires` in place of `now < expires`, and a verifier that ignores the expiry.
    const value = cookie(SECRET, String(EXPIRES));
    expect(await verifySession(value, SECRET, EXPIRES - 1)).toBe(true);
    expect(await verifySession(value, SECRET, EXPIRES)).toBe(false);
    expect(await verifySession(value, SECRET, EXPIRES + 1)).toBe(false);
  });

  it("refuses a wrong version prefix", async () => {
    // Catches a verifier that MACs whatever version is presented and never requires `v1`; each row's MAC is valid for its own prefix.
    for (const version of ["v2", "V1", "v0", "", "v1 "]) {
      const value = cookie(SECRET, String(EXPIRES), version);
      expect(await verifySession(value, SECRET, EXPIRES - 60), JSON.stringify(value)).toBe(false);
    }
  });

  it("refuses malformed values", async () => {
    // Catches a verifier that destructures the first three segments and ignores extras, or parses the expiry with Number() instead of requiring digits.
    const good = cookie(SECRET, String(EXPIRES));
    const rows: Array<string | undefined> = [
      undefined,
      "",
      "v1",
      `v1.${EXPIRES}`,
      `${good}.extra`,
      `${good}.`,
      // Expiry rows carry a MAC valid for their own payload, so only the digits check refuses them.
      cookie(SECRET, "1e10"),
      cookie(SECRET, "0x7fffffff"),
      cookie(SECRET, "+1900000000"),
      cookie(SECRET, " 1900000000"),
      cookie(SECRET, "1900000000.5"),
      cookie(SECRET, "Infinity"),
    ];
    for (const value of rows) {
      expect(await verifySession(value, SECRET, EXPIRES - 60), JSON.stringify(value)).toBe(false);
    }
  });

  it("a secret of 32 characters works and 31 is refused", async () => {
    // Catches a minimum secret length other than 32, or a verifier and reader that accept a short secret.
    expect(MIN_SECRET_LENGTH).toBe(32);
    const short = SECRET.slice(0, 31);
    expect(short).toHaveLength(31);

    expect(secretIsUsable(SECRET)).toBe(true);
    expect(secretIsUsable(short)).toBe(false);
    expect(secretIsUsable(undefined)).toBe(false);
    expect(secretIsUsable("")).toBe(false);

    const signed = await signSession(SECRET, EXPIRES);
    expect(await verifySession(signed, SECRET, EXPIRES - 60)).toBe(true);
    await expect(signSession(short, EXPIRES)).rejects.toThrow();
    // A cookie genuinely signed with the short secret is still refused.
    expect(await verifySession(cookie(short, String(EXPIRES)), short, EXPIRES - 60)).toBe(false);
    expect(await verifySession(signed, null, EXPIRES - 60)).toBe(false);
    expect(await verifySession(signed, undefined, EXPIRES - 60)).toBe(false);

    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    // A good read first leaves the "last read was good" state, so the bad read below is the first bad read and must log.
    expect(readSessionSecret({ CONSOLE_SESSION_SECRET: SECRET })).toBe(SECRET);
    expect(error).not.toHaveBeenCalled();

    expect(readSessionSecret({ CONSOLE_SESSION_SECRET: short })).toBeNull();
    expect(error).toHaveBeenCalledTimes(1);
    const logged = error.mock.calls[0].map(String).join(" ");
    expect(logged.startsWith("ERROR")).toBe(true);
    expect(logged).not.toContain(short);
  });

  it("a steadily bad secret logs once", () => {
    // Catches a reader that logs on every bad read instead of once per bad spell, or one that never logs again after a good read heals it.
    const error = vi.spyOn(console, "error").mockImplementation(() => undefined);
    // Establish a known "last read was good" state, then measure only the bad reads that follow.
    expect(readSessionSecret({ CONSOLE_SESSION_SECRET: SECRET })).toBe(SECRET);
    error.mockClear();

    for (let i = 0; i < 3; i += 1) {
      expect(readSessionSecret({})).toBeNull();
    }
    expect(error).toHaveBeenCalledTimes(1);

    // A good read clears the flag, so the next bad read starts a fresh spell and logs again.
    expect(readSessionSecret({ CONSOLE_SESSION_SECRET: SECRET })).toBe(SECRET);
    expect(readSessionSecret({})).toBeNull();
    expect(error).toHaveBeenCalledTimes(2);
  });

  it("the ttl is 43200 seconds", () => {
    // Catches a session lifetime or cookie name other than the plan's 43200 seconds and `opsconsole_session`.
    expect(SESSION_TTL_SECONDS).toBe(43_200);
    expect(SESSION_COOKIE).toBe("opsconsole_session");
  });
});
