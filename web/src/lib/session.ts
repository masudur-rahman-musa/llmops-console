/**
 * The console session: a stateless cookie `v1.<expires_unix>.<hmac_sha256_hex>`.
 *
 * Web Crypto only, so the same module runs in the edge middleware and in the
 * node route handlers. Rotating `CONSOLE_SESSION_SECRET` revokes every session
 * at once; there is no per-session revoke.
 *
 * Pinned by tests/session.test.ts.
 */

export const SESSION_COOKIE = "opsconsole_session";
export const SESSION_TTL_SECONDS = 43200;
export const MIN_SECRET_LENGTH = 32;

const VERSION = "v1";
const DIGITS = /^[0-9]+$/;

export function secretIsUsable(secret: string | undefined | null): secret is string {
  return typeof secret === "string" && secret.length >= MIN_SECRET_LENGTH;
}

// The middleware reads the secret on every request, so a steadily bad secret
// would log the same ERROR each time. This flag holds whether the last read was
// bad: the ERROR fires only when a bad read follows a good one (or the first
// read), so one bad spell logs once. A good read clears it, so a later bad spell
// logs again. Pinned by tests/session.test.ts "a steadily bad secret logs once".
let lastReadWasBad = false;

/** Fails closed: an unset or short secret yields no secret, and logs an ERROR once per bad spell. */
export function readSessionSecret(
  env: Record<string, string | undefined> = process.env,
): string | null {
  const secret = env.CONSOLE_SESSION_SECRET;
  if (secretIsUsable(secret)) {
    lastReadWasBad = false;
    return secret;
  }
  if (!lastReadWasBad) {
    console.error(
      `ERROR CONSOLE_SESSION_SECRET is unset or shorter than ${MIN_SECRET_LENGTH} characters; every login is refused`,
    );
  }
  lastReadWasBad = true;
  return null;
}

async function mac(secret: string, payload: string): Promise<string> {
  const encoder = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    encoder.encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(payload));
  return Array.from(new Uint8Array(signature), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

/** Constant-time over equal lengths; the lengths themselves are public (64 hex). */
function sameText(a: string, b: string): boolean {
  if (a.length !== b.length) {
    return false;
  }
  let difference = 0;
  for (let i = 0; i < a.length; i += 1) {
    difference |= a.charCodeAt(i) ^ b.charCodeAt(i);
  }
  return difference === 0;
}

export async function signSession(secret: string, expiresUnix: number): Promise<string> {
  if (!secretIsUsable(secret)) {
    throw new Error(`session secret must be at least ${MIN_SECRET_LENGTH} characters`);
  }
  const payload = `${VERSION}.${expiresUnix}`;
  return `${payload}.${await mac(secret, payload)}`;
}

export async function verifySession(
  value: string | undefined,
  secret: string | null | undefined,
  nowUnix: number,
): Promise<boolean> {
  if (!secretIsUsable(secret) || !value) {
    return false;
  }
  const parts = value.split(".");
  if (parts.length !== 3) {
    return false;
  }
  const [version, expires, presented] = parts;
  if (version !== VERSION || !DIGITS.test(expires)) {
    return false;
  }
  if (!sameText(presented, await mac(secret, `${version}.${expires}`))) {
    return false;
  }
  return nowUnix < Number(expires);
}
