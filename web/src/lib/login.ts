import { createHash, timingSafeEqual } from "node:crypto";

export const MIN_PASSWORD_LENGTH = 1;
export const MAX_PASSWORD_LENGTH = 256;

const SHA256_HEX_LENGTH = 64;

export function passwordLengthOk(password: string): boolean {
  return password.length >= MIN_PASSWORD_LENGTH && password.length <= MAX_PASSWORD_LENGTH;
}

/**
 * Fails closed when no hash is configured. The length check runs before any
 * hashing. Pinned by tests/login-route.test.ts.
 */
export function checkPassword(presented: string, expectedSha256Hex: string | undefined): boolean {
  if (!expectedSha256Hex || expectedSha256Hex.length !== SHA256_HEX_LENGTH) {
    return false;
  }
  if (!passwordLengthOk(presented)) {
    return false;
  }
  const actual = Buffer.from(createHash("sha256").update(presented, "utf8").digest("hex"), "utf8");
  const expected = Buffer.from(expectedSha256Hex.toLowerCase(), "utf8");
  return actual.length === expected.length && timingSafeEqual(actual, expected);
}
