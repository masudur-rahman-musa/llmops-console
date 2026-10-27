import { SESSION_COOKIE, SESSION_TTL_SECONDS, signSession } from "@/lib/session";

import { findJunk } from "../junk";

const TABS = ["/", "/requests", "/cost", "/quality"];

/**
 * Signs in with a session cookie of its own (the plain password is never kept, only
 * its hash), fetches every tab and returns one failure per bad thing, each starting
 * with the page: `/requests: long decimal: 13.0128054`. A redirect is not followed,
 * so a refused cookie shows as a failure and not as a clean login page.
 * Pinned by tests/console-check.test.ts.
 */
export async function checkConsole({
  baseUrl,
  secret,
  fetchFn = fetch,
}: {
  baseUrl: string;
  secret: string;
  fetchFn?: typeof fetch;
}): Promise<string[]> {
  const expires = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  const cookie = `${SESSION_COOKIE}=${await signSession(secret, expires)}`;
  const failures: string[] = [];
  for (const path of TABS) {
    let response: Response;
    try {
      response = await fetchFn(new URL(path, baseUrl), { headers: { cookie }, redirect: "manual" });
    } catch (error) {
      failures.push(`${path}: cannot reach ${baseUrl} (${error instanceof Error ? error.message : String(error)})`);
      continue;
    }
    if (response.status >= 300 && response.status < 400) {
      failures.push(`${path}: redirected to ${response.headers.get("location") ?? "somewhere"}`);
      continue;
    }
    if (response.status !== 200) {
      failures.push(`${path}: status ${response.status}`);
      continue;
    }
    for (const junk of findJunk(await response.text())) failures.push(`${path}: ${junk}`);
  }
  return failures;
}
