import { checkPassword } from "@/lib/login";
import { safeNext } from "@/lib/nav";
import { SESSION_COOKIE, SESSION_TTL_SECONDS, readSessionSecret, signSession } from "@/lib/session";

function seeOther(location: string, cookie?: string): Response {
  const headers = new Headers({ Location: location });
  if (cookie) {
    headers.set("Set-Cookie", cookie);
  }
  return new Response(null, { status: 303, headers });
}

/** Every refusal looks the same to the caller; the password is never logged. */
export async function POST(request: Request): Promise<Response> {
  const refused = seeOther("/login?error=1");
  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return refused;
  }
  const secret = readSessionSecret();
  if (secret === null) {
    return refused;
  }
  const password = form.get("password");
  if (typeof password !== "string" || !checkPassword(password, process.env.CONSOLE_PASSWORD_SHA256)) {
    return refused;
  }
  const next = form.get("next");
  const expires = Math.floor(Date.now() / 1000) + SESSION_TTL_SECONDS;
  const value = await signSession(secret, expires);
  return seeOther(
    safeNext(typeof next === "string" ? next : null),
    `${SESSION_COOKIE}=${value}; Path=/; Max-Age=${SESSION_TTL_SECONDS}; HttpOnly; Secure; SameSite=Strict`,
  );
}
