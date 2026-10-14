import { NextRequest, NextResponse } from "next/server";

import { SESSION_COOKIE, readSessionSecret, verifySession } from "@/lib/session";

const EXEMPT = new Set(["/login", "/api/login", "/api/logout", "/healthz", "/readyz"]);

/** Every non-exempt path needs a valid session. Pinned by tests/middleware.test.ts. */
export async function middleware(request: NextRequest): Promise<NextResponse> {
  const { pathname } = request.nextUrl;
  if (EXEMPT.has(pathname) || pathname.startsWith("/_next/")) {
    return NextResponse.next();
  }
  const cookie = request.cookies.get(SESSION_COOKIE)?.value;
  const now = Math.floor(Date.now() / 1000);
  if (await verifySession(cookie, readSessionSecret(), now)) {
    return NextResponse.next();
  }
  const login = new URL(`/login?next=${encodeURIComponent(pathname)}`, request.url);
  return NextResponse.redirect(login, 307);
}

export const config = { matcher: ["/((?!_next/).*)"] };
