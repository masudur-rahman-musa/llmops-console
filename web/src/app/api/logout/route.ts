import { SESSION_COOKIE } from "@/lib/session";

export async function POST(): Promise<Response> {
  return new Response(null, {
    status: 303,
    headers: {
      Location: "/login",
      "Set-Cookie": `${SESSION_COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict`,
    },
  });
}
