import { safeNext } from "@/lib/nav";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string; error?: string }>;
}) {
  const { next, error } = await searchParams;
  return (
    <main>
      <h1>Sign in</h1>
      {error === "1" ? <p role="alert">Sign-in failed.</p> : null}
      <form method="post" action="/api/login">
        <input type="hidden" name="next" value={safeNext(next)} />
        <label>
          Password
          <input type="password" name="password" required minLength={1} maxLength={256} autoComplete="current-password" />
        </label>
        <button type="submit">Sign in</button>
      </form>
    </main>
  );
}
