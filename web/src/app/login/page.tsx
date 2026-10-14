import { safeNext } from "@/lib/nav";

import styles from "./login.module.css";

export default async function LoginPage({
  searchParams,
}: {
  searchParams: Promise<{ next?: string; error?: string }>;
}) {
  const { next, error } = await searchParams;
  return (
    <main className={styles.page}>
      <div className={styles.card}>
        <h1>Sign in</h1>
        {error === "1" ? (
          <p role="alert" className="alert">
            Sign-in failed.
          </p>
        ) : null}
        <form method="post" action="/api/login" className={styles.form}>
          <input type="hidden" name="next" value={safeNext(next)} />
          <label className={styles.label}>
            Password
            <input type="password" name="password" required minLength={1} maxLength={256} autoComplete="current-password" />
          </label>
          <button type="submit" className={styles.submit}>
            Sign in
          </button>
        </form>
      </div>
    </main>
  );
}
