# Console

The web console for the AI Operations Console. It shows whether the AI is
working and what it costs: six read-only pages (home, cost, quality,
reliability, incidents, releases) behind a shared password login. Each page
opens with one line saying what question it answers.

It is a Next.js 15 app (TypeScript, the app router, server components). Every
page reads through `console-api` on the server only; the browser never talks to
a database or to Prometheus.

## Run and test

Install the workspace dependencies first (this folder is part of the repo's
pnpm-based workspace). Then, from the repo root:

- `pnpm -C web dev` — run the console in watch mode on port 3001.
- `pnpm -C web build` — build the production bundle.
- `pnpm -C web start` — serve the built bundle on port 3001.
- `pnpm -C web lint` — run eslint.
- `pnpm -C web typecheck` — run `tsc --noEmit`.
- `pnpm -C web test` — run the vitest suite.

## Settings

The console reads these environment variables (see `.env.example`). The login
pair comes first; the page client comes second.

- `CONSOLE_PASSWORD_SHA256` — the lowercase hex SHA-256 of the shared login
  password. Unset means every login is refused.
- `CONSOLE_SESSION_SECRET` — signs the session cookie; needs 32 characters or
  more. Rotating it logs everyone out.
- `CONSOLE_API_URL` — the base URL of console-api. Unset means every page shows
  "Console data unavailable (unconfigured)."
- `CONSOLE_ADMIN_KEY` — the bearer key sent to console-api. It is never logged.
- `CONSOLE_TRADEOFF_PATH` — the file the cost page reads for the model tradeoff
  table. Unset means that table shows "Not measured yet."
