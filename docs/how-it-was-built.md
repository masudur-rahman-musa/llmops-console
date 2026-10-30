# How this was built

This project was built with AI coding agents doing much of the typing, under strict
tests and review. The agents wrote code. The checks decided whether it stayed.

## The loop

Every change, a feature or a fix, went through the same steps.

1. **Say what done means.** Before any code, write down what the change must do,
   which files it touches, and what it must refuse or how it must fail safely.
2. **Write the tests first.** One test for each of those points. They fail at
   first, because the code does not exist yet.
3. **Make the smallest change that turns them green.** Touch nothing else.
4. **Run the checks.** `make verify` runs the same checks CI runs, so a green run on
   the laptop means a green run in CI.
5. **Review.** A fresh read of the diff, by a reviewer that did not write it,
   before the change is called done.
6. **Commit** with a plain message that says what the change does.

Two things make this work with an agent. The meaning of "done" is written before
the code, so the person and the agent agree on it. And the tests come from that
written meaning, not from the code, so they check what was asked for rather than
what happened to get built.

## What `make verify` runs

- `format`, `lint` and `typecheck`, which are ruff, mypy in strict mode, eslint and
  `tsc`.
- `test`, which is every backend, infra and console test, with Postgres, Redis and
  Qdrant running, an 80% coverage floor, and 100% on routing, limits and input
  validation.
- `security`, which is gitleaks over the whole git history, `pip-audit` and
  `pnpm audit`.

CI calls the same `make` targets, so the laptop and CI cannot disagree about what
passing means.

## Guard tests on the repo itself

Some tests check the repo rather than the app. They catch the mistakes an agent
makes most often.

- Every path a doc names must exist.
- The README's redaction table must match what the redaction report script
  produces.
- Comments and docs must not point at private notes or task numbers.
- The root folder holds only an allowed list of files.
- Every route must sit behind an auth check or be on a short open list.

## When it went wrong

It did go wrong. For eight days only each change's own tests were run, and five
tests on `main` went red without anyone seeing. The blameless write-up is in
[postmortems/2026-09-17-red-tests-went-unseen.md](postmortems/2026-09-17-red-tests-went-unseen.md).
The fix was to run the whole suite again, cap how many changes may land between
whole-suite runs, and read CI after every push.

## What is not in this repository

The working notes and task lists are not part of the current tree. What is here is
the result. That means the code, the tests, the docs and the commits.
