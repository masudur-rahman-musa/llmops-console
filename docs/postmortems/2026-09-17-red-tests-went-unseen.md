# Postmortem. Failing tests on main went unseen because the whole suite never ran

**Date:** 2026-09-17 · **Severity:** sev3 (no deployed system affected) · **Duration:** open since 2026-09-09 13:27 UTC · **Author:** Masudur Rahman

## Summary

From 2026-09-09 on, changes to this repo were checked by running only the tests
each change wrote, never the whole suite. Tests that other changes broke went red
on `main` with nobody running them. A wider run on 2026-09-11 found the first one,
about 44 hours after the commit that broke it. By 2026-09-16 five tests were known
to fail on `main`, and the changes since the last whole-suite run had never been
checked against each other.

Every fact below comes from this repo's git history. Commit times are in UTC.

## Impact

- Nothing deployed was affected. No manifest in this repo had been applied to a
  real cluster, so no user and no tenant saw a failure.
- Five tests were failing on `main`. One was in `backend/tests/evals/test_report.py`,
  one in `infra/tests/infra/test_tradeoff_run.py`, and one each in
  `backend/tests/api/test_cost_wiring.py`,
  `backend/tests/api/test_rate_limit_enforcement.py` and
  `backend/tests/api/test_rehydration.py`. The paths are today's. At the time the
  backend tests sat in a top-level `tests/` folder.
- A red test stays red here. A test is never edited to make it pass, so each one
  waited for a real fix, and every change made in the meantime was checked against
  a suite that was already failing.
- All five failures reached the shared branch. Every commit named in the timeline
  up to `2b12746` is an ancestor of `571e2c7`, which was `origin/main` then.

## Timeline (UTC)

- 2026-09-09 12:37 `8216af8` is the last change known to have run the whole suite.
- 2026-09-09 13:27 `cb25f5b` adds an LLM judge to the eval report. From here each
  case in the report carries a `judge` key that an exact key-set check in the
  report tests does not list, and that test fails.
- 2026-09-11 09:27 `c3bf64d` records a run wider than one change's own tests (716
  passed, 1 failed). It finds that failure. Detection took about 44 hours.
- 2026-09-12 22:45 `5604279` lands the nightly eval. The note made next says three
  failing tests predate it.
- 2026-09-13 02:36 `ee01d0f` records those three failures. They are a cache header
  in `test_cost_wiring.py`, a 429 in `test_rate_limit_enforcement.py`, and
  `KeyError: 'choices'` in `test_rehydration.py`. The commit that broke each of the
  three is not known.
- 2026-09-16 01:37 `69c7f2c` adds the tradeoff measurement harness with a test that
  bans `..` in a report path whose only legal form contains `..`. It is red from
  that commit.
- 2026-09-16 01:38 `2b12746` records that failure one minute later.
- 2026-09-17 `c14ece0`. At this commit sixty changes had never been checked against
  the whole suite, and all five failures were still open. This postmortem is
  written.

## Root cause

The check before each commit ran only the tests the change itself wrote. That
proves the change does what it claims. It cannot see a test in another file that
the change breaks, and it cannot see a test written wrongly in a way only other code
would show. The whole suite, the only check that sees both, did not run once in
eight days.

## Contributing factors

- The fast check was meant for short stretches over well-understood code. It stayed
  on for every change from 2026-09-09, new subsystems included.
- CI runs the suite on every push to `main`, but nobody read those results before
  starting the next change, and nothing recorded whether they were read.
- Failures were written down as known issues instead of blocking the next change,
  so the list grew instead of shrinking.
- One test (`test_tradeoff_run.py`) added a rule its own requirements did not
  state, which made it impossible to pass from the day it was written.

## What went well

- Every failure was written down on the day it was found, with the test name and
  its error, which is why this timeline can be built from git history alone.
- The list of changes never checked against the whole suite was kept current, so
  the size of the gap was always visible.
- None of the failures reached a running system.

## Action items (owner, due date)

- Run the whole suite on `main`, fix or deliberately record each of the five
  failures, and clear the list of unchecked changes. Owner is Masudur Rahman, due 2026-09-20.
- Cap the number of changes allowed between whole-suite runs, and make the check
  before each commit refuse a new change once the cap is reached. Owner is Masudur Rahman, due 2026-09-24.
- Read the CI result of every push to `main` before the next change starts. Owner is Masudur Rahman, due 2026-09-20.
- Record this incident on the console timeline. Dropped, because the console no
  longer has an incident timeline, and this postmortem is the record.

Blameless. Written within 48h of the latest failure being recorded. Linked from the README.
