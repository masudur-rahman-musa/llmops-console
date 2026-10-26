# Postmortem: failing tests on main went unseen because the whole suite never ran

**Date:** 2026-09-17 · **Severity:** sev3 (no deployed system affected) · **Duration:** open since 2026-09-09 13:27 UTC · **Author:** Masudur Rahman

## Summary

From 2026-09-09 onward, changes to this repository were checked by running only the
tests each change wrote, and never the whole test suite. Tests that other changes
broke therefore went red on `main` without anyone running them. A wider run on
2026-09-11 found the first one about 44 hours after the commit that broke it. By
2026-09-16 five tests were known to fail at `main`, and the changes since the last
whole-suite run had never been checked against each other at all.

Every fact below comes from the git history of this repository. Commit times are
converted to UTC.

## Impact

- Nothing deployed was affected: no manifest in this repository has been applied
  to a real cluster, so no user and no tenant saw a failure.
- Five tests are recorded as failing at `main`: one in `tests/evals/test_report.py`,
  one in `infra/tests/infra/test_tradeoff_run.py`, and one each in `test_cost_wiring.py`,
  `test_rate_limit_enforcement.py` and `test_rehydration.py`.
- A red test stays red here. The project's rules forbid editing a test to make it
  pass, so each one waits for a deliberate fix, and every change made meanwhile
  is checked against a suite that is already failing.
- All five failing tests reached the shared branch: every commit named in the
  timeline up to `2b12746` is an ancestor of `origin/main` (`571e2c7`).

## Timeline (UTC)

- 2026-09-09 12:37 `8216af8`: the last change whose close records a run of the whole suite.
- 2026-09-09 13:27 `cb25f5b`: adds an LLM judge to the eval report. From here each case
  in the report carries a `judge` key that the exact key-set assertion in
  `tests/evals/test_report.py::test_the_json_document_carries_every_identity_field_by_its_published_name`
  does not list, and that test fails.
- 2026-09-11 09:27 `c3bf64d`: a run wider than one change's own tests (716 passed, 1
  failed) finds that failure and records it. Detection took about 44 hours.
- 2026-09-12 22:45 `5604279`: the nightly eval lands. The record made next says three
  failing tests predate it.
- 2026-09-13 02:36 `ee01d0f`: those three failures are recorded: a cache header in
  `test_cost_wiring.py`, a 429 in `test_rate_limit_enforcement.py`, and
  `KeyError: 'choices'` in `test_rehydration.py`. The record also states that one
  change so far had its tests run whole. The commit that broke each of the three is
  not established.
- 2026-09-16 01:37 `69c7f2c`: adds the tradeoff measurement harness with a test that
  bans `..` in a report path whose only legal form contains `..`. It is red from
  that commit.
- 2026-09-16 01:38 `2b12746`: that failure is recorded one minute later.
- 2026-09-17 `c14ece0`: at this commit the record lists sixty changes that were never
  checked against the whole suite, and the five failures are all still open. This
  postmortem is written.

## Root cause

The check that runs before each commit was set to run only the tests the change
itself wrote. That proves the change does what it claims. It cannot see a test in
another file that the change breaks, and it cannot see a test written wrongly in a
way only other code would reveal. The whole suite, which is the only check that
sees both, did not run once in eight days.

## Contributing factors

- The fast setting was meant for short stretches over well-understood code. It
  stayed on for every change from 2026-09-09 onward, including new subsystems.
- CI runs the suite on every push to `main`, but its results were not part of the
  check made before each change. Nothing in the repository records whether those
  runs were read.
- Failures were recorded as known issues rather than blocking the next change, so
  the list grew instead of shrinking.
- One test (`test_tradeoff_run.py`) added a constraint its own requirements did not
  state, which made it impossible to pass from the day it was written.

## What went well

- Every failure was written down on the day it was found, with the test name and
  its error, which is why this timeline can be built from git history alone.
- The list of changes never checked against the whole suite was kept current, so
  the size of the gap was always visible.
- None of the failures reached a running system.

## Action items (owner, due date)

- Run the whole suite on `main`, fix or deliberately record each of the five
  failures, and clear the list of unchecked changes. Owner: Masudur Rahman. Due: 2026-09-20.
- Cap the number of changes allowed between whole-suite runs, and make the
  pre-commit check refuse a new change once the cap is reached. Owner: Masudur Rahman. Due: 2026-09-24.
- Read the CI result of every push to `main` before the next change starts. Owner: Masudur Rahman. Due: 2026-09-20.
- Record this incident on the console timeline. Dropped: the console no longer
  has an incident timeline, and this postmortem is the record.

Blameless. Written within 48h of the latest failure being recorded. Linked from the README.
