# How this was built

This project was built with a coding agent (Claude Code), working in a fixed
loop. The loop is what kept it reliable: nothing was written before it was
planned, and nothing was called done before the tests and the full check
proved it. If you want to build something the same way, you can follow it.

## The flow

1. **Plan it.** Write down what changes and why, which files it touches, and
   the acceptance criteria: the checks that prove it works, including what must
   be refused or must fail safely.
2. **Write the tests first.** One test per acceptance criterion. They fail at
   this point, because the code does not exist yet. A coding agent can write
   them straight from the criteria.
3. **Make the smallest change that turns them green.** Touch nothing the plan
   does not name.
4. **Run `make verify`.** It runs the same checks CI runs, so a green run here
   means a green run there.
5. **Commit** with a plain message that says what the change does.

Two things make the loop work. The acceptance criteria are written before any
code, so "done" means the same thing to the person and to the agent. And the
tests come from those criteria, not from the code, so they check what was asked
for rather than what happened to be built.

## A plan, in practice

Each plan was one short file, small enough to build and check in one sitting:

```
# <short name of the change>

## What and why
<one paragraph: what changes, and the reason>

## Files
<the files this change touches>

## Acceptance criteria
- <a check that proves it works>
- <a check that proves it refuses or fails safely>

## Tests
<the test files that hold one test per criterion>
```

## What `make verify` runs

- `format`, `lint` and `typecheck`: code style and static types (ruff, mypy).
- `test`: the backend tests, with a coverage floor.
- `security`: a secret scan over the whole history and dependency audits.

CI calls the same targets, so the laptop and CI cannot disagree.

## What is not in this repository

The plan files, the decision notes and the working notes were kept out of this
public copy. What is here is the result of that process: the code, the tests,
and a commit for each change.
