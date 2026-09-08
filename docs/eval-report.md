# The eval report

Every `ops-eval run` writes two files into the directory `--out` names, whatever
the run answered:

    report.json   the whole record, for a gate and for storage
    report.md     the same run for a person to read

Both names are fixed. A rerun into the same directory overwrites them, so a
caller who wants history keeps directories rather than file names, and a gate
reads a known path instead of globbing for the newest file. Both are written
before the exit code is chosen: a failed run is exactly the one somebody opens
the report for.

## report.json, field by field

### The run

| field | type | what it is |
|---|---|---|
| `id` | string | a `uuid4` hex, 32 characters, fresh per run |
| `started_at` | timestamp | UTC, stamped just before the first case |
| `finished_at` | timestamp | UTC, stamped after the last one |
| `set_name` | string | the golden set's file name |
| `set_sha` | string | which version of the set this was (see below) |
| `set_sha_source` | `commit` or `content` | how that answer was arrived at |
| `run_kind` | `ci`, `nightly` or `manual` | why the run happened |
| `git_sha` | string or `null` | this repo's `HEAD`, or `null` when git cannot answer |
| `tenant` | string | the tenant the run was made as |
| `target` | string | the base URL evaluated |
| `passed` | boolean | true when no case failed |
| `metrics` | object | the counts, below |
| `cases` | array | one entry per case, in the order the set listed them |

`set_sha` answers "which version of the golden set produced this run?". The set
is an adopter's file and usually lives in another repository, whose commit this
process cannot see, so `--set-sha` lets a caller pass the real commit and the
report records `set_sha_source: "commit"`. With no `--set-sha` the field is the
`sha256` of the set file's bytes and the source reads `content`. The source
travels with the value so no reader is ever told a content hash is a commit.

`git_sha` is this repository's `HEAD` and is `null` - never `unknown`, never an
empty string - when git is absent, the directory is not a repository, or the
call fails.

### metrics

| key | what it is |
|---|---|
| `cases` | how many cases ran, a broken turn included |
| `passed` | how many passed every assertion and stayed in region |
| `failed` | the rest, whatever the reason |
| `residency_violations` | how many turns were served outside the tenant's allowed regions |
| `leaked` | true when a reply carried another tenant's marker |
| `categories` | `{category: {"passed": n, "failed": n}}`, over the categories this run used |

Every count is derived from the per-case records, so the report and the line the
command prints cannot disagree about one run. `metrics` is an object rather than
a fixed schema: later work adds keys to it without changing the document's shape.

### cases

| field | type | what it is |
|---|---|---|
| `case_id` | string | the id the golden set gave the case |
| `category` | string | one of the twelve categories in `docs/eval-format.md` |
| `passed` | boolean | every assertion passed and the turn stayed in region |
| `latency_ms` | integer or `null` | what the turn measured |
| `cost_usd` | string or `null` | the cost, as a decimal STRING |
| `provider` | string or `null` | which provider served the turn |
| `region` | string or `null` | where it was served |
| `assertions` | object or `null` | the ten assertion results, or `null` |
| `residency` | object or `null` | the region verdict, or `null` |
| `error` | string or `null` | why the turn never answered |

`cost_usd` is a string and never a JSON number: a cost written through a float
stops being the number the ledger computed, and `0.0000001` is a real per-token
price rather than a corner case.

`assertions` and `residency` are `null` in exactly one situation, and it is the
same one: the turn never answered. `null` is not an empty list. An empty list
would say the target answered and nothing was wrong with it, which is the
opposite claim, and `error` is then the only record of what happened.

A case that answered always carries all ten assertion results, in the fixed
order `docs/eval-format.md` publishes, with an unstated expectation reported as
`skipped` rather than as a pass.

## A worked report

Four cases: one passed, one failed an assertion, one was served outside its
allowed regions with every assertion passing, and one never answered.

```json
{
  "id": "9f2c1d8b4a6e47c3a1b05e7d2f83c4a6",
  "started_at": "2026-03-04T09:12:31.204512Z",
  "finished_at": "2026-03-04T09:14:02.881044Z",
  "set_name": "helpdesk-golden-v1.jsonl",
  "set_sha": "e4b2c0d9a71f3856cbb4d20e9f61a7c845d3e0b2",
  "set_sha_source": "commit",
  "run_kind": "ci",
  "git_sha": "b71f0c3a9d5e42188a6c0f37d21be9054c8a7f6d",
  "tenant": "acme",
  "target": "https://helpdesk.example",
  "passed": false,
  "metrics": {
    "cases": 4,
    "passed": 1,
    "failed": 3,
    "residency_violations": 1,
    "leaked": false,
    "categories": {
      "access_auto": {
        "passed": 1,
        "failed": 0
      },
      "helpdesk": {
        "passed": 0,
        "failed": 1
      },
      "policy_qa": {
        "passed": 0,
        "failed": 1
      },
      "cross_tenant": {
        "passed": 0,
        "failed": 1
      }
    }
  },
  "cases": [
    {
      "case_id": "alpha",
      "category": "access_auto",
      "passed": true,
      "latency_ms": 1840,
      "cost_usd": "0.0142",
      "provider": "anthropic_eu",
      "region": "eu",
      "assertions": {
        "case_id": "alpha",
        "results": [
          {
            "name": "intent",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "tool_call_sequence",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "tool_call_args",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "decision",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "risk_class",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "refusal",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "clarify",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "citations",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "forbidden_phrases",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "cross_tenant_leak",
            "status": "skipped",
            "expected": null,
            "actual": null
          }
        ]
      },
      "residency": {
        "case_id": "alpha",
        "tenant": "acme",
        "data_class": "internal",
        "observed": "eu",
        "allowed": [
          "eu",
          "self"
        ],
        "violation": false
      },
      "error": null
    },
    {
      "case_id": "beta",
      "category": "helpdesk",
      "passed": false,
      "latency_ms": 2310,
      "cost_usd": "0.0208",
      "provider": "anthropic_eu",
      "region": "eu",
      "assertions": {
        "case_id": "beta",
        "results": [
          {
            "name": "intent",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "tool_call_sequence",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "tool_call_args",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "decision",
            "status": "failed",
            "expected": "auto_approved",
            "actual": "escalated"
          },
          {
            "name": "risk_class",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "refusal",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "clarify",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "citations",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "forbidden_phrases",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "cross_tenant_leak",
            "status": "skipped",
            "expected": null,
            "actual": null
          }
        ]
      },
      "residency": {
        "case_id": "beta",
        "tenant": "acme",
        "data_class": "internal",
        "observed": "eu",
        "allowed": [
          "eu",
          "self"
        ],
        "violation": false
      },
      "error": null
    },
    {
      "case_id": "gamma",
      "category": "policy_qa",
      "passed": false,
      "latency_ms": 1502,
      "cost_usd": "0.0091",
      "provider": "openai_us",
      "region": "us",
      "assertions": {
        "case_id": "gamma",
        "results": [
          {
            "name": "intent",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "tool_call_sequence",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "tool_call_args",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "decision",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "risk_class",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "refusal",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "clarify",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "citations",
            "status": "passed",
            "expected": null,
            "actual": null
          },
          {
            "name": "forbidden_phrases",
            "status": "skipped",
            "expected": null,
            "actual": null
          },
          {
            "name": "cross_tenant_leak",
            "status": "skipped",
            "expected": null,
            "actual": null
          }
        ]
      },
      "residency": {
        "case_id": "gamma",
        "tenant": "acme",
        "data_class": "internal",
        "observed": "us",
        "allowed": [
          "eu",
          "self"
        ],
        "violation": true
      },
      "error": null
    },
    {
      "case_id": "delta",
      "category": "cross_tenant",
      "passed": false,
      "latency_ms": null,
      "cost_usd": null,
      "provider": null,
      "region": null,
      "assertions": null,
      "residency": null,
      "error": "turn failed: HTTP 502 from https://helpdesk.example/v1/eval/turn"
    }
  ]
}
```

## report.md

The same run, in four sections: which run this was, the counts, the per-category
breakdown, and the failures. The failures section is last, and a clean run
renders one line there rather than a table with no rows under its header. Every
failed case names what was wrong with it - the assertions that failed, the
residency verdict when the turn left its region, or the error when the turn
never answered - so the JSON is for a machine and this file is enough for a
person.

## Storage

The same run is also stored in PostgreSQL, in two tables `003_eval_runs.sql`
creates in the `opsconsole` schema: one `eval_runs` row and one `eval_cases` row
per case. The rows are written AFTER both files, in one transaction, so a
storage outage costs a run its rows and never its report; a run whose cases only
half-landed would be counted by every later query as a run with a shorter golden
set, which is why it is one transaction rather than a statement per case.

With no database URL configured a run stores nothing, says so in one line on
stderr and keeps its own exit code: the golden set is usually run on a machine
with no PostgreSQL up, and refusing there would turn the whole harness off
rather than turn storage off. When a configured store does fail, the run exits
`2` - unless it had already answered `3` for a residency violation, which wins.

### eval_runs

| column | type | what it is |
|---|---|---|
| `id` | `text` primary key | the report's own run id, a `uuid4` hex |
| `started_at` | `timestamptz` | UTC, just before the first case |
| `finished_at` | `timestamptz` | UTC, after the last one |
| `set_name` | `text` | the golden set's file name |
| `set_sha` | `text` | which version of the set this was |
| `set_sha_source` | `text` | `commit` or `content` |
| `run_kind` | `text` | `ci`, `nightly` or `manual`, with a CHECK |
| `target` | `text` | the base URL evaluated |
| `tenant` | `text` | the tenant the run was made as |
| `git_sha` | `text` or NULL | this repo's `HEAD`, NULL when git cannot answer |
| `metrics` | `jsonb` | the report's own `metrics` object |
| `passed` | `boolean` | true when no case failed |
| `report_path` | `text` | the `report.json` this run wrote |
| `alias_resolution` | `jsonb` or NULL | which alias resolved where, filled later |
| `judge_kappa` | `double precision` or NULL | judge calibration, filled later |

The report's `run_kind` and the URL evaluated are two columns rather than one:
the kind of run and the thing evaluated are different questions, and a stored
number whose endpoint is unknown answers neither. `started_at DESC` is indexed,
which is the order the console's eval page reads.

### eval_cases

| column | type | what it is |
|---|---|---|
| `run_id` | `text` | the `eval_runs.id` this case belongs to, ON DELETE CASCADE |
| `case_id` | `text` | the case's id in the golden set |
| `category` | `text` | one of the twelve categories |
| `passed` | `boolean` | whether this case passed |
| `tags` | `text[]` | the failure taxonomy, empty until it is filled |
| `latency_ms` | `integer` or NULL | what the turn measured |
| `cost_usd` | `numeric(18, 6)` or NULL | the same type and scale as `cost_ledger` |
| `provider` | `text` or NULL | what served the turn |
| `region` | `text` or NULL | where it was served |
| `details` | `jsonb` | the assertion results, the residency verdict and the error |

`run_id` is indexed. The four measured columns are NULL, never `0` and never
`""`, on a case whose turn never answered: a zero there would be averaged in as
a free, instant, in-region case and would lift every number reported from these
rows exactly when the target is failing.

Three columns ship empty and are filled by later work, not by a run: `tags`,
`alias_resolution` and `judge_kappa`. They exist now because the shape is fixed
and an applied migration is never edited, and each is commented in the SQL file
with what fills it, so a NULL reads as "not measured yet" rather than as a bug.
