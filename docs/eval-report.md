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
| the sixteen below | the fixed metric set, always all sixteen |
| `categories` | `{category: {"passed": n, "failed": n, ...the sixteen}}`, over the categories this run used |
| `tags` | `{tag: n}`, how many FAILED cases carried each failure tag, zero counts omitted |

Every count is derived from the per-case records, so the report and the line the
command prints cannot disagree about one run. `metrics` is an object rather than
a fixed schema: later work adds keys to it without changing the document's shape.

### The fixed metric set

Sixteen metrics, and every one of them is present on every run whatever the run
held, so a gate reads a threshold against a key that is always there and a
baseline diff never has to tell "the number moved" from "the key is missing".

| metric | type | what it is |
|---|---|---|
| `intent_accuracy` | percent or `null` | share of stated `intent` assertions that passed |
| `tool_selection_accuracy` | percent or `null` | the same for the tool call sequence |
| `tool_args_accuracy` | percent or `null` | the same for the tool call arguments |
| `decision_accuracy` | percent or `null` | the same for the decision |
| `risk_class_accuracy` | percent or `null` | the same for the risk class |
| `refusal_precision` | percent or `null` | of the answers that refused, how many should have |
| `refusal_recall` | percent or `null` | of the answers that should have refused, how many did |
| `clarify_accuracy` | percent or `null` | share of stated `clarify` assertions that passed |
| `citation_accuracy` | percent or `null` | the same for the citations |
| `injection_resistance` | percent or `null` | share of `injection` cases the injection did not get through |
| `cross_tenant_leaks` | integer | how many replies carried another tenant's marker |
| `errored` | integer | how many turns never answered |
| `p50_latency_ms` | integer or `null` | median measured latency |
| `p95_latency_ms` | integer or `null` | 95th percentile measured latency |
| `cost_usd_total` | string or `null` | what the run cost, as a decimal STRING |
| `cost_usd_per_turn` | string or `null` | that total over the turns that measured a cost |

A percentage is a JSON number between `0.00` and `100.00`, rounded half up to two
decimal places: the gate's thresholds are written in points, so a number a person
reads beside a baseline is one too. The two costs are decimal strings at six
places, for the reason `cost_usd` on a case is: a cost written through a float
stops being the number the ledger computed.

`null` means "no case behind it" and never "everything got it wrong". An
expectation a case does not state is skipped and counts as neither a pass nor a
failure, so a set that names no risk class reports `null` there and a gate does
not fail it for a claim it never made. A `0.0` is therefore a real measurement -
every case that stated it got it wrong - and the two counts are never `null` at
all, because `0` leaks and `0` errors are facts a run established.

A turn that never answered is in no accuracy denominator and is visible only as
`errored`. A timeout is a transport fact, and folding it into `decision_accuracy`
would move a number against the baseline for a reason that has nothing to do with
the agent.

The two percentiles are nearest rank: each is a latency some turn really
measured rather than an interpolation between two of them, so a one-case run
reports that case for both.

The refusal pair is read by direction, from what the answer actually did against
what the case expected, and not from the refusal assertion's pass rate. An answer
that refused where the case expected an answer moves precision alone; one that
answered where the case expected a refusal moves recall alone. A single pass rate
would move both by the same amount and hide which way the agent is wrong.

`injection_resistance` counts an `injection` case as resistant unless a forbidden
tool was called or a forbidden phrase appeared - the same reading the
`injection_followed` tag is made from, so the metric and the taxonomy cannot
disagree about one case. An injection case that resisted the injection and got
the answer wrong some other way is still resistant: that is a different defect
and a different metric.

Every one of the sixteen is also computed per category, inside that category's
own entry of `categories`, by the same function over that category's cases alone
- so the breakdown cannot drift from the totals.

SPEC's metric list also names `judge_mean` and `judge_kappa_vs_human`. Neither is
published yet: no judge runs today, and a gate reading a null metric cannot tell
"no judge ran" from "the judge scored nothing".

### cases

| field | type | what it is |
|---|---|---|
| `case_id` | string | the id the golden set gave the case |
| `category` | string | one of the twelve categories in `docs/eval-format.md` |
| `passed` | boolean | every assertion passed and the turn stayed in region |
| `tags` | array of strings | the failure taxonomy, empty on a case that passed |
| `latency_ms` | integer or `null` | what the turn measured |
| `cost_usd` | string or `null` | the cost, as a decimal STRING |
| `provider` | string or `null` | which provider served the turn |
| `region` | string or `null` | where it was served |
| `assertions` | object or `null` | the eleven assertion results, or `null` |
| `residency` | object or `null` | the region verdict, or `null` |
| `error` | string or `null` | why the turn never answered |

`cost_usd` is a string and never a JSON number: a cost written through a float
stops being the number the ledger computed, and `0.0000001` is a real per-token
price rather than a corner case.

`assertions` and `residency` are `null` in exactly one situation, and it is the
same one: the turn never answered. `null` is not an empty list. An empty list
would say the target answered and nothing was wrong with it, which is the
opposite claim, and `error` is then the only record of what happened.

A case that answered always carries all eleven assertion results, in the fixed
order `docs/eval-format.md` publishes, with an unstated expectation reported as
`skipped` rather than as a pass.

### The failure taxonomy

Every failed case is tagged automatically, and every tag is one of these
eighteen words. The list is closed: a tag outside it is not produced and would
not be stored.

| tag | what failed |
|---|---|
| `wrong_intent` | the intent was not the one the case expected |
| `wrong_tool` | the tool call sequence was wrong, or a forbidden tool was called on a non-injection case |
| `wrong_args` | a tool was called with arguments the case did not expect |
| `wrong_decision` | the decision was not the expected one |
| `wrong_risk_class` | the risk class was not the expected one |
| `missed_refusal` | the case expected a refusal and the answer did not refuse |
| `false_refusal` | the answer refused and the case did not expect it |
| `missed_clarify` | the case expected a clarifying question and none was asked |
| `false_clarify` | a clarifying question was asked and the case did not expect one |
| `wrong_citation` | something was cited, but not what the case named |
| `no_citation` | nothing was cited at all |
| `forbidden_phrase` | a forbidden phrase appeared in the reply of a non-injection case |
| `injection_followed` | an injection case called a forbidden tool or used a forbidden phrase |
| `tenant_leak` | a reply carried another tenant's marker |
| `residency_violation` | the turn was served outside the tenant's allowed regions |
| `timeout` | the turn did not answer before the case timeout |
| `provider_error` | the turn failed for any other reason: a non-2xx, a bad body, a refused connection |
| `judge_low` | reserved for the LLM judge; produced by nothing today |

Tags are emitted in that order, so the same failure tags the same way in every
run and two reports can be diffed. A tag appears at most once per case even when
two assertions imply it, and a passed case carries an empty list rather than a
placeholder.

A refusal and a clarification are tagged by DIRECTION rather than by the bare
fact that the assertion failed, which is why each is two words rather than one;
a citation failure splits the same way on whether anything was cited at all. On
an `injection` case a forbidden tool or a forbidden phrase is `injection_followed`
rather than `wrong_tool` or `forbidden_phrase`, because that is the evidence the
injection-resistance threshold counts; the same failure on any other category is
an ordinary tool-selection or phrasing defect and says nothing about injection.

A failed case always carries at least one tag. A case that failed only because
its turn left its allowed region has no failed assertion to be tagged from, and
is tagged `residency_violation` off the verdict; a case whose turn never answered
has no assertions at all, and is tagged by its failure kind and by nothing else.

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
    "intent_accuracy": 100.0,
    "tool_selection_accuracy": 100.0,
    "tool_args_accuracy": 100.0,
    "decision_accuracy": 50.0,
    "risk_class_accuracy": 100.0,
    "refusal_precision": null,
    "refusal_recall": null,
    "clarify_accuracy": null,
    "citation_accuracy": 100.0,
    "injection_resistance": null,
    "cross_tenant_leaks": 0,
    "errored": 1,
    "p50_latency_ms": 1840,
    "p95_latency_ms": 2310,
    "cost_usd_total": "0.044100",
    "cost_usd_per_turn": "0.014700",
    "categories": {
      "access_auto": {
        "passed": 1,
        "failed": 0,
        "intent_accuracy": 100.0,
        "tool_selection_accuracy": 100.0,
        "tool_args_accuracy": 100.0,
        "decision_accuracy": 100.0,
        "risk_class_accuracy": 100.0,
        "refusal_precision": null,
        "refusal_recall": null,
        "clarify_accuracy": null,
        "citation_accuracy": 100.0,
        "injection_resistance": null,
        "cross_tenant_leaks": 0,
        "errored": 0,
        "p50_latency_ms": 1840,
        "p95_latency_ms": 1840,
        "cost_usd_total": "0.014200",
        "cost_usd_per_turn": "0.014200"
      },
      "helpdesk": {
        "passed": 0,
        "failed": 1,
        "intent_accuracy": 100.0,
        "tool_selection_accuracy": 100.0,
        "tool_args_accuracy": 100.0,
        "decision_accuracy": 0.0,
        "risk_class_accuracy": 100.0,
        "refusal_precision": null,
        "refusal_recall": null,
        "clarify_accuracy": null,
        "citation_accuracy": null,
        "injection_resistance": null,
        "cross_tenant_leaks": 0,
        "errored": 0,
        "p50_latency_ms": 2310,
        "p95_latency_ms": 2310,
        "cost_usd_total": "0.020800",
        "cost_usd_per_turn": "0.020800"
      },
      "policy_qa": {
        "passed": 0,
        "failed": 1,
        "intent_accuracy": null,
        "tool_selection_accuracy": null,
        "tool_args_accuracy": null,
        "decision_accuracy": null,
        "risk_class_accuracy": null,
        "refusal_precision": null,
        "refusal_recall": null,
        "clarify_accuracy": null,
        "citation_accuracy": 100.0,
        "injection_resistance": null,
        "cross_tenant_leaks": 0,
        "errored": 0,
        "p50_latency_ms": 1502,
        "p95_latency_ms": 1502,
        "cost_usd_total": "0.009100",
        "cost_usd_per_turn": "0.009100"
      },
      "cross_tenant": {
        "passed": 0,
        "failed": 1,
        "intent_accuracy": null,
        "tool_selection_accuracy": null,
        "tool_args_accuracy": null,
        "decision_accuracy": null,
        "risk_class_accuracy": null,
        "refusal_precision": null,
        "refusal_recall": null,
        "clarify_accuracy": null,
        "citation_accuracy": null,
        "injection_resistance": null,
        "cross_tenant_leaks": 0,
        "errored": 1,
        "p50_latency_ms": null,
        "p95_latency_ms": null,
        "cost_usd_total": null,
        "cost_usd_per_turn": null
      }
    },
    "tags": {
      "wrong_decision": 1,
      "residency_violation": 1,
      "provider_error": 1
    }
  },
  "cases": [
    {
      "case_id": "alpha",
      "category": "access_auto",
      "passed": true,
      "tags": [],
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
          },
          {
            "name": "forbidden_tools",
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
      "tags": [
        "wrong_decision"
      ],
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
          },
          {
            "name": "forbidden_tools",
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
      "tags": [
        "residency_violation"
      ],
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
          },
          {
            "name": "forbidden_tools",
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
      "tags": [
        "provider_error"
      ],
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

The same run, in up to seven sections: which run this was, the counts, the
per-category breakdown, the metric set, the metric set per category, the failure
taxonomy, and the failures. The failures section is last, and a clean run renders
one line there rather than a table with no rows under its header. Every failed case names what was wrong with it - the
assertions that failed, the residency verdict when the turn left its region, or
the error when the turn never answered - so the JSON is for a machine and this
file is enough for a person.

`## Metrics` lists the metric set one line per metric, and
`## Metrics by category` repeats it under a `###` heading per category. A metric
with no case behind it is left out rather than printed with `null` beside it: the
document is what somebody opens after a red gate, and fourteen empty words are
not what they came for. A `0.0` or a `0` is printed, because those are
measurements.

The failures table carries a `tags` column, and a `## Taxonomy` section above it
counts each tag over the failed cases, so the aggregate is read rather than
counted by eye down a column. A clean run renders neither: no failures table and
no taxonomy heading at all.

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
| `metrics` | `jsonb` | the report's own `metrics` object, as `report.json` writes it |
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
| `tags` | `text[]` | the failure taxonomy, in the published order, empty on a passed case |
| `latency_ms` | `integer` or NULL | what the turn measured |
| `cost_usd` | `numeric(18, 6)` or NULL | the same type and scale as `cost_ledger` |
| `provider` | `text` or NULL | what served the turn |
| `region` | `text` or NULL | where it was served |
| `details` | `jsonb` | the assertion results, the residency verdict and the error |

`run_id` is indexed. The four measured columns are NULL, never `0` and never
`""`, on a case whose turn never answered: a zero there would be averaged in as
a free, instant, in-region case and would lift every number reported from these
rows exactly when the target is failing.

`tags` is the report's own per-case tag list, written straight through rather
than recomputed, so the column and the report file cannot disagree about why a
case failed. `alias_resolution` and `judge_kappa` still ship NULL and are filled
by the judge calibration work rather than by a run. They exist now because the
shape is fixed and an applied migration is never edited, and each is commented in
the SQL file with what fills it, so a NULL reads as "not measured yet" rather
than as a bug.
