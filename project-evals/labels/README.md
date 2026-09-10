# Judge calibration labels

Human-graded replies used to check the LLM judge before any of its numbers are
trusted. One JSON object per line, `*.jsonl`, at least 40 records:

```json
{"id": "helpdesk-refund-014", "output": "Your refund was issued on 3 March and should land within five working days.", "human_score": 4, "judge_rubric": "Score 5 only if the reply names both the date and the expected arrival window.", "note": "names both, but hedges the window"}
```

| field | type | what it is |
|---|---|---|
| `id` | lowercase slug, up to 128 characters, unique in the file | the record a disagreement is looked up by |
| `output` | text, up to 20000 characters | the reply that was graded |
| `human_score` | integer 1 to 5 | what a person scored it against the rubric |
| `judge_rubric` | text, up to 20000 characters | the standard it was graded against |
| `note` | text, optional | why the human scored what they did; never sent to the judge |

The rubric is carried on the label rather than looked up in a golden set, so a
labels file stands on its own and outlives the set the replies came from.

Run it:

    ops-eval calibrate --labels project-evals/labels/helpdesk.jsonl \
                       --judge-url https://gateway.example \
                       --out project-evals/labels/out

`OPSCONSOLE_EVAL_JUDGE_KEY` must be set; without it the command refuses rather
than writing a calibration nothing scored. The output is one
`calibration.json` carrying the agreement, the judge prompt's version hash and
one row per scored label.

Agreement is Cohen's kappa with quadratic weights, so a judge that answers 4
where a person said 5 is penalised a sixteenth of what one that answers 1 is.
It is published at three decimal places, and a judge below **0.7** is not
trusted: its scores are still recorded, and nothing reads them as evidence.

Two things the command refuses rather than working around:

- **fewer than 40 usable labels**, both before it starts and after any the judge
  could not score are dropped. A kappa over a dozen pairs is worse than no
  calibration, because a file that exists gets trusted.
- **a table with no expected disagreement** - every human score identical, or
  every judge score identical. The agreement is real and tells you nothing, so
  the kappa is reported as `null` with the reason beside it, never as `1.0`.

A label the judge could not score after its one retry is dropped and counted in
the artifact. It is never scored as a zero, and never as the human's own number.
