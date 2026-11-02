# Judge labels and calibration

The judge answers each reply check with `yes`, `no` or `does_not_apply`. Before its
answers count toward reply quality, it is checked against answers a person gave to the
same questions. This page says what the labels file holds and how the check works.

## The labels file

`backend/evals/labels/labels.jsonl` holds one JSON line per reply check of one play:

| field | what it is |
|---|---|
| `case_id` | the case the play came from |
| `play` | which play, 1 to 10 |
| `check_id` | the reply check type: lowercase letters, digits and `_`, 1 to 64 characters |
| `question` | the question the judge is asked, as a run asks it |
| `conversation` | the turns that were judged: each with `person`, `reply` and `tool_calls` (`name`, `args`, `ok`, `error_code`, `result`) |
| `answer` | `yes`, `no` or `does_not_apply` |
| `reason` | one sentence on why |
| `labelled_by` | `model` (a draft) or `musa` (reviewed by the owner) |
| `labelled_on` | the date, as `YYYY-MM-DD` |

There is one row per case, play and check. Any other field, a fourth answer or a second
row for the same three stops the load, naming the line.

The project only reads this file. Rows are prepared outside it and reviewed by the owner;
the file ships empty.

## Calibrating

    ops-eval calibrate --labels backend/evals/labels/labels.jsonl \
        --gateway-url URL --out backend/evals/labels/calibration.json

The judge is asked about every row labelled by `musa`, one request per row, with the
same prompt and tags a run uses. Rows labelled by `model` are not asked and not counted.
With no `musa` row the command stops and writes nothing. It needs
`OPSCONSOLE_EVAL_JUDGE_KEY`, as a run does.

For each check type it counts:

| field | what it is |
|---|---|
| `labels` | the owner's rows of this type |
| `answered` | how many the judge answered; one it could not answer is left out |
| `agreed` | how many answers matched the owner's |
| `kappa` | Cohen's kappa over the answered rows, four decimals |
| `trusted` | `true` when kappa is 0.7 or better |

Kappa here is unweighted: the three answers have no order, so `yes` against
`does_not_apply` is as wrong as `yes` against `no`. When every label and every answer is
the same word, there is no way to tell agreement from luck: kappa is `null` and the type
is not trusted.

`calibration.json` also names the judge model (the first model the `judge` alias routes
to), the sha256 of the judge prompt file and the sha256 of the labels file.

## How a run uses it

`ops-eval conversations --calibration PATH` reads the file. Reply quality then counts the
check types marked trusted, but only when the file's judge prompt hash is the prompt in
use now. After the prompt changes, nothing is trusted until the judge is calibrated
again. `make eval` passes the file when it exists.
