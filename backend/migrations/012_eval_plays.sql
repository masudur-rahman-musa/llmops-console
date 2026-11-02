-- One row per play of a run: its result, the end state it reached, the rules it broke,
-- the never texts the judge flagged, the reply check answers and its effort.
--
-- A run's scores stay in `eval_runs.metrics` and a case's conversations in
-- `eval_cases.details`; this table holds what a query counts or lists play by play, such
-- as the plays that need a look in the newest run.
--
-- `result` takes the four words and nothing else, so a fifth result no query counts can
-- never be stored. `end_state` is the index of the right end state matched, NULL when the
-- play matched none. `cost_usd` is NULL when the help desk reported no cost, never 0.
-- Deleting a run deletes its plays.
--
-- `{schema}` is substituted by `apply_migrations`, which refuses any schema name that is
-- not a lowercase slug before it reads this file.

CREATE TABLE IF NOT EXISTS {schema}.eval_plays (
    run_id      text             NOT NULL
                REFERENCES {schema}.eval_runs (id) ON DELETE CASCADE,
    case_id     text             NOT NULL,
    play        integer          NOT NULL,
    result      text             NOT NULL,
    end_state   integer,
    broken      text[]           NOT NULL DEFAULT '{}',
    flagged     text[]           NOT NULL DEFAULT '{}',
    checks      jsonb            NOT NULL,
    turns       integer          NOT NULL,
    tool_calls  integer          NOT NULL,
    seconds     double precision NOT NULL,
    cost_usd    numeric(18, 6),
    PRIMARY KEY (run_id, case_id, play),
    CONSTRAINT eval_plays_result_check
        CHECK (result IN ('setup_wrong', 'fail', 'pass', 'needs_a_look'))
);
