-- The eval tables: one row per run, one row per case in it (SPEC §18).
--
-- Written by `opsconsole.evals.store`, which lives in `evals/` rather than here
-- (plan 16d decision A): the writer needs `evals.report` for the record it turns
-- into rows, and `ledger/` is the lower layer that `evals/` and `gateway/` both
-- sit on, so the dependency points one way only. The FILE has no such choice -
-- `opsconsole.ledger.db.apply_migrations` globs exactly this directory.
--
-- SPEC §18 gives `eval_runs` one column named `target` carrying `ci`/`nightly`/
-- `manual`. That conflates the KIND of run with the thing evaluated, and the
-- report already carries both apart, so this file splits them (decision C):
-- `run_kind` is the three words, with a CHECK, and `target` is the URL that was
-- measured. Storing only the kind would lose which endpoint produced a number,
-- which is the first question anyone asks of a regression.
--
-- `cost_usd` is `numeric(18, 6)`, the same column type `cost_ledger` uses, so a
-- run's spend and the ledger's own sum are the same number at the same scale. A
-- narrower scale rounds a cost the price table computed exactly; `double
-- precision` would not sum reproducibly at all.
--
-- The four measured columns on `eval_cases` are NULLABLE on purpose. A turn that
-- never answered measured nothing, and `0` there would be averaged in as a free,
-- instant case - it would lift every latency and cost number exactly when the
-- target is failing.
--
-- `{schema}` is substituted by `opsconsole.ledger.db.apply_migrations`, which
-- refuses any schema name that is not a lowercase slug before it reads this
-- file. A schema name cannot be a bound parameter.

CREATE TABLE IF NOT EXISTS {schema}.eval_runs (
    id              text        PRIMARY KEY,
    started_at      timestamptz NOT NULL,
    finished_at     timestamptz NOT NULL,
    set_name        text        NOT NULL,
    set_sha         text        NOT NULL,
    set_sha_source  text        NOT NULL,
    run_kind        text        NOT NULL,
    target          text        NOT NULL,
    tenant          text        NOT NULL,
    git_sha         text,
    metrics         jsonb       NOT NULL,
    passed          boolean     NOT NULL,
    report_path     text        NOT NULL,
    -- Filled by row 17, NULL until then, and commented so a NULL reads as "not
    -- measured yet" rather than as a bug (decision F). Created now rather than by
    -- a later migration because §18 already fixes the shape and an applied
    -- migration is never edited.
    alias_resolution jsonb,
    judge_kappa      double precision,
    CONSTRAINT eval_runs_run_kind_check CHECK (run_kind IN ('ci', 'nightly', 'manual'))
);

-- `id` is a `uuid4` hex the report generates per build, so two nightly runs of
-- the same set on the same commit are two rows rather than a collision - which
-- is the comparison this table exists to support.

-- The question the console's eval page asks: "the most recent runs". Without it
-- that page sequentially scans every run ever stored, which gets slower every
-- night and never fails.
CREATE INDEX IF NOT EXISTS eval_runs_started_at_idx
    ON {schema}.eval_runs (started_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.eval_cases (
    run_id      text           NOT NULL
                REFERENCES {schema}.eval_runs (id) ON DELETE CASCADE,
    case_id     text           NOT NULL,
    category    text           NOT NULL,
    passed      boolean        NOT NULL,
    -- Row 17's failure taxonomy (SPEC §12.4). Empty rather than NULL until then:
    -- a NULL array is a value every later `unnest` has to know about (decision F).
    tags        text[]         NOT NULL DEFAULT '{}',
    latency_ms  integer,
    cost_usd    numeric(18, 6),
    provider    text,
    region      text,
    details     jsonb          NOT NULL
);

-- The other question: "this run's cases". The foreign key does not create an
-- index of its own, so without this every case lookup and every cascade delete
-- scans the whole table.
CREATE INDEX IF NOT EXISTS eval_cases_run_id_idx
    ON {schema}.eval_cases (run_id);
