-- The two console tables nothing else in the system produces (SPEC §18, §16.4).
--
-- Every other table in this directory is filled by a code path: `cost_ledger` and
-- `residency_log` by a served request, `eval_runs` and `eval_cases` by an eval run.
-- These two are filled by a human and by CI through `/admin/incidents` and
-- `/admin/releases`, which is why plan 28a arrives as an endpoint rather than as a
-- writer behind one.
--
-- `incidents.resolved_at` is NULLABLE and that is the whole definition of "open":
-- the timeline selects `resolved_at IS NULL` rather than carrying a status column
-- that can disagree with the timestamps beside it. `POST /admin/incidents` has no
-- `resolved_at` field at all, so an incident is always born open and only
-- `PATCH /admin/incidents/{incident_id}` closes it.
--
-- `releases.eval_run_id` is NULLABLE for a reason the column type cannot express: a
-- sha deployed before any eval ran has no run to name, and `''` there would read as
-- a run id that does not exist rather than as "not measured". It is a FOREIGN KEY as
-- well as a pre-check, because the pre-check only guards the route - a psql session,
-- a later backfill or CI posting a run whose rows never stored all reach the table
-- another way, and a release joined to a missing run shows a deploy with no quality
-- number and no error.
--
-- The primary key is `(git_sha, deployed_at)` rather than `git_sha` (plan 28a
-- decision B). SPEC §17 calls the page "deploys", plural, and a rollback re-deploys
-- an earlier sha; a `git_sha` key with an upsert would overwrite the first deploy's
-- timestamp and lose the rollback entirely.
--
-- An applied migration is never edited - a later plan needing another column adds
-- 005, exactly as 002 and 003 say of themselves.
--
-- `{schema}` is substituted by `opsconsole.ledger.db.apply_migrations`, which
-- refuses any schema name that is not a lowercase slug before it reads this file. A
-- schema name cannot be a bound parameter.

CREATE TABLE IF NOT EXISTS {schema}.incidents (
    id             text        PRIMARY KEY,
    started_at     timestamptz NOT NULL,
    resolved_at    timestamptz,
    severity       text        NOT NULL,
    title          text        NOT NULL,
    summary        text        NOT NULL,
    postmortem_url text,
    CONSTRAINT incidents_severity_check CHECK (severity IN ('sev1', 'sev2', 'sev3'))
);

-- `id` is a `uuid4` hex the route generates per incident (decision A), so no client
-- string shapes a primary key and two incidents filed in the same second cannot
-- collide on a slug someone typed twice.

-- The question the timeline asks: "the most recent incidents". Without it that page
-- sequentially scans every incident ever recorded, which gets slower every quarter
-- and never fails.
CREATE INDEX IF NOT EXISTS incidents_started_at_idx
    ON {schema}.incidents (started_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.releases (
    git_sha     text        NOT NULL,
    deployed_at timestamptz NOT NULL,
    eval_run_id text        REFERENCES {schema}.eval_runs (id),
    pr_url      text,
    status      text        NOT NULL,
    PRIMARY KEY (git_sha, deployed_at),
    CONSTRAINT releases_status_check CHECK (status IN ('deployed', 'rolled_back', 'blocked'))
);
