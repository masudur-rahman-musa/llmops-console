-- The residency decision log: one row per request a provider actually served.
--
-- RANGE partitioned by `ts`, one partition per day, because retention is the
-- only thing that ever deletes from this table and dropping a partition is a
-- catalogue update while deleting ninety days of rows is a table rewrite.
--
-- `residency_log_default` is the backstop. The writer creates a day's partition
-- on demand and a refused CREATE is swallowed, so without a DEFAULT partition
-- that refusal would turn into `no partition of relation found` and lose the
-- row rather than misfile it.
--
-- `{schema}` is substituted by `opsconsole.ledger.db.apply_migrations`, which
-- refuses any schema name that is not a lowercase slug before it reads this
-- file. A schema name cannot be a bound parameter.

CREATE TABLE IF NOT EXISTS {schema}.residency_log (
    ts          timestamptz NOT NULL,
    tenant      text        NOT NULL,
    data_class  text        NOT NULL,
    alias       text        NOT NULL,
    provider    text        NOT NULL,
    region      text        NOT NULL,
    allowed     boolean     NOT NULL
) PARTITION BY RANGE (ts);

CREATE TABLE IF NOT EXISTS {schema}.residency_log_default
    PARTITION OF {schema}.residency_log DEFAULT;

-- The two questions this table is asked: "where did this tenant's data go" and
-- "show me every violation". The second is answered by a partial index because
-- `allowed = false` is empty by construction, so a full index on the column
-- would be ninety days of `true` maintained for nothing.
CREATE INDEX IF NOT EXISTS residency_log_tenant_ts_idx
    ON {schema}.residency_log (tenant, ts DESC);

CREATE INDEX IF NOT EXISTS residency_log_violations_idx
    ON {schema}.residency_log (ts DESC) WHERE NOT allowed;
