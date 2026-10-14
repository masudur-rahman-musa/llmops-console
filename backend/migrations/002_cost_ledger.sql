-- The cost ledger: one row per request, and the daily rollup the console reads.
--
-- NOT partitioned, unlike `residency_log`. Partitions there pay for themselves
-- because a 90-day retention drops them, and dropping a partition is a catalogue
-- update where deleting the same rows is a table rewrite. This table has no
-- retention at all - the console reads months of it - so partitions would buy a
-- create-on-demand path on the hot write and a DEFAULT-partition backstop to go
-- with it, in exchange for a purge nobody would ever run.
--
-- `cost_usd` is `numeric(18,6)` because `opsconsole.limits.cost` quantizes to six
-- decimal places and a month of rows has to sum to the same number twice. A
-- narrower scale rounds a number the library computed exactly and shows up as a
-- cent per million requests rather than as a failure; `double precision` would
-- not sum reproducibly at all.
--
-- `session` is the one nullable column: a request without a session header is
-- still a request that cost money, and NOT NULL here would lose exactly those
-- rows. `estimated` carries no default, so a writer that forgot the column fails
-- rather than booking every estimate as measured.
--
-- `{schema}` is substituted by `opsconsole.ledger.db.apply_migrations`, which
-- refuses any schema name that is not a lowercase slug before it reads this
-- file. A schema name cannot be a bound parameter.

CREATE TABLE IF NOT EXISTS {schema}.cost_ledger (
    ts             timestamptz    NOT NULL,
    tenant         text           NOT NULL,
    alias          text           NOT NULL,
    provider       text           NOT NULL,
    model          text           NOT NULL,
    region         text           NOT NULL,
    input_tokens   integer        NOT NULL,
    output_tokens  integer        NOT NULL,
    cached         boolean        NOT NULL,
    cost_usd       numeric(18, 6) NOT NULL,
    price_version  text           NOT NULL,
    estimated      boolean        NOT NULL,
    session        text,
    trace_id       text           NOT NULL
);

-- The question the console asks: "what has this tenant spent, most recent
-- first". The same shape as `residency_log_tenant_ts_idx` because it is the same
-- question about a different fact.
CREATE INDEX IF NOT EXISTS cost_ledger_tenant_ts_idx
    ON {schema}.cost_ledger (tenant, ts DESC);

-- The daily rollup. MATERIALISED because a console page aggregating the whole
-- ledger on every load would scan a month of rows per visitor; refreshed by
-- `make refresh-rollup` rather than on write, so the write path stays one INSERT.
--
-- The day is the UTC date of `ts`, not the server's local one: a gateway and a
-- database in different zones would otherwise disagree about which day a
-- late-evening request belongs to, and the number would drift by a few hours'
-- spend at every boundary.
CREATE MATERIALIZED VIEW IF NOT EXISTS {schema}.cost_ledger_daily AS
    SELECT
        (ts AT TIME ZONE 'UTC')::date       AS day,
        tenant,
        alias,
        provider,
        model,
        region,
        count(*)                            AS requests,
        count(*) FILTER (WHERE cached)      AS cached_requests,
        sum(input_tokens)                   AS input_tokens,
        sum(output_tokens)                  AS output_tokens,
        sum(cost_usd)                       AS cost_usd
    FROM {schema}.cost_ledger
    GROUP BY 1, tenant, alias, provider, model, region
    WITH NO DATA;

-- A UNIQUE index over the WHOLE group key, which is what makes
-- `REFRESH MATERIALIZED VIEW CONCURRENTLY` legal at all. An index missing one
-- column is accepted here and refused at refresh time, and only at refresh time,
-- so `test_the_first_refresh_populates_and_the_second_runs_concurrently` is what
-- proves it: the first refresh cannot be concurrent, the second one must be.
CREATE UNIQUE INDEX IF NOT EXISTS cost_ledger_daily_key_idx
    ON {schema}.cost_ledger_daily (day, tenant, alias, provider, model, region);
