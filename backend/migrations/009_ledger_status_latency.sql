-- The HTTP status and the time a request took (ADR 0054).
--
-- `status` is NOT NULL with a default of 200: every row written before this
-- migration was a served request, so 200 is true for them. `latency_ms` is
-- nullable: old rows and embedding rows have no time to show, and the console
-- reads null as n/a.
--
-- A request that failed with a 5xx now books a row too. Every number the console
-- shows counts served requests only, so the daily rollup is rebuilt with
-- `status < 500`. It is filled again at once when it held data or the ledger has rows, so
-- Daily cost keeps its days; a fresh database with no rows stays unfilled, as it did before.

ALTER TABLE {schema}.cost_ledger ADD COLUMN IF NOT EXISTS status smallint NOT NULL DEFAULT 200;
ALTER TABLE {schema}.cost_ledger ADD COLUMN IF NOT EXISTS latency_ms integer;

CREATE TEMP TABLE IF NOT EXISTS rollup_was_populated (populated boolean) ON COMMIT DROP;
INSERT INTO rollup_was_populated
    SELECT coalesce(
        (SELECT ispopulated FROM pg_matviews
          WHERE schemaname = '{schema}' AND matviewname = 'cost_ledger_daily'),
        false);

DROP MATERIALIZED VIEW IF EXISTS {schema}.cost_ledger_daily;

CREATE MATERIALIZED VIEW {schema}.cost_ledger_daily AS
    SELECT
        (ts AT TIME ZONE 'UTC')::date       AS day,
        tenant,
        alias,
        provider,
        model,
        count(*)                            AS requests,
        count(*) FILTER (WHERE cached)      AS cached_requests,
        sum(input_tokens)                   AS input_tokens,
        sum(output_tokens)                  AS output_tokens,
        sum(cost_usd)                       AS cost_usd
    FROM {schema}.cost_ledger
    WHERE status < 500
    GROUP BY 1, tenant, alias, provider, model
    WITH NO DATA;

-- The whole group key, so `REFRESH MATERIALIZED VIEW CONCURRENTLY` stays legal.
CREATE UNIQUE INDEX cost_ledger_daily_key_idx
    ON {schema}.cost_ledger_daily (day, tenant, alias, provider, model);

DO $$
BEGIN
    IF (SELECT populated FROM rollup_was_populated LIMIT 1)
       OR EXISTS (SELECT 1 FROM {schema}.cost_ledger) THEN
        REFRESH MATERIALIZED VIEW {schema}.cost_ledger_daily;
    END IF;
END
$$;
