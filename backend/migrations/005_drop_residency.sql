-- The old location log and every `region` column go.
--
-- This cannot be undone: the old region values are lost. The daily rollup is
-- rebuilt without `region`, so it starts empty and fills on the next
-- `make refresh-rollup`.
--
-- Dropping the parent table drops its partitions and indexes with it.

DROP TABLE IF EXISTS {schema}.residency_log CASCADE;

DROP MATERIALIZED VIEW IF EXISTS {schema}.cost_ledger_daily;

ALTER TABLE {schema}.cost_ledger DROP COLUMN IF EXISTS region;
ALTER TABLE {schema}.eval_cases DROP COLUMN IF EXISTS region;

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
    GROUP BY 1, tenant, alias, provider, model
    WITH NO DATA;

-- The whole group key, so `REFRESH MATERIALIZED VIEW CONCURRENTLY` is legal.
CREATE UNIQUE INDEX cost_ledger_daily_key_idx
    ON {schema}.cost_ledger_daily (day, tenant, alias, provider, model);
