-- Tokens a provider read from, or wrote to, its prompt cache (ADR 0051).
--
-- NOT NULL with a default of 0: a provider that reports no cache, and every row
-- written before this migration, reads back 0. They are priced at their own
-- rates, apart from `input_tokens`. The daily rollup does not group by them, so
-- the view is left as it is.

ALTER TABLE {schema}.cost_ledger ADD COLUMN IF NOT EXISTS cache_read_tokens integer NOT NULL DEFAULT 0;
ALTER TABLE {schema}.cost_ledger ADD COLUMN IF NOT EXISTS cache_write_tokens integer NOT NULL DEFAULT 0;
