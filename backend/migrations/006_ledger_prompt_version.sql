-- The prompt version a request was sent under.
--
-- NULLABLE on purpose: `X-Prompt-Version` is an optional header, and a request
-- sent without it is still a request that cost money. Every row written before
-- this migration, and every writer that names no `prompt_version`, reads back
-- null, which the console shows as "none sent". The daily rollup does not group
-- by it, so the view is left as it is.

ALTER TABLE {schema}.cost_ledger ADD COLUMN IF NOT EXISTS prompt_version text;
