-- The random user id a request carried in `X-User` (ADR 0056).
--
-- Nullable: old rows and requests with no `X-User` have no user. The id is `u_`
-- and 32 hex characters, never a name or an email; only the help desk can map it
-- to a person. The daily rollup is not changed.
--
-- The partial index serves the console's top users read: per tenant, inside a
-- time window, rows with a user only.

ALTER TABLE {schema}.cost_ledger ADD COLUMN IF NOT EXISTS user_id text;

CREATE INDEX IF NOT EXISTS cost_ledger_tenant_user_ts_idx
    ON {schema}.cost_ledger (tenant, user_id, ts)
    WHERE user_id IS NOT NULL;
