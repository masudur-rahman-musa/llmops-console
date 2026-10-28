-- The console works out each day straight from `cost_ledger`, so the
-- daily rollup goes. It held only derived data: every ledger row stays.

DROP MATERIALIZED VIEW IF EXISTS {schema}.cost_ledger_daily;
