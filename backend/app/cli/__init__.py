"""The command-line tools: the ledger maintenance jobs and the eval runner.

`ledger` holds `migrate` and `refresh_rollup` (the `ops-migrate`
and `ops-refresh-rollup` scripts a scheduled job runs);
`evals` holds the `ops-eval` command CI and the nightly job shell over. Neither
is imported by the running gateway: they are entry points, not a request path.
"""
