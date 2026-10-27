"""The command-line tools: the ledger maintenance jobs and the eval runner.

`ledger` holds `migrate` (the `ops-migrate` script a deploy step runs);
`evals` holds the `ops-eval` command CI and the nightly job shell over. Neither
is imported by the running gateway: they are entry points, not a request path.
"""
