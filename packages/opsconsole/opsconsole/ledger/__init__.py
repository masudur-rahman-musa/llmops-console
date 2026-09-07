"""The Postgres write path: residency evidence now, cost and eval rows later.

the design gives this package the project's Postgres writes, and an earlier change is the
first of them, so the connection pool, the migration runner and the schema
convention arrive here rather than in the one table that needed them first.

Two modules: `db` is the seam and the runner, `residency` is the log itself.
`cli` is not re-exported - it is what `make migrate` and `make purge-residency`
call, and nothing in the gateway imports it.
"""

from opsconsole.ledger.db import (
    DEFAULT_SCHEMA as DEFAULT_SCHEMA,
)
from opsconsole.ledger.db import (
    apply_migrations as apply_migrations,
)
from opsconsole.ledger.db import (
    build_pool as build_pool,
)
from opsconsole.ledger.residency import (
    RESIDENCY_LOG_RETENTION_DAYS as RESIDENCY_LOG_RETENTION_DAYS,
)
from opsconsole.ledger.residency import (
    InMemoryResidencyLogWriter as InMemoryResidencyLogWriter,
)
from opsconsole.ledger.residency import (
    PostgresResidencyLogWriter as PostgresResidencyLogWriter,
)
from opsconsole.ledger.residency import (
    ResidencyLogRow as ResidencyLogRow,
)
from opsconsole.ledger.residency import (
    ResidencyLogWriter as ResidencyLogWriter,
)
from opsconsole.ledger.residency import (
    partition_name as partition_name,
)
from opsconsole.ledger.residency import (
    record_residency as record_residency,
)
from opsconsole.ledger.residency import (
    region_allowed as region_allowed,
)
