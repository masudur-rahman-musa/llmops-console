"""The Postgres write path: residency evidence and cost rows now, eval rows later.

the design gives this package the project's Postgres writes, and an earlier change is the
first of them, so the connection pool, the migration runner and the schema
convention arrive here rather than in the one table that needed them first.

The modules: `db` is the seam and the runner, `residency_log` is the decision
log, `cost_ledger` is the spend row and its daily rollup, `console_events` is the
two records an operator and CI write by hand through `/admin/*`, `eval_store` is
the eval run and case rows, and `dashboard_reads` is the console's read side. The
command-line entry points are not re-exported - they live in `app.cli.ledger`,
which `make migrate`, `make purge-residency` and `make refresh-rollup` call.
"""

from app.storage.console_events import (
    ConsoleEventWriter as ConsoleEventWriter,
)
from app.storage.console_events import (
    IncidentRow as IncidentRow,
)
from app.storage.console_events import (
    InMemoryConsoleEventWriter as InMemoryConsoleEventWriter,
)
from app.storage.console_events import (
    PostgresConsoleEventWriter as PostgresConsoleEventWriter,
)
from app.storage.console_events import (
    ReleaseRow as ReleaseRow,
)
from app.storage.cost_ledger import (
    CostLedgerRow as CostLedgerRow,
)
from app.storage.cost_ledger import (
    CostLedgerWriter as CostLedgerWriter,
)
from app.storage.cost_ledger import (
    InMemoryCostLedgerWriter as InMemoryCostLedgerWriter,
)
from app.storage.cost_ledger import (
    PostgresCostLedgerWriter as PostgresCostLedgerWriter,
)
from app.storage.db import (
    DEFAULT_SCHEMA as DEFAULT_SCHEMA,
)
from app.storage.db import (
    apply_migrations as apply_migrations,
)
from app.storage.db import (
    build_pool as build_pool,
)
from app.storage.residency_log import (
    RESIDENCY_LOG_RETENTION_DAYS as RESIDENCY_LOG_RETENTION_DAYS,
)
from app.storage.residency_log import (
    InMemoryResidencyLogWriter as InMemoryResidencyLogWriter,
)
from app.storage.residency_log import (
    PostgresResidencyLogWriter as PostgresResidencyLogWriter,
)
from app.storage.residency_log import (
    ResidencyLogRow as ResidencyLogRow,
)
from app.storage.residency_log import (
    ResidencyLogWriter as ResidencyLogWriter,
)
from app.storage.residency_log import (
    partition_name as partition_name,
)
