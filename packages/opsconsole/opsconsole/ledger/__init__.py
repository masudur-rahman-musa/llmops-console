"""The Postgres write path: residency evidence and cost rows now, eval rows later.

the design gives this package the project's Postgres writes, and an earlier change is the
first of them, so the connection pool, the migration runner and the schema
convention arrive here rather than in the one table that needed them first.

Four modules: `db` is the seam and the runner, `residency` is the decision log,
`cost_ledger` is the spend row and its daily rollup, and `console_events` is the
two records an operator and CI write by hand through `/admin/*`. `cli` is not
re-exported - it is what `make migrate`, `make purge-residency` and
`make refresh-rollup` call, and nothing in the gateway imports it.
"""

from opsconsole.ledger.console_events import (
    ConsoleEventWriter as ConsoleEventWriter,
)
from opsconsole.ledger.console_events import (
    IncidentRow as IncidentRow,
)
from opsconsole.ledger.console_events import (
    InMemoryConsoleEventWriter as InMemoryConsoleEventWriter,
)
from opsconsole.ledger.console_events import (
    PostgresConsoleEventWriter as PostgresConsoleEventWriter,
)
from opsconsole.ledger.console_events import (
    ReleaseRow as ReleaseRow,
)
from opsconsole.ledger.cost_ledger import (
    CostLedgerRow as CostLedgerRow,
)
from opsconsole.ledger.cost_ledger import (
    CostLedgerWriter as CostLedgerWriter,
)
from opsconsole.ledger.cost_ledger import (
    InMemoryCostLedgerWriter as InMemoryCostLedgerWriter,
)
from opsconsole.ledger.cost_ledger import (
    PostgresCostLedgerWriter as PostgresCostLedgerWriter,
)
from opsconsole.ledger.cost_ledger import (
    record_cost as record_cost,
)
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
