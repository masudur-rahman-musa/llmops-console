-- The incidents table goes with the console's Incidents tab (ADR 0049).
--
-- This cannot be undone: every incident row is lost. Incidents are now written
-- up as postmortems in docs/postmortems/. The releases table stays, because the
-- Quality tab's release markers and blocked releases read it.
--
-- Dropping the table drops its index with it.

DROP TABLE IF EXISTS {schema}.incidents;
