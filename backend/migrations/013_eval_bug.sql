-- A fifth play result and the judge's answer to whose fault a play is.
--
-- `eval_bug` is a play the eval itself cut short (a help desk turn failed, the simulated
-- person gave nothing, or a script ran out before any right end state). It is not scored,
-- so it needs its own word rather than hiding among the plays that need a look. The check
-- is widened by exactly that one word; any other is still refused.
--
-- `fault` is the judge's one-word answer for a play that needs a look (`bot`, `case`,
-- `eval` or `fine`) and `fault_reason` its one sentence; an eval bug play keeps its plain
-- reason there with no `fault`. Both are NULL for every play stored before this, and for a
-- play nobody asked about. No score reads them.
--
-- `{schema}` is substituted by `apply_migrations`, which refuses any schema name that is
-- not a lowercase slug before it reads this file.

ALTER TABLE {schema}.eval_plays DROP CONSTRAINT IF EXISTS eval_plays_result_check;
ALTER TABLE {schema}.eval_plays ADD CONSTRAINT eval_plays_result_check
    CHECK (result IN ('setup_wrong', 'fail', 'pass', 'needs_a_look', 'eval_bug'));

ALTER TABLE {schema}.eval_plays ADD COLUMN IF NOT EXISTS fault text
    CONSTRAINT eval_plays_fault_check CHECK (fault IN ('bot', 'case', 'eval', 'fine'));
ALTER TABLE {schema}.eval_plays ADD COLUMN IF NOT EXISTS fault_reason text;
