r"""Put the raw values back: the read half of what `redact` produced.

`redact` answers a text full of `<KIND_N>` placeholders and a mapping from each
placeholder to the value it stood in for. This module holds the one function
that walks that mapping in the other direction, and nothing else: no store, no
tenant, no HTTP. Which map to use, whose it is and whether the caller asked for
it at all are the route's questions, answered in
`opsconsole.gateway.routes.chat`.

The substitution is ONE `re.sub` over an alternation of the mapping's keys, not
a loop of `str.replace` per key. The difference is visible whenever a restored
value contains something placeholder-shaped: the loop rescans the text it has
already rewritten and substitutes inside its own output, while a single pass
cannot look at a span it has just emitted
(`tests/redaction/test_rehydrate.py::test_a_restored_value_containing_a_placeholder_is_not_substituted_again`).

The alternation is built from the mapping's OWN keys, `re.escape`d, so a
placeholder the map does not name is not matched at all and survives exactly as
the model wrote it - an implementation scanning for the generic `<KIND_N>` shape
and looking each hit up would have to decide what to do with a miss, and every
answer to that is worse than leaving it alone
(`test_a_placeholder_absent_from_the_map_is_left_alone`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping


def rehydrate(text: str, mapping: Mapping[str, str]) -> str:
    r"""`text` with every placeholder `mapping` names replaced by its raw value.

    One pass, one compiled alternation, and the replacement is a function rather
    than a template string so a raw value carrying a backslash or a `\g<1>` is
    inserted literally instead of being read as a group reference.

    An empty mapping is the identity and compiles nothing: an alternation built
    from no branches is the empty pattern, which matches at every position and
    would rewrite a string that has no placeholder in it at all
    (`test_an_empty_map_is_the_identity`).

    Branches are sorted longest-first. Python's alternation is first-match, not
    longest-match, so with `<EMAIL_1>` before `<EMAIL_11>` the shorter branch
    would win on the longer token and leave a stray `1>` behind. Every key ends
    in `>`, so the escaped branches cannot in fact overlap and the order is
    belt-and-braces rather than load-bearing - stated here because the test that
    covers the pair (`test_a_longer_placeholder_is_not_eaten_by_a_shorter_one`)
    stays green with the sort removed, and a reader is owed the reason.
    """
    if not mapping:
        return text
    alternation = "|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True))
    return re.sub(alternation, lambda match: mapping[match.group(0)], text)
