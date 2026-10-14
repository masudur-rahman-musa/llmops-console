r"""One tenant's monthly spend: where it is counted, and what it decides (#27).

The gateway prices every served request already - `app.limits.cost` turns
a usage report into a `Decimal`, and `app.storage` files it. This module is
the running TOTAL of those rows for the month in progress, and the two thresholds
the design puts on it: a warning at 80% of the tenant's `budget_usd_month` and a
refusal at 100%.

Three things here are decisions rather than mechanics.

THE COUNTER IS INTEGER MICRO-USD, never a float and never `INCRBYFLOAT` (plan
15a decision C). The ledger quantizes every row to `COST_QUANTUM`, six decimal
places, so one micro-USD is exactly one ledger quantum and a counter in this unit
can hold a month of rows with no drift at all. A float counter would reintroduce
precisely the error the `Decimal` ledger exists to avoid, and it would do it
silently, in the one number an operator uses to refuse traffic
(`test_micro_usd_rounding_matches_the_ledger_quantum`).

A STORE THAT CANNOT ANSWER SERVES THE REQUEST. Both store
methods answer `None` for "unknown" rather than raising or answering 0, and the
route treats unknown as serve-and-move-no-gauge. Failing closed would turn a
Redis blip into a total outage on traffic every provider would have served, which
is a worse failure than a tenant briefly overspending; answering 0 would be worse
still, because 0 is a spend, and every dashboard would read healthy for exactly
as long as the incident lasted. This is the response cache's "a failure is a
miss" argument applied to a counter, and it is written here so nobody later
"fixes" it (`test_an_unreachable_redis_serves_and_moves_no_gauge` in
tests/limits/test_redis_budget.py,
`test_an_unreachable_store_serves_and_moves_no_gauge` in
tests/gateway/test_budget_enforcement.py).

Serving is not the same as saying nothing. Every swallowed failure is logged at
ERROR with the tenant and the month, because the whole cost of this decision is
that a cap silently stops being enforced, and the log is the only place that
window is visible at all - the gauge, by the same decision, deliberately does not
move. A debit lost that way is money spent against a counter that never rose
(`test_every_swallowed_failure_is_logged_with_the_tenant_and_the_month`,
`test_a_stored_value_that_is_not_an_integer_is_logged_before_it_reads_unknown`).

THE OVERRIDE IS SCOPED TO A TENANT AND A MONTH.
`verify_override` never reads the tenant out of the token - it takes it from the
authenticated context - so a captured token is useless for anyone else, and it
carries the month it was signed for, so it is dead at the month boundary. the design
says "signed with the admin key"; the deviation and its reasoning are a design note.

Both key segments are pre-checked before any command is sent,
and the check lives in `month_key` rather than in each store method, because the
key is the one thing every command needs first: there is no order in which a
command reaches Redis with an unchecked tenant or month
(`test_a_malformed_tenant_is_refused_before_any_command`,
`test_a_malformed_month_is_refused_before_any_command`).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Final, Protocol

import redis

from app.core.validation import IDENTIFIER_PATTERN, MAX_IDENTIFIER_LENGTH
from app.limits.cost import COST_QUANTUM
from app.model_routing.errors import IdentifierInvalid

BUDGET_KEY_PREFIX: Final[str] = "budget"
"""The first segment of every key a monthly counter is stored under.

A running gateway's counters are addressed by this string, so renaming it
restarts every tenant's spend at zero mid-month and hands each of them a second
budget; pinned by `test_the_key_prefix_is_the_published_one`."""

BUDGET_KEY_TTL_SECONDS: Final[int] = 5_356_800
"""How long a month's counter is kept, in seconds: 62 days, two long months.

A counter is only ever read for the month in progress, so the TTL exists to stop
one key per tenant per month accumulating for ever. Both
wrong values are silent: shorter than a month and a counter expires mid-month,
resetting a tenant's spend to zero; absent and nothing is ever reclaimed. Sixty-
two days is two of the longest months, so a counter written on the first of a
31-day month survives the whole of the next one. Pinned exactly by
`test_the_budget_key_ttl_is_pinned`, and read back off a live server by
`test_the_key_expires_after_exactly_the_published_ttl`."""

MONTH_PATTERN: Final[re.Pattern[str]] = re.compile(r"\d{4}-(?:0[1-9]|1[0-2])")
r"""What a month must look like before it is allowed to shape a key.

`YYYY-MM`, with the month alternation written out so `2026-00` and `2026-13` -
periods no request will ever be booked under again - are refused rather than
silently opening a counter nobody reads. Used with `fullmatch` and never with
`match`: in Python `$` also matches before a trailing newline,
so `match` would let `2026-06\n` put a caller-controlled line break into a Redis
key. The rejection table - a one-digit month, month zero, month thirteen, a
three- and a five-digit year, an underscore separator, no separator, a day
appended, a colon, non-digits, a trailing newline, a trailing CRLF, a trailing
space, a leading space and a glob character - is
`test_a_malformed_month_is_refused_before_any_command`, and the pattern object
itself is asserted by `test_the_month_pattern_is_used_with_fullmatch_semantics`."""

MICRO_USD_PER_USD: Final[int] = 1_000_000
"""How many counter units make one dollar - one micro-USD, the ledger's quantum.

`Decimal(1) / MICRO_USD_PER_USD == COST_QUANTUM` is asserted alongside this
constant by `test_micro_usd_rounding_matches_the_ledger_quantum`, which is what
makes decision C's claim - that the counter can hold a ledger row exactly -
something a test settles rather than something this docstring asserts."""

WARN_RATIO: Final[Decimal] = Decimal("0.8")
"""The fraction of the cap at which a tenant is warned, inclusive.

A `Decimal` rather than a float, and compared against the cap rather than against
a threshold computed once by integer division: 80% of a cap of 9 micro-USD is
7.2, and `cap * 8 // 10` is 7, which warns a micro early on every cap that is not
a multiple of five (the two fractional rows of `test_the_decision_table`)."""


class BudgetDecision(StrEnum):
    """What one spend against one cap means for the request in hand.

    Three members and not a boolean: `WARN` is served, so an implementation that
    collapsed the two thresholds into one comparison would either refuse at 80%
    or never warn at all (`test_the_decision_table`).
    """

    SERVE = "serve"
    WARN = "warn"
    REJECT = "reject"


def current_month(now: datetime | None = None) -> str:
    """The `YYYY-MM` a spend made now is booked under, in UTC.

    UTC and not the process's local time: at 23:30 on the last day of a month in
    UTC+2 the local month is already the next one, so half an hour of spend would
    be booked under a key nothing reads and every cap would be briefly
    unenforceable (`test_the_current_month_is_the_utc_month_not_the_local_one`).
    An aware datetime handed in is converted rather than trusted.

    `now` has a default because no caller has a clock to pass: `create_app` and
    both routes take none (`test_the_current_month_defaults_to_now`).
    """
    moment = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return moment.strftime("%Y-%m")


def month_key(tenant: str, month: str, *, prefix: str = BUDGET_KEY_PREFIX) -> str:
    """The key one tenant's spend for one month is counted under, both segments checked.

    `{prefix}:{tenant}:{YYYY-MM}` (`test_the_month_key_is_the_specced_shape`).
    Both segments are client-influenced strings that shape a key, and a colon in
    either would forge a segment and address another tenant's counter, so
    `tenant` is checked against the gateway's OWN `IDENTIFIER_PATTERN` and
    `MAX_IDENTIFIER_LENGTH` - imported rather than restated, so the name a
    counter accepts and the name `gateway/context.py` authenticated can never
    disagree - and `month` against `MONTH_PATTERN`. A rejected value raises
    `IdentifierInvalid` and no command is sent at all, which is what the tests
    assert rather than "no key exists afterwards"
    (`test_debit_refuses_a_malformed_tenant_before_touching_redis`,
    `test_spend_refuses_a_malformed_tenant_before_touching_redis`,
    `test_a_malformed_month_reaches_no_command_on_either_method`).

    `prefix` exists for the suite: per-test isolation against a live Redis is a
    namespace rather than a `FLUSHDB` that would delete keys the suite does not
    own.
    """
    if len(tenant) > MAX_IDENTIFIER_LENGTH:
        raise IdentifierInvalid(f"tenant: longer than {MAX_IDENTIFIER_LENGTH} characters")
    if IDENTIFIER_PATTERN.fullmatch(tenant) is None:
        raise IdentifierInvalid("tenant: not a lowercase slug")
    if MONTH_PATTERN.fullmatch(month) is None:
        raise IdentifierInvalid(f"month: {month!r} is not a YYYY-MM period")
    return f"{prefix}:{tenant}:{month}"


def to_micro_usd(usd: Decimal) -> int:
    """One cost as the counter's own unit, rounded the way the ledger rounds.

    `ROUND_HALF_UP` at `COST_QUANTUM`, exactly as `app.limits.cost`
    quantizes a row, so the counter and the ledger agree on every value rather
    than on most of them. `int(usd * 1_000_000)` would truncate instead: a row of
    0.0000005 books 1 in the ledger and would add 0 here, and the counter would
    drift one quantum below the ledger on every such request for ever
    (`test_micro_usd_rounding_matches_the_ledger_quantum`).
    """
    return int(usd.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP) * MICRO_USD_PER_USD)


def decide(*, spend_micro_usd: int, cap_micro_usd: int) -> BudgetDecision:
    """What a tenant's spend so far means for the request in hand.

    BOTH boundaries are inclusive, which is the whole of criteria 2 and 3: a
    spend of exactly 80% of the cap warns, and a spend exactly equal to the cap
    is refused. Written `>=` on both, because `>` on either serves one request a
    month that the criteria say is the 429 and never warns the tenant whose spend
    lands precisely on the threshold (`test_the_decision_table`,
    `test_the_warn_boundary_is_inclusive_at_eighty_percent`,
    `test_the_reject_boundary_is_inclusive_at_the_cap`).

    The warn threshold is a comparison against `cap * WARN_RATIO` in `Decimal`
    rather than an integer threshold computed once: see `WARN_RATIO`.

    The spend is what has ALREADY been booked, never this request's cost added to
    it. That is an earlier change decision E: the request that crosses the cap completes
    and the NEXT one is refused, because reserving an estimate up front would
    need a token count the gateway does not have until the provider has answered,
    and would refuse requests on a guess
    (`test_a_tenant_one_micro_under_the_cap_is_served`).
    """
    if spend_micro_usd >= cap_micro_usd:
        return BudgetDecision.REJECT
    if Decimal(spend_micro_usd) >= Decimal(cap_micro_usd) * WARN_RATIO:
        return BudgetDecision.WARN
    return BudgetDecision.SERVE


def budget_ratio(*, spend_micro_usd: int, cap_micro_usd: int) -> float:
    """What `gateway_budget_ratio{tenant}` reads: spend divided by cap.

    A fraction and not the raw total, so one alert threshold and one dashboard
    panel work for every tenant whatever its cap
    (`test_the_ratio_is_spend_over_cap`).

    NOT clamped to 1.0. The gauge is what an operator reads to see how far past
    its cap a tenant with a standing override has gone, and a clamp would make
    100% and 400% the same picture (`test_the_ratio_keeps_reading_above_one_over_cap`).

    A cap of zero has no ratio to report, so it answers 0.0 rather than raising:
    every such request is refused by `decide` anyway, and the one that is served
    carries an override, which must not turn into a 500 in the middle of booking
    its cost (`test_a_zero_cap_has_no_ratio_rather_than_a_division_error`).
    """
    if cap_micro_usd <= 0:
        return 0.0
    return spend_micro_usd / cap_micro_usd


_log = logging.getLogger(__name__)


def to_usd(micro_usd: int) -> Decimal:
    """One counter reading back as dollars, at the ledger's own quantum.

    The inverse of `to_micro_usd`, used only where a number leaves the counter for
    a human to read - the warning below - so an operator sees `8.000000` and not
    `8000000` (`test_a_tenant_at_eighty_percent_is_served_and_warned_once`).
    """
    return (Decimal(micro_usd) / MICRO_USD_PER_USD).quantize(COST_QUANTUM)


def warn_budget(*, tenant: str, spend_micro_usd: int, cap_micro_usd: int) -> None:
    """Log the one warning a tenant at or above 80% of its cap gets (criterion 2).

    ONE record per request, from this one call site, so an alert built on it
    counts requests rather than call sites
    (`test_a_tenant_at_eighty_percent_is_served_and_warned_once`), and none at all
    below the boundary (`test_a_tenant_just_under_eighty_percent_is_served_without_a_warning`).

    The tenant is in the message AND in `extra`: an alert that cannot say WHOSE
    budget is nearly gone is one nobody can act on, and a log pipeline that
    indexes structured fields should not have to parse the sentence to find it.
    The three values are what the criterion names - the tenant, the spend and the
    cap - with the two figures in dollars, which is the unit the cap is written
    in.
    """
    _log.warning(
        "tenant %s has spent %s of its %s monthly budget",
        tenant,
        to_usd(spend_micro_usd),
        to_usd(cap_micro_usd),
        extra={
            "tenant": tenant,
            "spend_usd": to_usd(spend_micro_usd),
            "cap_usd": to_usd(cap_micro_usd),
        },
    )


BUDGET_OVERRIDE_HEADER: Final[str] = "X-Gateway-Budget-Override"
"""The header a request over its cap carries to be served anyway.

Pinned by `test_the_override_header_name_is_the_published_one`: a rename leaves
every token an operator has minted unusable, and the gateway would keep answering
429 with nothing to say the header was read under another name."""

MAX_OVERRIDE_HEADER_CHARS: Final[int] = 72
"""The longest an override header may be before it is refused unread.

Exactly a well-formed token: seven characters of month, one colon and sixty-four
of hex. Pinned against the thing it has to admit by
`test_the_override_header_cap_is_exactly_a_well_formed_token`, so a cap set below
a real token - which would refuse every legitimate override while looking like a
signature failure - fails there."""

OVERRIDE_SIGNATURE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
r"""What the signature half of a token must look like before it is compared.

`HMAC-SHA256` rendered as lowercase hex, which is what the mint produces. The
check is not decoration: `hmac.compare_digest` on `str` operands requires both to
be ASCII and raises `TypeError` otherwise, so a header carrying any non-ASCII
character would reach the comparison and crash the request rather than being
refused. Used with `fullmatch` for a project rule's reason, and pinned by
`test_the_override_signature_pattern_admits_a_minted_signature_and_nothing_else`."""


def verify_override(header: str | None, *, tenant: str, month: str, secret: str) -> bool:
    """Whether this header is a token THIS gateway minted, for THIS tenant and month.

    The token is `{YYYY-MM}:{hex64}`, where the hex is
    `HMAC-SHA256(secret, f"{tenant}:{month}")`. Four properties make it worth
    more than a shared password, and each is a row of `test_the_override_table`:

    * The TENANT is a parameter, taken from the authenticated context, and is
      never read out of the token. A captured token is therefore useless for
      anyone else; an implementation that read it from the header would have
      turned one leaked token into a universal bypass (an earlier change's second risk,
      `test_a_token_minted_for_another_tenant_is_still_refused`).
    * The MONTH is signed AND compared. The header states a month, but the
      signature is verified against the month the GATEWAY is in, so neither last
      month's token nor a token whose month field was edited is accepted
      (`test_a_token_signed_for_last_month_but_stating_this_one_is_still_refused`).
    * An EMPTY secret refuses everything outright. The shipped default of
      `budget_override_secret` is `""`, and HMAC over an empty key is still
      perfectly good HMAC, so a deployment that never set the variable would
      otherwise honour tokens anyone who knows the scheme could mint
      (`test_a_gateway_with_no_override_secret_refuses_a_token_minted_from_the_empty_one`).
    * The LENGTH is capped before anything is parsed, so a ten-thousand character
      header whose first seventy-two characters are a genuine token is refused
      unread rather than sliced
      (`test_an_over_length_override_header_is_refused_before_it_is_parsed`).

    Answers `False` for every failure and raises for none: a caller cannot tell a
    malformed token from a wrong signature, and either way the answer is the same
    429 the request would have got with no header at all.
    """
    if not secret or header is None:
        return False
    if len(header) > MAX_OVERRIDE_HEADER_CHARS:
        return False
    stated, separator, signature = header.partition(":")
    if not separator or stated != month:
        return False
    if OVERRIDE_SIGNATURE_PATTERN.fullmatch(signature) is None:
        return False
    expected = hmac.new(
        secret.encode("utf-8"), f"{tenant}:{month}".encode(), hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(signature, expected)


class BudgetStore(Protocol):
    """Add to a tenant's spend for a month; read what it has spent so far.

    The contract is two methods and one rule: `None` means UNKNOWN, and it is the
    only thing a store that cannot answer may say. Not an exception, which would
    make a Redis blip a 5xx on traffic every provider would have served; and not
    `0`, which is a spend - a route handed 0 during an incident would publish a
    confident `gateway_budget_ratio` of 0.0 for every tenant and paint the one
    dashboard an operator is looking at healthy (decision B,
    `test_an_unreachable_redis_serves_and_moves_no_gauge`).

    `debit` answers the RUNNING TOTAL after its own increment, not the amount it
    added, so the gauge a route publishes includes the request just served and
    needs no second read (`test_the_gauge_reads_the_ratio_after_a_served_request`).
    """

    def spend(self, tenant: str, month: str) -> int | None:
        """This tenant's spend for `month` in micro-USD, 0 if none, `None` if unknown."""
        ...

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        """Add `micro_usd` and answer the new total, or `None` if it could not be added."""
        ...


class InProcessBudgetStore:
    """A `BudgetStore` counting in one process's memory.

    What `create_app` keeps when `OPSCONSOLE_REDIS_URL` is unset, exactly as the
    response cache and the directory store do, so the suite and a local run need
    no backing service. Two gateway replicas do not share a counter this way, so
    each would enforce its own share of a cap: acceptable for a single G0 node and
    the reason `RedisBudgetStore` below is the one that ships to the cluster.

    It never answers `None` (`test_the_in_process_twin_never_answers_unknown`):
    a dictionary cannot be unreachable, so "unknown" has no meaning here, and a
    route that saw `None` from it would skip a gauge it could have published.
    """

    def __init__(self) -> None:
        self._totals: dict[str, int] = {}

    def spend(self, tenant: str, month: str) -> int | None:
        """This tenant's spend so far, 0 for a month it has not spent anything in."""
        return self._totals.get(month_key(tenant, month), 0)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        """Add to this month's counter and answer the running total.

        A debit of zero is a debit: it books nothing and still answers the total,
        which is what a cache hit needs (`test_a_zero_debit_still_answers_the_running_total`).
        """
        key = month_key(tenant, month)
        total = self._totals.get(key, 0) + micro_usd
        self._totals[key] = total
        return total


class RedisBudgetStore:
    """A `BudgetStore` counting in Redis, where every replica adds to one total.

    The implementation the gateway ships with once `OPSCONSOLE_REDIS_URL` is set.
    It differs from the in-process twin in three ways that matter.

    The increment is `INCRBY` on an integer, never `INCRBYFLOAT` (decision C):
    the counter is micro-USD, and `INCRBY` is atomic, so two replicas booking at
    the same instant add up rather than overwrite each other.

    The expiry is re-sent on every debit, immediately after the increment
    (`test_a_debit_sets_the_expiry_in_the_same_call_sequence`,
    `test_a_second_debit_does_not_drop_the_expiry`). `INCRBY` on a missing key
    creates one with no TTL, so an `EXPIRE` that ran only on the first debit
    would leave every counter immortal whenever the first debit's `EXPIRE` was
    the command that failed.

    Every `redis.RedisError` is swallowed and becomes `None` - decision B, argued
    in the module docstring. The catch is on the base class and not on
    `ConnectionError`: a timeout, a `READONLY` reply from a demoted replica and an
    authentication failure are all "the counter cannot answer", and catching one
    of them would leave the other three escaping into a request
    (the four rows of `test_an_unreachable_redis_serves_and_moves_no_gauge`).
    A stored value that is not an integer is unknown for the same reason
    (`test_a_stored_value_that_is_not_an_integer_reads_unknown_rather_than_zero`).

    Not annotated `redis.Redis[bytes]`: the installed client is not generic at
    runtime, so a parametrised annotation fails `mypy --strict` with `type-arg`.
    """

    def __init__(self, client: redis.Redis, *, key_prefix: str = BUDGET_KEY_PREFIX) -> None:
        self._client = client
        self._key_prefix = key_prefix

    def key(self, tenant: str, month: str) -> str:
        """The Redis key one tenant's month is counted under, both segments checked."""
        return month_key(tenant, month, prefix=self._key_prefix)

    def spend(self, tenant: str, month: str) -> int | None:
        """This tenant's spend so far: 0 for a month with no counter, `None` if unknown.

        A missing key really is zero spend and is reported as such
        (`test_a_missing_counter_reads_zero_and_not_unknown`); only a Redis that
        could not answer, or a value that is not an integer, is unknown.
        """
        stored = self.key(tenant, month)
        try:
            raw = self._client.get(stored)
        except redis.RedisError:
            _log.exception("budget counter unreadable for tenant %s, month %s", tenant, month)
            return None
        if raw is None:
            return 0
        try:
            return int(raw)
        except (TypeError, ValueError):
            _log.error("budget counter for tenant %s, month %s holds %r", tenant, month, raw)
            return None

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        """Add `micro_usd` to this month's counter and answer the new total.

        `None` when Redis could not be reached, which the route reads as "serve,
        and move no gauge".
        """
        stored = self.key(tenant, month)
        try:
            total = int(self._client.incrby(stored, micro_usd))
            self._client.expire(stored, BUDGET_KEY_TTL_SECONDS)
        except redis.RedisError:
            _log.exception(
                "budget debit of %s lost for tenant %s, month %s", micro_usd, tenant, month
            )
            return None
        return total
