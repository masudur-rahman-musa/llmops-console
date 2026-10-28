r"""One tenant's monthly spend: where it is counted, and what it decides.

The gateway prices every served request already - `app.limits.cost` turns
a usage report into a `Decimal`, and `app.storage` files it. This module is
the running TOTAL of those rows for the month in progress, and the two thresholds
on it: a warning at 80% of the tenant's `budget_usd_month` and a
refusal at 100%.

Three things here are decisions rather than mechanics.

THE COUNTER IS INTEGER MICRO-USD, never a float and never `INCRBYFLOAT` The ledger quantizes every
row to `COST_QUANTUM`, six decimal
places, so one micro-USD is exactly one ledger quantum and a counter in this unit
can hold a month of rows with no drift at all. A float counter would reintroduce
precisely the error the `Decimal` ledger exists to avoid, and it would do it
silently, in the one number an operator uses to refuse traffic.

A STORE THAT CANNOT ANSWER SERVES THE REQUEST. Both store
methods answer `None` for "unknown" rather than raising or answering 0, and the
route treats unknown as serve-and-move-no-gauge. Failing closed would turn a
Redis blip into a total outage on traffic every provider would have served, which
is a worse failure than a tenant briefly overspending; answering 0 would be worse
still, because 0 is a spend, and every dashboard would read healthy for exactly
as long as the incident lasted. This is the response cache's "a failure is a
miss" argument applied to a counter, and it is written here so nobody later
"fixes" it.

Serving is not the same as saying nothing. Every swallowed failure is logged at
ERROR with the tenant and the month, because the whole cost of this decision is
that a cap silently stops being enforced, and the log is the only place that
window is visible at all - the gauge, by the same decision, deliberately does not
move. A debit lost that way is money spent against a counter that never rose.

THE OVERRIDE IS SCOPED TO A TENANT AND A MONTH.
`verify_override` never reads the tenant out of the token - it takes it from the
authenticated context - so a captured token is useless for anyone else, and it
carries the month it was signed for, so it is dead at the month boundary. The
spec says "signed with the admin key"; this is a deliberate change from that.

Both key segments are pre-checked before any command is sent,
and the check lives in `month_key` rather than in each store method, because the
key is the one thing every command needs first: there is no order in which a
command reaches Redis with an unchecked tenant or month.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from typing import Final, Protocol

import redis

from app.core.validation import IdentifierInvalid, checked_identifier, checked_user_id
from app.limits.cost import COST_QUANTUM

BUDGET_KEY_PREFIX: Final[str] = "budget"
"""The first segment of every key a monthly counter is stored under.

A running gateway's counters are addressed by this string, so renaming it
restarts every tenant's spend at zero mid-month and hands each of them a second
budget."""

BUDGET_KEY_TTL_SECONDS: Final[int] = 5_356_800
"""How long a month's counter is kept, in seconds: 62 days, two long months.

A counter is only ever read for the month in progress, so the TTL exists to stop
one key per tenant per month accumulating for ever. Both
wrong values are silent: shorter than a month and a counter expires mid-month,
resetting a tenant's spend to zero; absent and nothing is ever reclaimed. Sixty-
two days is two of the longest months, so a counter written on the first of a
31-day month survives the whole of the next one."""

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
refused before any command is sent."""

MICRO_USD_PER_USD: Final[int] = 1_000_000
"""How many counter units make one dollar - one micro-USD, the ledger's quantum.

`Decimal(1) / MICRO_USD_PER_USD == COST_QUANTUM` is asserted alongside this
constant, so the claim that the counter can hold a ledger row exactly is
checked by a test and not only stated here."""

WARN_RATIO: Final[Decimal] = Decimal("0.8")
"""The fraction of the cap at which a tenant is warned, inclusive.

A `Decimal` rather than a float, and compared against the cap rather than against
a threshold computed once by integer division: 80% of a cap of 9 micro-USD is
7.2, and `cap * 8 // 10` is 7, which warns a micro early on every cap that is not
a multiple of five."""


class BudgetDecision(StrEnum):
    """What one spend against one cap means for the request in hand.

    Three members and not a boolean: `WARN` is served, so an implementation that
    collapsed the two thresholds into one comparison would either refuse at 80%
    or never warn at all.
    """

    SERVE = "serve"
    WARN = "warn"
    REJECT = "reject"


def current_month(now: datetime | None = None) -> str:
    """The `YYYY-MM` a spend made now is booked under, in UTC.

    UTC and not the process's local time: at 23:30 on the last day of a month in
    UTC+2 the local month is already the next one, so half an hour of spend would
    be booked under a key nothing reads and every cap would be briefly
    unenforceable.
    An aware datetime handed in is converted rather than trusted.

    `now` has a default because no caller has a clock to pass: `create_app` and
    both routes take none.
    """
    moment = datetime.now(UTC) if now is None else now.astimezone(UTC)
    return moment.strftime("%Y-%m")


def month_key(tenant: str, month: str, *, prefix: str = BUDGET_KEY_PREFIX) -> str:
    """The key one tenant's spend for one month is counted under, both segments checked.

    `{prefix}:{tenant}:{YYYY-MM}`.
    Both segments are client-influenced strings that shape a key, and a colon in
    either would forge a segment and address another tenant's counter, so
    `tenant` is checked against the gateway's OWN `IDENTIFIER_PATTERN` and
    `MAX_IDENTIFIER_LENGTH` - imported rather than restated, so the name a
    counter accepts and the name the request context authenticated can never
    disagree - and `month` against `MONTH_PATTERN`. A rejected value raises
    `IdentifierInvalid` and no command is sent at all, which is what the tests
    assert rather than "no key exists afterwards".

    `prefix` exists for the suite: per-test isolation against a live Redis is a
    namespace rather than a `FLUSHDB` that would delete keys the suite does not
    own.
    """
    checked_identifier(tenant, "tenant")
    if MONTH_PATTERN.fullmatch(month) is None:
        raise IdentifierInvalid(f"month: {month!r} is not a YYYY-MM period")
    return f"{prefix}:{tenant}:{month}"


def user_month_key(
    tenant: str, user_id: str, month: str, *, prefix: str = BUDGET_KEY_PREFIX
) -> str:
    """The key one user's spend for one month is counted under, every segment checked.

    `{prefix}-user:{tenant}:{user_id}:{YYYY-MM}`, apart from the tenant's own key so a
    user's debit never adds to the tenant total. The tenant and month get the
    `month_key` checks and the user id `checked_user_id`, all before the key exists:
    a colon in any of them would forge a segment. A test pins the key shape.
    """
    month_key(tenant, month, prefix=prefix)
    checked_user_id(user_id)
    return f"{prefix}-user:{tenant}:{user_id}:{month}"


def budget_seeded_key(tenant: str, month: str, *, prefix: str = BUDGET_KEY_PREFIX) -> str:
    """The marker saying one tenant's month counter was loaded from the ledger.

    `{prefix}-seeded:{tenant}:{YYYY-MM}`, checked the way `month_key` checks the counter's
    key, before any command. A test pins this.
    """
    month_key(tenant, month, prefix=prefix)
    return f"{prefix}-seeded:{tenant}:{month}"


def user_seeded_key(
    tenant: str, user_id: str, month: str, *, prefix: str = BUDGET_KEY_PREFIX
) -> str:
    """The marker for one user's month counter.

    `{prefix}-seeded-user:{tenant}:{user_id}:{YYYY-MM}`, every segment checked first.
    """
    user_month_key(tenant, user_id, month, prefix=prefix)
    return f"{prefix}-seeded-user:{tenant}:{user_id}:{month}"


SpendLoader = Callable[[], int | None]
"""Reads one month's spend from the ledger in micro-USD, or `None` with no database.

It may raise `app.storage.cost_ledger.LedgerUnreadable`; the stores let that out, so the
route refuses rather than counting the month from zero."""

SEED_SCRIPT: Final[str] = """
if redis.call('SET', KEYS[1], '1', 'NX', 'EX', ARGV[2]) then
  redis.call('SET', KEYS[2], ARGV[1], 'EX', ARGV[2])
end
return tonumber(redis.call('GET', KEYS[2]) or '0')
"""
"""Load a counter from the ledger once: the replica that sets the marker sets the counter.

One script, so two replicas loading at once cannot interleave: the second finds the
marker and only reads. The counter is SET, not added to, so a debit that landed before
the load is replaced by the ledger total, which holds or will hold that row. A test pins
this."""


def to_micro_usd(usd: Decimal) -> int:
    """One cost as the counter's own unit, rounded the way the ledger rounds.

    `ROUND_HALF_UP` at `COST_QUANTUM`, exactly as `app.limits.cost`
    quantizes a row, so the counter and the ledger agree on every value rather
    than on most of them. `int(usd * 1_000_000)` would truncate instead: a row of
    0.0000005 books 1 in the ledger and would add 0 here, and the counter would
    drift one quantum below the ledger on every such request for ever.
    """
    return int(usd.quantize(COST_QUANTUM, rounding=ROUND_HALF_UP) * MICRO_USD_PER_USD)


def decide(*, spend_micro_usd: int, cap_micro_usd: int) -> BudgetDecision:
    """What a tenant's spend so far means for the request in hand.

    BOTH boundaries are inclusive, which is the whole point: a
    spend of exactly 80% of the cap warns, and a spend exactly equal to the cap
    is refused. Written `>=` on both, because `>` on either serves one request a
    month that the criteria say is the 429 and never warns the tenant whose spend
    lands precisely on the threshold.

    The warn threshold is a comparison against `cap * WARN_RATIO` in `Decimal`
    rather than an integer threshold computed once: see `WARN_RATIO`.

    The spend is what has ALREADY been booked, never this request's cost added to
    it. So the request that crosses the cap completes
    and the NEXT one is refused, because reserving an estimate up front would
    need a token count the gateway does not have until the provider has answered,
    and would refuse requests on a guess.
    """
    if spend_micro_usd >= cap_micro_usd:
        return BudgetDecision.REJECT
    if Decimal(spend_micro_usd) >= Decimal(cap_micro_usd) * WARN_RATIO:
        return BudgetDecision.WARN
    return BudgetDecision.SERVE


def budget_ratio(*, spend_micro_usd: int, cap_micro_usd: int) -> float:
    """What `gateway_budget_ratio{tenant}` reads: spend divided by cap.

    A fraction and not the raw total, so one alert threshold and one dashboard
    panel work for every tenant whatever its cap.

    NOT clamped to 1.0. The gauge is what an operator reads to see how far past
    its cap a tenant with a standing override has gone, and a clamp would make
    100% and 400% the same picture.

    A cap of zero has no ratio to report, so it answers 0.0 rather than raising:
    every such request is refused by `decide` anyway, and the one that is served
    carries an override, which must not turn into a 500 in the middle of booking
    its cost.
    """
    if cap_micro_usd <= 0:
        return 0.0
    return spend_micro_usd / cap_micro_usd


_log = logging.getLogger(__name__)


def to_usd(micro_usd: int) -> Decimal:
    """One counter reading back as dollars, at the ledger's own quantum.

    The inverse of `to_micro_usd`, used only where a number leaves the counter for
    a human to read - the warning below - so an operator sees `8.000000` and not
    `8000000`.
    """
    return (Decimal(micro_usd) / MICRO_USD_PER_USD).quantize(COST_QUANTUM)


def warn_budget(*, tenant: str, spend_micro_usd: int, cap_micro_usd: int) -> None:
    """Log the one warning a tenant at or above 80% of its cap gets.

    ONE record per request, from this one call site, so an alert built on it
    counts requests rather than call sites, and none at all
    below the boundary.

    The tenant is in the message AND in `extra`: an alert that cannot say WHOSE
    budget is nearly gone is one nobody can act on, and a log pipeline that
    indexes structured fields should not have to parse the sentence to find it.
    The three values are the tenant, the spend and the
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

A rename leaves
every token an operator has minted unusable, and the gateway would keep answering
429 with nothing to say the header was read under another name."""

MAX_OVERRIDE_HEADER_CHARS: Final[int] = 72
"""The longest an override header may be before it is refused unread.

Exactly a well-formed token: seven characters of month, one colon and sixty-four
of hex. A cap set below
a real token would refuse every legitimate override while looking like a
signature failure."""

OVERRIDE_SIGNATURE_PATTERN: Final[re.Pattern[str]] = re.compile(r"[0-9a-f]{64}")
r"""What the signature half of a token must look like before it is compared.

`HMAC-SHA256` rendered as lowercase hex, which is what the mint produces. The
check is not decoration: `hmac.compare_digest` on `str` operands requires both to
be ASCII and raises `TypeError` otherwise, so a header carrying any non-ASCII
character would reach the comparison and crash the request rather than being
refused. Used with `fullmatch` for the project's reason."""


def verify_override(header: str | None, *, tenant: str, month: str, secret: str) -> bool:
    """Whether this header is a token THIS gateway minted, for THIS tenant and month.

    The token is `{YYYY-MM}:{hex64}`, where the hex is
    `HMAC-SHA256(secret, f"{tenant}:{month}")`. Four properties make it worth
    more than a shared password:

    * The TENANT is a parameter, taken from the authenticated context, and is
      never read out of the token. A captured token is therefore useless for
      anyone else; an implementation that read it from the header would have
      turned one leaked token into a universal bypass.
    * The MONTH is signed AND compared. The header states a month, but the
      signature is verified against the month the GATEWAY is in, so neither last
      month's token nor a token whose month field was edited is accepted.
    * An EMPTY secret refuses everything outright. The shipped default of
      `budget_override_secret` is `""`, and HMAC over an empty key is still
      perfectly good HMAC, so a deployment that never set the variable would
      otherwise honour tokens anyone who knows the scheme could mint.
    * The LENGTH is capped before anything is parsed, so a ten-thousand character
      header whose first seventy-two characters are a genuine token is refused
      unread rather than sliced.

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
    dashboard an operator is looking at healthy.

    `debit` answers the RUNNING TOTAL after its own increment, not the amount it
    added, so the gauge a route publishes includes the request just served and
    needs no second read.
    """

    def spend(self, tenant: str, month: str) -> int | None:
        """This tenant's spend for `month` in micro-USD, 0 if none, `None` if unknown."""
        ...

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        """Add `micro_usd` and answer the new total, or `None` if it could not be added."""
        ...

    def user_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        """One user's spend for `month`, with the same `None` rule as `spend`."""
        ...

    def user_debit(self, tenant: str, user_id: str, month: str, micro_usd: int) -> int | None:
        """Add to one user's counter and answer its new total, or `None`."""
        ...

    def seeded_spend(self, tenant: str, month: str, loader: SpendLoader) -> int | None:
        """`spend`, after loading the counter from `loader` once this month."""
        ...

    def seeded_user_spend(
        self, tenant: str, user_id: str, month: str, loader: SpendLoader
    ) -> int | None:
        """`user_spend`, after loading the user's counter from `loader` once this month."""
        ...


class InProcessBudgetStore:
    """A `BudgetStore` counting in one process's memory.

    What `create_app` keeps when `OPSCONSOLE_REDIS_URL` is unset, exactly as the
    response cache and the directory store do, so the suite and a local run need
    no backing service. Two gateway replicas do not share a counter this way, so
    each would enforce its own share of a cap: acceptable for a single G0 node and
    the reason `RedisBudgetStore` below is the one that ships to the cluster.

    It never answers `None`:
    a dictionary cannot be unreachable, so "unknown" has no meaning here, and a
    route that saw `None` from it would skip a gauge it could have published.
    """

    def __init__(self) -> None:
        self._totals: dict[str, int] = {}
        self._seeded: set[str] = set()

    def spend(self, tenant: str, month: str) -> int | None:
        """This tenant's spend so far, 0 for a month it has not spent anything in."""
        return self._totals.get(month_key(tenant, month), 0)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        """Add to this month's counter and answer the running total.

        A debit of zero is a debit: it books nothing and still answers the total,
        which is what a cache hit needs.
        """
        return self._add(month_key(tenant, month), micro_usd)

    def user_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        """One user's spend so far, 0 for a month it has not spent anything in."""
        return self._totals.get(user_month_key(tenant, user_id, month), 0)

    def user_debit(self, tenant: str, user_id: str, month: str, micro_usd: int) -> int | None:
        """Add to one user's month and answer its running total."""
        return self._add(user_month_key(tenant, user_id, month), micro_usd)

    def seeded_spend(self, tenant: str, month: str, loader: SpendLoader) -> int | None:
        """This tenant's spend, the counter set from the ledger the first time it is read."""
        return self._seed(budget_seeded_key(tenant, month), month_key(tenant, month), loader)

    def seeded_user_spend(
        self, tenant: str, user_id: str, month: str, loader: SpendLoader
    ) -> int | None:
        """One user's spend, loaded from the ledger the same way."""
        return self._seed(
            user_seeded_key(tenant, user_id, month),
            user_month_key(tenant, user_id, month),
            loader,
        )

    def _seed(self, marker: str, key: str, loader: SpendLoader) -> int:
        if marker not in self._seeded:
            loaded = loader()
            if loaded is not None:
                self._totals[key] = loaded
                self._seeded.add(marker)
        return self._totals.get(key, 0)

    def _add(self, key: str, micro_usd: int) -> int:
        total = self._totals.get(key, 0) + micro_usd
        self._totals[key] = total
        return total


class RedisBudgetStore:
    """A `BudgetStore` counting in Redis, where every replica adds to one total.

    The implementation the gateway ships with once `OPSCONSOLE_REDIS_URL` is set.
    It differs from the in-process twin in three ways that matter.

    The increment is `INCRBY` on an integer, never `INCRBYFLOAT`:
    the counter is micro-USD, and `INCRBY` is atomic, so two replicas booking at
    the same instant add up rather than overwrite each other.

    The expiry is re-sent on every debit, immediately after the increment. `INCRBY` on a missing key
    creates one with no TTL, so an `EXPIRE` that ran only on the first debit
    would leave every counter immortal whenever the first debit's `EXPIRE` was
    the command that failed.

    Every `redis.RedisError` is swallowed and becomes `None`, for the reason given
    in the module docstring. The catch is on the base class and not on
    `ConnectionError`: a timeout, a `READONLY` reply from a demoted replica and an
    authentication failure are all "the counter cannot answer", and catching one
    of them would leave the other three escaping into a request.
    A stored value that is not an integer is unknown for the same reason.

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

        A missing key really is zero spend and is reported as such; only a Redis that
        could not answer, or a value that is not an integer, is unknown.
        """
        return self._read(self.key(tenant, month), tenant, month)

    def debit(self, tenant: str, month: str, micro_usd: int) -> int | None:
        """Add `micro_usd` to this month's counter and answer the new total.

        `None` when Redis could not be reached, which the route reads as "serve,
        and move no gauge".
        """
        return self._add(self.key(tenant, month), tenant, month, micro_usd)

    def user_key(self, tenant: str, user_id: str, month: str) -> str:
        """The Redis key one user's month is counted under, every segment checked."""
        return user_month_key(tenant, user_id, month, prefix=self._key_prefix)

    def user_spend(self, tenant: str, user_id: str, month: str) -> int | None:
        """One user's spend so far, read the way `spend` reads the tenant's."""
        return self._read(self.user_key(tenant, user_id, month), tenant, month)

    def user_debit(self, tenant: str, user_id: str, month: str, micro_usd: int) -> int | None:
        """Add to one user's counter, with the same expiry and `None` rule as `debit`.

        The logs name the tenant and month only, never the user id.
        """
        return self._add(self.user_key(tenant, user_id, month), tenant, month, micro_usd)

    def seeded_spend(self, tenant: str, month: str, loader: SpendLoader) -> int | None:
        """This tenant's spend, the counter set from the ledger once a month by `SEED_SCRIPT`.

        A Redis that cannot answer is unknown before the ledger is asked, so an outage
        costs no ledger read. `LedgerUnreadable` from `loader` is let out on purpose.
        """
        return self._seed(
            budget_seeded_key(tenant, month, prefix=self._key_prefix),
            self.key(tenant, month),
            tenant,
            month,
            loader,
        )

    def seeded_user_spend(
        self, tenant: str, user_id: str, month: str, loader: SpendLoader
    ) -> int | None:
        """One user's spend, loaded the same way under the user's own marker."""
        return self._seed(
            user_seeded_key(tenant, user_id, month, prefix=self._key_prefix),
            self.user_key(tenant, user_id, month),
            tenant,
            month,
            loader,
        )

    def _seed(
        self, marker: str, stored: str, tenant: str, month: str, loader: SpendLoader
    ) -> int | None:
        try:
            loaded_already = bool(self._client.exists(marker))
        except redis.RedisError:
            _log.exception("budget marker unreadable for tenant %s, month %s", tenant, month)
            return None
        if loaded_already:
            return self._read(stored, tenant, month)
        ledger = loader()
        if ledger is None:
            return self._read(stored, tenant, month)
        try:
            return int(
                self._client.eval(SEED_SCRIPT, 2, marker, stored, ledger, BUDGET_KEY_TTL_SECONDS)
            )
        except redis.RedisError:
            _log.exception("budget load failed for tenant %s, month %s", tenant, month)
            return None

    def _read(self, stored: str, tenant: str, month: str) -> int | None:
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

    def _add(self, stored: str, tenant: str, month: str, micro_usd: int) -> int | None:
        try:
            total = int(self._client.incrby(stored, micro_usd))
            self._client.expire(stored, BUDGET_KEY_TTL_SECONDS)
        except redis.RedisError:
            _log.exception(
                "budget debit of %s lost for tenant %s, month %s", micro_usd, tenant, month
            )
            return None
        return total
