"""The HTTP half every provider adapter shares: one client, fixed timeouts, one retry table.

A provider adapter is two things that change at different rates. The MAPPING is
per-vendor and is the thing this project owns; the TRANSPORT - how a request is
sent, how long it may take, what is worth a second attempt, and how much of a
provider's answer is kept for the log - is the same for every HTTP provider and
is decided once. This module is that second half, so a new adapter
inherits the policy instead of restating it.

Everything that leaves here is a `ProviderError` carrying `retryable`, so a
caller reads one flag rather than re-deriving a verdict from a status it can no
longer see. `sleep` and `jitter` are injected, so the retry table costs
microseconds to test and produces exact delays rather than a range.

This module knows no vendor. The path, the headers and the base URL are the
adapter's to pass; nothing here reads a provider's body beyond truncating it for
the log, pulling `error.type` and `error.message` out of a failure so the reason a
call failed is in the gateway log rather than behind a bare 502, and reading
`error.code` to spot a content policy refusal, which is never retried.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any

import httpx

from app.core.errors import ProviderError

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 60.0
"""The fixed timeouts. Not caller-settable: a caller able to raise the
read timeout could hold a worker for as long as it liked."""

STREAM_READ_TIMEOUT = 120.0
"""The read timeout on a streamed call, longer because the gap being measured is
the one between EVENTS rather than the whole answer.

Set per request rather than on the client, so an ordinary completion still gives
up after 60s: raising the client's timeout to 120 would let every non-streaming
call hold a worker twice as long."""

MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5
"""Two retries, so three attempts at most, with the delay doubling and jittered.

Retries are restricted to 429, any 5xx and a timeout: those are the failures a
second attempt can fix. Every other status is raised on the first attempt,
because retrying a 400 bills the tenant three times for a request that cannot
succeed."""

REFUSAL_CODES = frozenset({"content_policy_violation", "content_filter"})
REFUSAL_INNER_CODES = frozenset({"ResponsibleAIPolicyViolation"})
"""The `error.code` values (OpenAI, Azure) and the Azure `error.innererror.code`
value that mean the provider refused the request for content policy. Matched at
any status, because a refusal is not something another attempt can fix."""

PROVIDER_MESSAGE_LIMIT = 500
"""How much of a provider's error body is kept for the log, so a proxy answering
with a megabyte of HTML cannot write a megabyte per failure. It never reaches the client."""


def unreadable(exc: Exception) -> ProviderError:
    """The one answer to a body we cannot parse: a 502, not a traceback.

    Not retryable: a well-formed failure would have carried a status, so a body
    we cannot read is a shape change or a bug, and a second identical request
    produces the same unreadable answer.
    """
    return ProviderError(
        "the provider answered with a body the adapter could not read",
        retryable=False,
        provider_message=f"{type(exc).__name__}: {exc}",
    )


class ProviderTransport:
    """One provider endpoint's HTTP client, retry loop and error boundary.

    `name` is the POLICY provider name, not the vendor: it is what a failure
    message says, so two accounts on one vendor stay distinguishable in a log
    exactly as they are in the ledger and the resolver.

    One thing is logged here: one WARNING per failed call, once the retries are
    spent (see `_log_failure`). It holds the status, the provider's parsed
    `error.type` and `error.message`, and its request id, or the transport's own
    exception text, never the request body or its headers.
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        headers: Mapping[str, str],
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.name = name
        self._sleep = sleep
        self._jitter = jitter
        self.client = httpx.Client(
            base_url=base_url,
            headers=dict(headers),
            timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT),
            transport=transport,
        )

    def close(self) -> None:
        """Release the connection pool."""
        self.client.close()

    def send(self, path: str, body: Mapping[str, Any]) -> httpx.Response:
        """POST the body, retrying per the table, and raise `ProviderError` otherwise.

        The body is built once and resent unchanged, so a retry cannot ask the
        provider something the caller did not.
        """
        attempt = 0
        while True:
            failed: httpx.Response | httpx.RequestError
            try:
                response = self.client.post(path, json=dict(body))
            except httpx.RequestError as exc:
                failed, error = exc, self.transport_error(exc)
            else:
                if response.status_code == httpx.codes.OK:
                    return response
                failed, error = response, self.status_error(response)

            if error.retryable and attempt < MAX_RETRIES:
                self._wait(attempt)
                attempt += 1
                continue
            self._log_failure(failed, attempts=attempt + 1)
            raise error

    @contextlib.contextmanager
    def open_stream(self, path: str, body: Mapping[str, Any]) -> Iterator[httpx.Response]:
        """Open the streamed POST, retrying per the table, and yield the live response.

        The read timeout is `STREAM_READ_TIMEOUT` on this request only, so the
        client's 60s still applies to every ordinary completion. A failure status
        is read to the end before it is turned into an error, because the
        provider's body is what `provider_message` carries and an unread
        streaming response has none.

        The retry lives here rather than around the iteration on purpose: once one
        event has been yielded the client has already been shown part of an
        answer, and nothing that cannot be un-sent is resent.
        """
        attempt = 0
        while True:
            failed: httpx.Response | httpx.RequestError
            try:
                with self.client.stream(
                    "POST",
                    path,
                    json=dict(body),
                    timeout=httpx.Timeout(STREAM_READ_TIMEOUT, connect=CONNECT_TIMEOUT),
                ) as response:
                    if response.status_code == httpx.codes.OK:
                        yield response
                        return
                    response.read()
                    failed, error = response, self.status_error(response)
            except httpx.RequestError as exc:
                failed, error = exc, self.transport_error(exc)

            if error.retryable and attempt < MAX_RETRIES:
                self._wait(attempt)
                attempt += 1
                continue
            self._log_failure(failed, attempts=attempt + 1)
            raise error

    def status_error(self, response: httpx.Response) -> ProviderError:
        """Turn a failure status into the typed error, keeping the provider's words out.

        `message` is client-facing and carries the provider name and the status
        only; the body goes to `provider_message`. Nothing is logged here, since a
        retried status that then succeeds is not a failure; `_log_failure` logs the
        last one.

        A body whose `error.code` names a content policy refusal is `refused` at
        any status, and so never retried.
        """
        status = response.status_code
        payload = _parsed(response)
        if _is_refusal(payload):
            return ProviderError(
                f"{self.name}: the provider refused the request for content policy",
                refused=True,
                provider_message=response.text[:PROVIDER_MESSAGE_LIMIT],
            )
        return ProviderError(
            f"{self.name}: the provider answered {status}",
            retryable=status == httpx.codes.TOO_MANY_REQUESTS or status >= 500,
            provider_message=response.text[:PROVIDER_MESSAGE_LIMIT],
        )

    def _log_failure(self, failed: httpx.Response | httpx.RequestError, *, attempts: int) -> None:
        """Log one WARNING for a call that failed for good, so a 502 has its reason.

        For a failure status: the status, the provider's parsed `error.type` and
        `error.message` cut to `PROVIDER_MESSAGE_LIMIT`, and the provider's request id.
        Only the parsed message, never the raw body, which is free-form and may echo
        request text; a body with no readable message logs its length only. A 401 or
        403 logs its type but not its message, since that message can quote part of
        the key. For a call that never completed: the exception's type and text.
        """
        tried = f"after {attempts} attempts" if attempts > 1 else "on the first attempt"
        if isinstance(failed, httpx.RequestError):
            logger.warning(
                "%s: the request did not complete %s: %s: %s",
                self.name,
                tried,
                type(failed).__name__,
                str(failed)[:PROVIDER_MESSAGE_LIMIT],
            )
            return
        status = failed.status_code
        error = _error_of(_parsed(failed))
        kind = error.get("type") or error.get("code")
        message = error.get("message")
        request_id = failed.headers.get("request-id") or failed.headers.get("x-request-id")
        where = f" (request id {request_id})" if request_id else ""
        label = f" {kind}" if isinstance(kind, str) else ""
        if status in (httpx.codes.UNAUTHORIZED, httpx.codes.FORBIDDEN):
            logger.warning(
                "%s: the provider answered %d%s %s; its message is not logged%s",
                self.name,
                status,
                label,
                tried,
                where,
            )
        elif isinstance(message, str):
            logger.warning(
                "%s: the provider answered %d%s %s: %s%s",
                self.name,
                status,
                label,
                tried,
                message[:PROVIDER_MESSAGE_LIMIT],
                where,
            )
        else:
            logger.warning(
                "%s: the provider answered %d %s with no readable error message (%d bytes)%s",
                self.name,
                status,
                tried,
                len(failed.content),
                where,
            )

    def transport_error(self, exc: httpx.RequestError) -> ProviderError:
        """Turn a request that never completed into the typed error.

        Only a timeout is retryable. Any other transport failure is reported as
        it is rather than tried twice, which keeps the retry table to the three
        cases that are worth a second attempt.
        """
        return ProviderError(
            f"{self.name}: the request did not complete ({type(exc).__name__})",
            retryable=isinstance(exc, httpx.TimeoutException),
            provider_message=str(exc),
        )

    def _wait(self, attempt: int) -> None:
        """Sleep the jittered doubling backoff before attempt `attempt + 1`.

        A 429's `retry-after` is deliberately not read: honouring it would let a
        provider hold a gateway worker for as long as it asked, past the read
        timeout this transport fixes. Two short jittered waits, then the caller
        is told; a sustained 429 is what trips `providers/breaker.py`, which
        counts what this transport gives up on and changes nothing about what it
        retries.
        """
        self._sleep(RETRY_BASE_DELAY * (2**attempt) * (0.5 + 0.5 * self._jitter()))


def _parsed(response: httpx.Response) -> object:
    """A failure body as JSON, or `None` when it is not JSON. Read once per failure."""
    try:
        return json.loads(response.content)
    except ValueError:
        return None


def _error_of(payload: object) -> dict[str, Any]:
    """The body's `error` object, or an empty one when there is none."""
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            return error
    return {}


def _is_refusal(payload: object) -> bool:
    """Does a failure body say the provider refused the request for content policy."""
    error = _error_of(payload)
    inner = error.get("innererror")
    code = error.get("code")
    inner_code = inner.get("code") if isinstance(inner, dict) else None
    return (isinstance(code, str) and code in REFUSAL_CODES) or (
        isinstance(inner_code, str) and inner_code in REFUSAL_INNER_CODES
    )
