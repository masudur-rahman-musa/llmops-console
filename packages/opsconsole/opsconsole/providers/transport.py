"""The HTTP half every provider adapter shares: one client, fixed timeouts, one retry table.

A provider adapter is two things that change at different rates. The MAPPING is
per-vendor and is the thing this project owns; the TRANSPORT - how a request is
sent, how long it may take, what is worth a second attempt, and how much of a
provider's answer is kept for the log - is the same for every HTTP provider and
is decided once, in the design. This module is that second half, so a new adapter
inherits the policy instead of restating it.

Everything that leaves here is a `ProviderError` carrying `retryable`, so a
caller reads one flag rather than re-deriving a verdict from a status it can no
longer see. `sleep` and `jitter` are injected, so the retry table costs
microseconds to test and produces exact delays rather than a range.

This module knows no vendor. The path, the headers and the base URL are the
adapter's to pass; nothing here reads a provider's body beyond truncating it for
the log, and pulling `error.message` out of a 400 so the reason a request was
refused is in the gateway log rather than behind a bare 502.
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

from opsconsole.core.errors import ProviderError

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 60.0
"""The fixed timeouts (the design). Not caller-settable: a caller able to raise the
read timeout could hold a worker for as long as it liked. Pinned on the constants
by `test_timeout_constants_are_exactly_the_spec_values` and on the built client by
`test_client_timeouts_are_fixed_at_five_and_sixty`."""

STREAM_READ_TIMEOUT = 120.0
"""The read timeout on a streamed call, longer because the gap being measured is
the one between EVENTS rather than the whole answer (the design).

Set per request rather than on the client, so an ordinary completion still gives
up after 60s: raising the client's timeout to 120 would let every non-streaming
call hold a worker twice as long. Both halves are pinned by
`test_stream_read_timeout_is_exactly_120`, which asserts the streamed request's
timeout and the client's own in the same test."""

MAX_RETRIES = 2
RETRY_BASE_DELAY = 0.5
"""Two retries, so three attempts at most, with the delay doubling and jittered.

Retries are restricted to 429, any 5xx and a timeout: those are the failures a
second attempt can fix. Every other status is raised on the first attempt,
because retrying a 400 bills the tenant three times for a request that cannot
succeed. `test_retry_table_by_status` holds the whole table, request counts
included, and `test_backoff_doubles_and_is_jittered` pins the two delays."""

PROVIDER_MESSAGE_LIMIT = 500
"""How much of a provider's error body is kept for the log, so a proxy answering
with a megabyte of HTML cannot write a megabyte per failure. Pinned by
`test_provider_message_limit_is_exactly_500` and
`test_an_oversized_provider_body_is_truncated`; it never reaches the client
(`test_provider_message_never_reaches_the_client_message`)."""


def unreadable(exc: Exception) -> ProviderError:
    """The one answer to a body we cannot parse: a 502, not a traceback.

    Not retryable: a well-formed failure would have carried a status, so a body
    we cannot read is a shape change or a bug, and a second identical request
    produces the same unreadable answer
    (`test_a_non_json_success_body_is_a_provider_error`).
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

    One thing is logged here: a 400's parsed `error.message` (see
    `status_error`). Otherwise what reaches a log on a failure is
    `ProviderError.provider_message`, which holds the provider's answer or the
    transport's own exception text, never the request body or its headers
    (`test_provider_message_never_reaches_the_client_message`).
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
        provider something the caller did not (`test_retry_resends_the_same_body`).
        """
        attempt = 0
        while True:
            try:
                response = self.client.post(path, json=dict(body))
            except httpx.RequestError as exc:
                error = self.transport_error(exc)
            else:
                if response.status_code == httpx.codes.OK:
                    return response
                error = self.status_error(response)

            if error.retryable and attempt < MAX_RETRIES:
                self._wait(attempt)
                attempt += 1
                continue
            raise error

    @contextlib.contextmanager
    def open_stream(self, path: str, body: Mapping[str, Any]) -> Iterator[httpx.Response]:
        """Open the streamed POST, retrying per the table, and yield the live response.

        The read timeout is `STREAM_READ_TIMEOUT` on this request only, so the
        client's 60s still applies to every ordinary completion. A failure status
        is read to the end before it is turned into an error, because the
        provider's body is what `provider_message` carries and an unread
        streaming response has none (`test_a_failure_status_on_the_open_never_starts_a_stream`).

        The retry lives here rather than around the iteration on purpose: once one
        event has been yielded the client has already been shown part of an
        answer, and nothing that cannot be un-sent is resent
        (`test_stream_open_retries_only_before_the_first_event`).
        """
        attempt = 0
        while True:
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
                    error = self.status_error(response)
            except httpx.RequestError as exc:
                error = self.transport_error(exc)

            if error.retryable and attempt < MAX_RETRIES:
                self._wait(attempt)
                attempt += 1
                continue
            raise error

    def status_error(self, response: httpx.Response) -> ProviderError:
        """Turn a failure status into the typed error, keeping the provider's words out.

        `message` is client-facing and carries the provider name and the status
        only; the body goes to `provider_message` for the log
        (`test_provider_message_never_reaches_the_client_message`).

        A 400 also logs one WARNING with the provider's parsed `error.message`,
        cut to `PROVIDER_MESSAGE_LIMIT`: a 400 is a request the provider refused
        as malformed, and without the reason it reads as a bare 502
        (`test_a_400_logs_the_providers_message`, `test_the_400_log_is_cut_at_500`).
        Only the parsed message, never the raw body, which is free-form and may
        echo request text; a body with no readable message logs its length only
        (`test_a_400_with_no_readable_message_logs_status_and_length_only`). Other
        statuses log nothing, since a 401 body can quote part of the key
        (`test_a_500_logs_nothing_new`).
        """
        status = response.status_code
        if status == httpx.codes.BAD_REQUEST:
            self._log_bad_request(response)
        return ProviderError(
            f"{self.name}: the provider answered {status}",
            retryable=status == httpx.codes.TOO_MANY_REQUESTS or status >= 500,
            provider_message=response.text[:PROVIDER_MESSAGE_LIMIT],
        )

    def _log_bad_request(self, response: httpx.Response) -> None:
        """Log a 400's `error.message`, or only the body's length when there is none."""
        message: object = None
        try:
            payload = json.loads(response.content)
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                message = error.get("message")
        if isinstance(message, str):
            logger.warning(
                "%s: the provider answered 400: %s",
                self.name,
                message[:PROVIDER_MESSAGE_LIMIT],
            )
        else:
            logger.warning(
                "%s: the provider answered 400 with no readable error message (%d bytes)",
                self.name,
                len(response.content),
            )

    def transport_error(self, exc: httpx.RequestError) -> ProviderError:
        """Turn a request that never completed into the typed error.

        Only a timeout is retryable. Any other transport failure is reported as
        it is rather than tried twice, which keeps the retry table to the three
        cases the plan allows (`test_retry_table_by_status`).
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
        retries. The two delays are pinned exactly by
        `test_backoff_doubles_and_is_jittered`.
        """
        self._sleep(RETRY_BASE_DELAY * (2**attempt) * (0.5 + 0.5 * self._jitter()))
