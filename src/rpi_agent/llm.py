"""Model client with quota detection and rate-limit aware retries."""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import AsyncGenerator, Callable
from typing import Any

import litellm
import openai
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from pydantic import PrivateAttr

from rpi_agent.constants import (
    DEFAULT_MAX_RETRIES,
    MAX_RETRY_AFTER_SECONDS,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
)
from rpi_agent.exceptions import QuotaExhaustedError

# Transient failures worth retrying with backoff.
RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    litellm.RateLimitError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.Timeout,
)

# Quota exhaustion is reported as a 429 like a TPM limit, but retrying within a
# turn cannot succeed. e.g. "Daily credit limit exceeded 5000000/5000000 credits
# used. Quota resets in 24h 42m."
_QUOTA_PATTERN = re.compile(
    r"daily\s+(?:\w+\s+)?limit\s+exceeded|insufficient_quota|quota\s+(?:exceeded|exhausted)",
    re.IGNORECASE,
)
_RESET_PATTERN = re.compile(r"quota\s+resets\s+in\s+([^.\n]+)", re.IGNORECASE)

RetryNotifier = Callable[[int, int, float, Exception], None]

# Indirection so tests can skip real sleeps.
_sleep = asyncio.sleep


def is_quota_exhausted(error: BaseException) -> bool:
    """Returns True if the error reports an exhausted (non-rate) quota."""
    return bool(_QUOTA_PATTERN.search(str(error)))


def quota_reset_hint(error: BaseException) -> str | None:
    """Extracts the gateway's "Quota resets in ..." hint, if present."""
    match = _RESET_PATTERN.search(str(error))
    return match.group(1).strip() if match else None


def retry_after_seconds(error: BaseException) -> float | None:
    """Reads Retry-After (or retry-after-ms) from the error's response headers."""
    headers: Any = getattr(error, "litellm_response_headers", None)
    if headers is None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)
    if not headers:
        return None

    for header, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(header)
        if value is None:
            continue
        try:
            seconds = float(value) / divisor
        except TypeError, ValueError:
            continue
        if seconds >= 0:
            return seconds
    return None


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """Computes the wait before retry number `attempt` (0-based).

    Honors a server Retry-After when given; otherwise exponential backoff with
    jitter, capped so a per-minute window can elapse within a few attempts.
    """
    if retry_after is not None:
        return retry_after
    ceiling = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * 2**attempt)
    return ceiling * random.uniform(0.5, 1.0)


class RetryingLiteLlm(LiteLlm):
    """LiteLlm that retries transient gateway errors and surfaces quota exhaustion.

    SDK-level retries are disabled (num_retries=0) so this class owns the policy:
    it can wait out per-minute limits instead of giving up after a few seconds,
    and fails fast with QuotaExhaustedError when retrying cannot help.
    """

    _max_retries: int = PrivateAttr(default=DEFAULT_MAX_RETRIES)
    _on_retry: RetryNotifier | None = PrivateAttr(default=None)

    def __init__(
        self,
        model: str,
        *,
        max_retries: int = DEFAULT_MAX_RETRIES,
        on_retry: RetryNotifier | None = None,
        **kwargs: Any,
    ) -> None:
        """Initializes the client.

        Args:
            model: LiteLLM model identifier.
            max_retries: Retries after the first attempt for transient errors.
            on_retry: Called as (attempt, max_retries, delay, error) before waiting.
            **kwargs: Passed through to LiteLlm / litellm completion.
        """
        kwargs["num_retries"] = 0
        super().__init__(model=model, **kwargs)
        self._max_retries = max_retries
        self._on_retry = on_retry

    @property
    def max_retries(self) -> int:
        """Retries attempted after the first call for transient errors."""
        return self._max_retries

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        """Generates content, retrying transient failures before any output."""
        attempt = 0
        while True:
            yielded = False
            try:
                async for response in super().generate_content_async(
                    llm_request, stream
                ):
                    yielded = True
                    yield response
                return
            except openai.OpenAIError as err:
                # Checked for every API error: some gateways report quota as 403.
                if is_quota_exhausted(err):
                    raise QuotaExhaustedError(
                        str(err), reset_hint=quota_reset_hint(err)
                    ) from err

                retry_after = retry_after_seconds(err)
                # Never replay a partially streamed response; give up when out of
                # attempts or when the server asks us to wait longer than a turn.
                if (
                    not isinstance(err, RETRYABLE_ERRORS)
                    or yielded
                    or attempt >= self._max_retries
                    or (
                        retry_after is not None
                        and retry_after > MAX_RETRY_AFTER_SECONDS
                    )
                ):
                    raise

                delay = backoff_delay(attempt, retry_after)
                attempt += 1
                if self._on_retry is not None:
                    self._on_retry(attempt, self._max_retries, delay, err)
                await _sleep(delay)
