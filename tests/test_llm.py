from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import httpx
import litellm
import pytest
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from rpi_agent import llm
from rpi_agent.constants import (
    MAX_RETRY_AFTER_SECONDS,
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
)
from rpi_agent.exceptions import QuotaExhaustedError
from rpi_agent.llm import (
    RetryingLiteLlm,
    backoff_delay,
    is_quota_exhausted,
    quota_reset_hint,
    retry_after_seconds,
)

# A typical message returned by a gateway once the daily credit quota is used up.
DAILY_QUOTA_MESSAGE = (
    "litellm.RateLimitError: RateLimitError: OpenAIException - Daily credit limit "
    "exceeded 5000000/5000000 credits used. Quota resets in 24h 42m. Need a higher "
    "limit? Request an increase from your gateway administrator."
)


def rate_limit(
    message: str = "TPM limit", headers: dict[str, str] | None = None
) -> Any:
    err = litellm.RateLimitError(message, llm_provider="openai", model="m")
    if headers is not None:
        # Set by LiteLLM at runtime; not declared on the exception class.
        setattr(err, "litellm_response_headers", httpx.Headers(headers))  # noqa: B010
    return err


def response(text: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=text)])
    )


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    monkeypatch.setattr(llm, "_sleep", fake_sleep)
    return recorded


def scripted(monkeypatch: pytest.MonkeyPatch, steps: list[Any]) -> list[int]:
    """Makes LiteLlm.generate_content_async replay `steps`; returns call counter."""
    calls = [0]

    async def fake_generate(
        self: LiteLlm, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse]:
        calls[0] += 1
        step = steps.pop(0)
        for item in step if isinstance(step, list) else [step]:
            if isinstance(item, Exception):
                raise item
            yield item

    monkeypatch.setattr(LiteLlm, "generate_content_async", fake_generate)
    return calls


def collect(model: RetryingLiteLlm) -> list[LlmResponse]:
    async def run() -> list[LlmResponse]:
        return [r async for r in model.generate_content_async(LlmRequest())]

    return asyncio.run(run())


# ------------------------------------------------------------------------------
# Error classification
# ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        DAILY_QUOTA_MESSAGE,
        "Daily token limit exceeded",
        "daily limit exceeded",
        "Error code: 429 - {'error': {'code': 'insufficient_quota'}}",
        "Monthly quota exceeded for this key",
        "quota exhausted",
    ],
)
def test_quota_messages_detected(message: str) -> None:
    assert is_quota_exhausted(Exception(message))


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit reached for gpt in organization: Limit 500000 tokens per min",
        "Too many requests",
        "Internal server error",
    ],
)
def test_rate_limit_messages_not_quota(message: str) -> None:
    assert not is_quota_exhausted(Exception(message))


def test_quota_reset_hint() -> None:
    assert quota_reset_hint(Exception(DAILY_QUOTA_MESSAGE)) == "24h 42m"
    assert quota_reset_hint(Exception("insufficient_quota")) is None


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "12"}, 12.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"retry-after-ms": "2500", "retry-after": "99"}, 2.5),
        ({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None),
        ({"retry-after": "-3"}, None),
        ({}, None),
    ],
)
def test_retry_after_from_litellm_headers(
    headers: dict[str, str], expected: float | None
) -> None:
    assert retry_after_seconds(rate_limit(headers=headers)) == expected


def test_retry_after_falls_back_to_response_headers() -> None:
    err = rate_limit()
    err.response = httpx.Response(429, headers={"retry-after": "7"})
    assert retry_after_seconds(err) == 7.0


def test_retry_after_without_headers() -> None:
    assert retry_after_seconds(Exception("plain")) is None


def test_backoff_honors_retry_after() -> None:
    assert backoff_delay(3, retry_after=42.0) == 42.0


@pytest.mark.parametrize("attempt", range(8))
def test_backoff_is_exponential_jittered_and_capped(attempt: int) -> None:
    ceiling = min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * 2**attempt)
    for _ in range(20):
        assert ceiling * 0.5 <= backoff_delay(attempt) <= ceiling


# ------------------------------------------------------------------------------
# RetryingLiteLlm
# ------------------------------------------------------------------------------


def test_client_disables_sdk_retries_and_keeps_policy_private() -> None:
    model = RetryingLiteLlm(model="openai/m", max_retries=3, api_key="k")
    assert model.max_retries == 3
    assert model._additional_args["num_retries"] == 0
    assert "max_retries" not in model._additional_args
    assert "on_retry" not in model._additional_args
    assert model._additional_args["api_key"] == "k"


def test_success_passes_through(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    resp_with_usage = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="b")]),
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10,
            candidates_token_count=5,
            total_token_count=15,
        ),
    )
    calls = scripted(monkeypatch, [[response("a"), resp_with_usage]])
    out = collect(RetryingLiteLlm(model="openai/m"))
    assert [r.content.parts[0].text for r in out if r.content and r.content.parts] == [
        "a",
        "b",
    ]
    assert calls[0] == 1
    assert sleeps == []


@pytest.mark.parametrize(
    "error",
    [
        rate_limit(),
        litellm.APIConnectionError("conn reset", llm_provider="openai", model="m"),
        litellm.InternalServerError("500", llm_provider="openai", model="m"),
        litellm.ServiceUnavailableError("503", llm_provider="openai", model="m"),
        litellm.Timeout("timed out", model="m", llm_provider="openai"),
    ],
)
def test_transient_errors_are_retried(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float], error: Exception
) -> None:
    notices: list[tuple[int, int, float, str]] = []
    calls = scripted(monkeypatch, [error, error, response("ok")])
    model = RetryingLiteLlm(
        model="openai/m",
        max_retries=5,
        on_retry=lambda a, m, d, e: notices.append((a, m, d, type(e).__name__)),
    )
    out = collect(model)
    assert len(out) == 1
    assert calls[0] == 3
    assert len(sleeps) == 2
    assert [(a, m) for a, m, _, _ in notices] == [(1, 5), (2, 5)]
    assert [d for _, _, d, _ in notices] == sleeps


def test_retry_after_header_is_used_as_delay(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    scripted(monkeypatch, [rate_limit(headers={"retry-after": "30"}), response("ok")])
    collect(RetryingLiteLlm(model="openai/m"))
    assert sleeps == [30.0]


def test_gives_up_after_max_retries(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    calls = scripted(monkeypatch, [rate_limit()] * 4)
    with pytest.raises(litellm.RateLimitError):
        collect(RetryingLiteLlm(model="openai/m", max_retries=3))
    assert calls[0] == 4
    assert len(sleeps) == 3


def test_zero_retries(monkeypatch: pytest.MonkeyPatch, sleeps: list[float]) -> None:
    scripted(monkeypatch, [rate_limit()])
    with pytest.raises(litellm.RateLimitError):
        collect(RetryingLiteLlm(model="openai/m", max_retries=0))
    assert sleeps == []


def test_excessive_retry_after_is_not_waited(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    long_wait = str(int(MAX_RETRY_AFTER_SECONDS) + 1)
    scripted(monkeypatch, [rate_limit(headers={"retry-after": long_wait})])
    with pytest.raises(litellm.RateLimitError):
        collect(RetryingLiteLlm(model="openai/m"))
    assert sleeps == []


def test_daily_quota_fails_fast_with_reset_hint(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    calls = scripted(monkeypatch, [rate_limit(DAILY_QUOTA_MESSAGE)])
    with pytest.raises(QuotaExhaustedError) as exc:
        collect(RetryingLiteLlm(model="openai/m"))
    assert exc.value.reset_hint == "24h 42m"
    assert isinstance(exc.value.__cause__, litellm.RateLimitError)
    assert calls[0] == 1
    assert sleeps == []


def test_quota_reported_as_non_retryable_status_is_detected(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    err = litellm.PermissionDeniedError(
        "insufficient_quota",
        llm_provider="openai",
        model="m",
        response=httpx.Response(403, request=httpx.Request("POST", "http://x")),
    )
    scripted(monkeypatch, [err])
    with pytest.raises(QuotaExhaustedError):
        collect(RetryingLiteLlm(model="openai/m"))


def test_non_retryable_api_error_raises_immediately(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    err = litellm.BadRequestError("bad request", model="m", llm_provider="openai")
    calls = scripted(monkeypatch, [err])
    with pytest.raises(litellm.BadRequestError):
        collect(RetryingLiteLlm(model="openai/m"))
    assert calls[0] == 1
    assert sleeps == []


def test_no_retry_after_partial_output(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    calls = scripted(monkeypatch, [[response("partial"), rate_limit()]])
    with pytest.raises(litellm.RateLimitError):
        collect(RetryingLiteLlm(model="openai/m"))
    assert calls[0] == 1
    assert sleeps == []


def test_default_sleep_is_asyncio_sleep() -> None:
    assert llm._sleep is asyncio.sleep
