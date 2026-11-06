"""End-to-end: quota exhaustion mid-research, then resume once quota resets.

Runs the real CLI stack (ADK Runner, SQLite sessions, checkpoints, pruner,
RetryingLiteLlm). Only the network call inside LiteLlm is replaced by a script
keyed on which agent is asking.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import litellm
import pytest
from google.adk.models.lite_llm import LiteLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from rpi_agent import cli
from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.repl import loop as loop_module
from tests.fakes import FakePromptSession

DAILY_QUOTA = (
    "OpenAIException - Daily credit limit exceeded 5000000/5000000 credits used. "
    "Quota resets in 24h 42m. Need a higher limit?"
)
LOCATOR = "rpi-codebase-locator"
# Larger than MAX_TOOL_RESULT_PRESERVE_CHARS, so pruning would stub it if it ran.
LOCATOR_RESULT = "LOCATOR FINDINGS: src/auth.py\n" + "- src/auth/token.py:42\n" * 40
ANALYZER = "rpi-codebase-analyzer"


def call(skill: str, task: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(
            role="model",
            parts=[
                types.Part(
                    function_call=types.FunctionCall(
                        name="run_subagent",
                        args={"skill_name": skill, "task_prompt": task},
                    )
                )
            ],
        )
    )


def text(value: str) -> LlmResponse:
    return LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text=value)])
    )


class GatewayScript:
    """Per-agent scripted responses standing in for the model gateway."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.steps: dict[str, list[Any]] = {"root": [], LOCATOR: [], ANALYZER: []}
        self.requests: dict[str, list[list[types.Content]]] = {
            k: [] for k in self.steps
        }
        script = self

        async def fake_generate(
            self: LiteLlm, llm_request: LlmRequest, stream: bool = False
        ) -> AsyncGenerator[LlmResponse]:
            role = script.role_for(llm_request)
            script.requests[role].append(list(llm_request.contents))
            step = script.steps[role].pop(0)
            if isinstance(step, Exception):
                raise step
            yield step

        monkeypatch.setattr(LiteLlm, "generate_content_async", fake_generate)

    @staticmethod
    def role_for(llm_request: LlmRequest) -> str:
        instruction = str(llm_request.config.system_instruction or "")
        for skill in (LOCATOR, ANALYZER):
            if f"executing skill '{skill}'" in instruction:
                return skill
        return "root"


def run_cli(
    argv: list[str],
    inputs: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> str:
    monkeypatch.setattr(
        loop_module,
        "PromptSession",
        lambda **kw: FakePromptSession(inputs, EOFError, **kw),
    )
    asyncio.run(cli.main(argv))
    return " ".join(capsys.readouterr().out.split())


def function_responses(contents: list[types.Content]) -> list[tuple[str, str]]:
    found = []
    for content in contents:
        for part in content.parts or []:
            response = part.function_response
            if response is not None and isinstance(response.response, dict):
                found.append((response.name or "", str(response.response["result"])))
    return found


def function_calls(contents: list[types.Content]) -> list[str]:
    return [
        str((part.function_call.args or {}).get("skill_name"))
        for content in contents
        for part in content.parts or []
        if part.function_call is not None
    ]


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, git_repo: Path, tmp_path: Path) -> None:
    for var in ("STARK_API_KEY", "STARK_BASE_URL", "STARK_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "key")
    monkeypatch.chdir(git_repo)


def test_quota_mid_research_then_resume(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    git_repo: Path,
) -> None:
    gateway = GatewayScript(monkeypatch)
    skills = ["--skills-dir", str(git_repo / "no-skills")]

    # --- Day 1: locator completes, analyzer exhausts the daily quota. ---------
    gateway.steps["root"] += [
        call(LOCATOR, "find auth code"),
        call(ANALYZER, "analyze auth code"),
    ]
    gateway.steps[LOCATOR] += [text(LOCATOR_RESULT)]
    gateway.steps[ANALYZER] += [
        litellm.RateLimitError(DAILY_QUOTA, llm_provider="openai", model="m")
    ]

    day1 = run_cli(skills, ["/research auth", "never sent"], monkeypatch, capsys)

    assert "Model Quota Exhausted" in day1
    assert "Quota resets in 24h 42m." in day1
    assert "1 completed sub-agent result(s)" in day1
    assert "rpi-agent --resume sess_" in day1
    session_id = day1.split("rpi-agent --resume ")[1].split()[0]

    store = CheckpointStore(git_repo / ".rpi_agent" / "subagents")
    (saved,) = store.for_session(session_id)
    assert (saved.skill_name, saved.result) == (LOCATOR, LOCATOR_RESULT.strip())
    assert gateway.steps[ANALYZER] == []  # quota is not retried

    # --- Day 2: resume. Locator is reused; analyzer runs; report finishes. ----
    gateway.steps["root"] += [
        call(LOCATOR, "find auth code"),
        call(ANALYZER, "analyze auth code"),
        text("FINAL REPORT"),
    ]
    gateway.steps[ANALYZER] += [text("ANALYSIS: token refresh bug")]

    day2 = run_cli(
        [*skills, "--resume"], ["continue the research"], monkeypatch, capsys
    )

    assert (
        f"Resumed session {session_id} (1 prior prompts, 1 saved sub-agent results)"
        in day2
    )
    assert "FINAL REPORT" in day2

    # The first resumed request still carries the locator's completed result
    # (the interrupted turn is not pruned) and no dangling analyzer call.
    first_resumed = gateway.requests["root"][2]
    assert ("run_subagent", LOCATOR_RESULT.strip()) in function_responses(first_resumed)
    assert function_calls(first_resumed) == [LOCATOR]

    # The locator model was never called again; its checkpoint was reused.
    assert len(gateway.requests[LOCATOR]) == 1
    reused = function_responses(gateway.requests["root"][3])
    assert any(
        r.startswith("[Reused saved sub-agent result") and "LOCATOR FINDINGS" in r
        for _, r in reused
    )

    assert [c.skill_name for c in store.for_session(session_id)] == [LOCATOR, ANALYZER]
    assert all(not steps for steps in gateway.steps.values())


def test_transient_rate_limit_is_retried_within_turn(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    git_repo: Path,
) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr("rpi_agent.llm._sleep", fake_sleep)
    gateway = GatewayScript(monkeypatch)
    tpm = litellm.RateLimitError(
        "500000 tokens per min", llm_provider="openai", model="m"
    )
    gateway.steps["root"] += [tpm, tpm, text("DONE")]

    out = run_cli(
        ["--skills-dir", str(git_repo / "none")], ["hello"], monkeypatch, capsys
    )
    assert "RateLimitError from model gateway; retrying in" in out
    assert "(attempt 2/5)" in out
    assert "DONE" in out
    assert len(sleeps) == 2
