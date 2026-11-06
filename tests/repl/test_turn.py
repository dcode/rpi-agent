from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import cast

import litellm
import pytest
from google.adk.runners import Runner
from rich.console import Console

from rpi_agent.constants import USER_ID
from rpi_agent.exceptions import CodebaseToolError, QuotaExhaustedError
from rpi_agent.pruning import ContextPruner
from rpi_agent.repl.turn import execute_agent_turn
from tests.fakes import FakeEvent, FakeRunner


def run_turn(
    runner: FakeRunner, prompt: str = "hi", pruner: ContextPruner | None = None
) -> str:
    console = Console(record=True, width=200, color_system=None)
    asyncio.run(
        execute_agent_turn(cast(Runner, runner), prompt, "sess_1", console, pruner)
    )
    return console.export_text()


def test_renders_final_markdown_and_sends_message() -> None:
    runner = FakeRunner([FakeEvent(text="# Done", final=True)])
    out = run_turn(runner, prompt="do it")
    assert "Done" in out
    assert runner.calls[0]["user_id"] == USER_ID
    assert runner.calls[0]["session_id"] == "sess_1"
    assert runner.calls[0]["new_message"].parts[0].text == "do it"


def test_renders_tool_calls_with_truncated_args() -> None:
    call = SimpleNamespace(
        name="read_file", args={"file_path": "a.py", "content": "x" * 200}
    )
    runner = FakeRunner(
        [
            FakeEvent(calls=[call]),
            FakeEvent(responses=[object()]),
            FakeEvent(text="ok", final=True),
        ]
    )
    out = run_turn(runner)
    assert "Tool Execution: read_file(file_path='a.py', content='xxxx" in out
    assert "x" * 60 not in out
    assert "...)" in out


def test_tool_call_without_name_or_args() -> None:
    call = SimpleNamespace(name=None, args=None)
    runner = FakeRunner([FakeEvent(calls=[call]), FakeEvent(text="ok", final=True)])
    assert "Tool Execution: <unnamed>()" in run_turn(runner)


def test_final_response_skips_non_text_parts() -> None:
    runner = FakeRunner([FakeEvent(parts=["Alpha ", None, "", "Omega"], final=True)])
    assert "Alpha Omega" in run_turn(runner)


def test_final_response_with_only_non_text_parts() -> None:
    out = run_turn(FakeRunner([FakeEvent(parts=[None], final=True)]))
    assert "Turn completed with no direct response." in out


def test_empty_response_message() -> None:
    out = run_turn(FakeRunner([FakeEvent(final=True)]))
    assert "Turn completed with no direct response." in out


def test_prune_report_shown_when_chars_pruned() -> None:
    pruner = ContextPruner()

    class PruningRunner(FakeRunner):
        async def run_async(self, **kwargs: object):  # type: ignore[override]
            pruner.last_pruned_chars = 12345  # as the before_model_callback would
            async for event in super().run_async(**kwargs):
                yield event

    out = run_turn(PruningRunner([FakeEvent(text="ok", final=True)]), pruner=pruner)
    assert "omitted ~12,345 characters of earlier tool output" in out


def test_prune_count_reset_each_turn() -> None:
    pruner = ContextPruner()
    pruner.last_pruned_chars = 999  # stale value from a previous turn
    out = run_turn(FakeRunner([FakeEvent(text="ok", final=True)]), pruner=pruner)
    assert pruner.last_pruned_chars == 0
    assert "Context Optimized" not in out


def test_no_prune_report_without_pruner() -> None:
    out = run_turn(FakeRunner([FakeEvent(text="ok", final=True)]))
    assert "Context Optimized" not in out


@pytest.mark.parametrize(
    "error", [ValueError("bad value"), KeyError("k"), CodebaseToolError("tool")]
)
def test_handled_errors_are_reported(error: Exception) -> None:
    out = run_turn(FakeRunner(error=error))
    assert "Execution Error:" in out


def test_model_api_errors_keep_session_alive() -> None:
    err = litellm.RateLimitError("[bold]TPM limit", llm_provider="openai", model="m")
    out = run_turn(FakeRunner(error=err))
    assert "Model API Error:" in out
    assert "[bold]TPM limit" in out  # markup escaped
    assert "send the prompt again to retry" in out


def test_quota_exhaustion_propagates() -> None:
    with pytest.raises(QuotaExhaustedError):
        run_turn(FakeRunner(error=QuotaExhaustedError("daily", reset_hint="1h")))


def test_unhandled_errors_propagate() -> None:
    with pytest.raises(RuntimeError):
        run_turn(FakeRunner(error=RuntimeError("boom")))
