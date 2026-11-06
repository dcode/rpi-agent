from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from rich.console import Console

from rpi_agent.config import AppConfig
from rpi_agent.constants import USER_ID
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.repl import turn as turn_module
from rpi_agent.repl.turn import execute_agent_turn
from tests.fakes import FakeEvent, FakeRunner


def run_turn(runner: FakeRunner, config: AppConfig, prompt: str = "hi") -> str:
    console = Console(record=True, width=200, color_system=None)
    asyncio.run(
        execute_agent_turn(
            cast(Runner, runner),
            InMemorySessionService(),
            prompt,
            "sess_1",
            config,
            console,
        )
    )
    return console.export_text()


@pytest.fixture
def no_prune(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    seen: list[str] = []

    async def fake_prune(_service: object, session_id: str) -> int:
        seen.append(session_id)
        return 0

    monkeypatch.setattr(turn_module, "prune_session_history", fake_prune)
    return seen


def test_renders_final_markdown_and_sends_message(
    config: AppConfig, no_prune: list[str]
) -> None:
    runner = FakeRunner([FakeEvent(text="# Done", final=True)])
    out = run_turn(runner, config, prompt="do it")
    assert "Done" in out
    assert runner.calls[0]["user_id"] == USER_ID
    assert runner.calls[0]["session_id"] == "sess_1"
    assert runner.calls[0]["new_message"].parts[0].text == "do it"
    assert no_prune == ["sess_1"]


def test_renders_tool_calls_with_truncated_args(
    config: AppConfig, no_prune: list[str]
) -> None:
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
    out = run_turn(runner, config)
    assert "Tool Execution: read_file(file_path='a.py', content='xxxx" in out
    assert "x" * 60 not in out
    assert "...)" in out


def test_tool_call_without_name_or_args(config: AppConfig, no_prune: list[str]) -> None:
    call = SimpleNamespace(name=None, args=None)
    runner = FakeRunner([FakeEvent(calls=[call]), FakeEvent(text="ok", final=True)])
    assert "Tool Execution: <unnamed>()" in run_turn(runner, config)


def test_final_response_skips_non_text_parts(
    config: AppConfig, no_prune: list[str]
) -> None:
    runner = FakeRunner([FakeEvent(parts=["Alpha ", None, "", "Omega"], final=True)])
    assert "Alpha Omega" in run_turn(runner, config)


def test_final_response_with_only_non_text_parts(
    config: AppConfig, no_prune: list[str]
) -> None:
    out = run_turn(FakeRunner([FakeEvent(parts=[None], final=True)]), config)
    assert "Turn completed with no direct response." in out


def test_empty_response_message(config: AppConfig, no_prune: list[str]) -> None:
    out = run_turn(FakeRunner([FakeEvent(final=True)]), config)
    assert "Turn completed with no direct response." in out


def test_prune_report_shown_when_chars_pruned(
    config: AppConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_prune(_service: object, _session_id: str) -> int:
        return 12345

    monkeypatch.setattr(turn_module, "prune_session_history", fake_prune)
    out = run_turn(FakeRunner([FakeEvent(text="ok", final=True)]), config)
    assert "pruned ~12,345 characters" in out


def test_prune_skipped_when_disabled(
    make_config: Callable[..., AppConfig], no_prune: list[str]
) -> None:
    run_turn(
        FakeRunner([FakeEvent(text="ok", final=True)]), make_config(prune_context=False)
    )
    assert no_prune == []


@pytest.mark.parametrize(
    "error", [ValueError("bad value"), KeyError("k"), CodebaseToolError("tool")]
)
def test_handled_errors_are_reported(
    config: AppConfig, no_prune: list[str], error: Exception
) -> None:
    out = run_turn(FakeRunner(error=error), config)
    assert "Execution Error:" in out


def test_unhandled_errors_propagate(config: AppConfig, no_prune: list[str]) -> None:
    with pytest.raises(RuntimeError):
        run_turn(FakeRunner(error=RuntimeError("boom")), config)
