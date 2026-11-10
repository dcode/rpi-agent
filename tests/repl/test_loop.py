from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest
from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService

from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import AppConfig
from rpi_agent.exceptions import ConfigurationError, QuotaExhaustedError
from rpi_agent.repl import loop as loop_module
from rpi_agent.repl.loop import repl_loop
from rpi_agent.repo import RepoContext
from rpi_agent.sessions import create_session, list_sessions, open_session_service
from rpi_agent.skills import SkillRegistry
from tests.fakes import FakePromptSession


@dataclass
class LoopResult:
    out: str
    dispatched: list[tuple[str, str]]
    prompt: FakePromptSession
    service: BaseSessionService
    turn_kwargs: list[tuple[Any, ...]]


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    path = tmp_path / "state"
    path.mkdir()
    return path


@pytest.fixture
def checkpoints(state_dir: Path) -> CheckpointStore:
    return CheckpointStore(state_dir / "subagents")


@pytest.fixture
def run_loop(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    state_dir: Path,
    checkpoints: CheckpointStore,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Callable[..., LoopResult]:
    def _run(
        inputs: list[str],
        end: type[BaseException] = EOFError,
        turn_error: Exception | None = None,
        resume: str | None = None,
        service: BaseSessionService | None = None,
    ) -> LoopResult:
        dispatched: list[tuple[str, str]] = []
        turn_args: list[tuple[Any, ...]] = []
        FakePromptSession.instances.clear()

        async def fake_turn(
            _runner: Any, prompt: str, session_id: str, *rest: Any
        ) -> None:
            dispatched.append((session_id, prompt))
            turn_args.append(rest)
            if turn_error is not None:
                raise turn_error

        monkeypatch.setattr(loop_module, "execute_agent_turn", fake_turn)
        monkeypatch.setattr(
            loop_module,
            "PromptSession",
            lambda **kw: FakePromptSession(inputs, end, **kw),
        )
        svc = service or open_session_service(state_dir)
        runner = cast(Runner, object())  # execute_agent_turn is faked
        asyncio.run(
            repl_loop(
                replace(config, resume=resume),
                repo_ctx,
                registry,
                runner,
                svc,
                state_dir,
                checkpoints,
            )
        )
        return LoopResult(
            capsys.readouterr().out,
            dispatched,
            FakePromptSession.instances[0],
            svc,
            turn_args,
        )

    return _run


def test_exit_command(run_loop: Callable[..., LoopResult]) -> None:
    result = run_loop(["/exit"])
    assert "RPI Agent REPL" in result.out
    assert "Goodbye!" in result.out
    assert result.dispatched == []


def test_quit_command(run_loop: Callable[..., LoopResult]) -> None:
    assert "Goodbye!" in run_loop(["/quit"]).out


@pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
def test_eof_and_interrupt_exit(
    run_loop: Callable[..., LoopResult], end: type[BaseException]
) -> None:
    assert "Session cancelled by user. Exiting." in run_loop([], end=end).out


def test_prompt_session_uses_state_dir_history_and_completer(
    run_loop: Callable[..., LoopResult], state_dir: Path
) -> None:
    prompt = run_loop(["/exit"]).prompt
    assert Path(prompt.kwargs["history"].filename) == state_dir / "history"
    assert type(prompt.kwargs["completer"]).__name__ == "RpiCliCompleter"


def test_builtin_commands_do_not_dispatch(run_loop: Callable[..., LoopResult]) -> None:
    result = run_loop(["", "   ", "/help", "/skills", "/status", "/sessions", "/exit"])
    assert result.dispatched == []
    assert "RPI Agent Slash Commands" in result.out
    assert "Available Skills Catalog" in result.out
    assert "Active Session Status" in result.out
    assert "Saved Sessions" in result.out


def test_prompts_dispatch_and_count_turns(run_loop: Callable[..., LoopResult]) -> None:
    result = run_loop(["hello", "/plan caching", "/exit"])
    assert [p for _, p in result.dispatched] == [
        "hello",
        "Please execute the 'rpi-plan' skill for: caching",
    ]
    assert result.prompt.toolbars[0].endswith("<b>Turns:</b> 0")
    assert result.prompt.toolbars[-1].endswith("<b>Turns:</b> 2")
    # console and pruner (None here) are forwarded to the turn.
    assert result.turn_kwargs[0][1] is None


def test_new_session_is_persisted(run_loop: Callable[..., LoopResult]) -> None:
    result = run_loop(["hello", "/exit"])

    async def ids() -> list[str]:
        return [s.id for s in await list_sessions(result.service)]

    assert asyncio.run(ids()) == [result.dispatched[0][0]]


def test_clear_starts_new_session(run_loop: Callable[..., LoopResult]) -> None:
    result = run_loop(["one", "/clear", "two", "/new", "/exit"])
    first, second = (sid for sid, _ in result.dispatched)
    assert first != second
    assert result.out.count("Session context reset") == 2
    assert result.prompt.toolbars[-1].endswith("<b>Turns:</b> 0")

    async def count() -> int:
        return len(await list_sessions(result.service))

    assert asyncio.run(count()) == 3


def test_resume_continues_saved_session(
    run_loop: Callable[..., LoopResult],
    state_dir: Path,
    checkpoints: CheckpointStore,
) -> None:
    service = open_session_service(state_dir)
    first = run_loop(["research auth", "/exit"], service=service)
    session_id = first.dispatched[0][0]
    checkpoints.save(session_id, "rpi-codebase-locator", "find auth", "found")

    resumed = run_loop(["continue", "/exit"], resume=session_id, service=service)
    assert resumed.dispatched == [(session_id, "continue")]
    flat = " ".join(resumed.out.split())  # Rich wraps long lines
    assert (
        f"Resumed session {session_id} (0 prior prompts, 1 saved sub-agent results)"
        in flat
    )


def test_resume_latest(run_loop: Callable[..., LoopResult], state_dir: Path) -> None:
    service = open_session_service(state_dir)

    async def make() -> str:
        await create_session(service)
        await asyncio.sleep(0.01)
        return await create_session(service)

    latest = asyncio.run(make())
    result = run_loop(["go", "/exit"], resume="latest", service=service)
    assert result.dispatched == [(latest, "go")]


def test_resume_unknown_session_raises(run_loop: Callable[..., LoopResult]) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        run_loop(["/exit"], resume="sess_missing")


def test_quota_exhaustion_stops_with_resume_instructions(
    run_loop: Callable[..., LoopResult], checkpoints: CheckpointStore
) -> None:
    error = QuotaExhaustedError("Daily credit limit exceeded", reset_hint="24h 42m")
    result = run_loop(["research", "never sent"], turn_error=error)

    session_id = result.dispatched[0][0]
    assert len(result.dispatched) == 1
    assert "Model Quota Exhausted" in result.out
    assert "Quota resets in 24h 42m." in result.out
    assert f"rpi-agent --resume {session_id}" in result.out
    assert "Session cancelled" not in result.out


def test_logs_commands_toggle_level_clear_and_save(
    run_loop: Callable[..., LoopResult],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ensure fallback SessionLogBuffer creation is covered when active buffer is None
    monkeypatch.setattr(loop_module, "get_active_log_buffer", lambda: None)
    save_file = tmp_path / "saved_logs.jsonl"
    bad_dir = tmp_path / "blocker"
    bad_dir.write_text("file", encoding="utf-8")

    result = run_loop(
        [
            "/logs",
            "/logs toggle",
            "/logs show",
            "/logs hide",
            "/logs INFO",
            "/logs level DEBUG",
            "/logs WARNING",
            "/logs invalid_level",
            "/logs save",
            f"/logs save {save_file} DEBUG",
            f"/logs save {save_file}",
            f"/logs save {save_file} INVALID",
            f"/logs save {bad_dir / 'sub.jsonl'}",
            "/logs clear",
            "/exit",
        ]
    )
    assert "Log pane shown" in result.out
    assert "Log pane hidden" in result.out
    assert "Log level set to INFO" in result.out
    assert "Log level set to DEBUG" in result.out
    assert "Log level set to WARNING" in result.out
    assert "Invalid log level 'invalid_level'" in result.out
    assert "Usage: /logs save <path>" in result.out
    assert "Saved" in result.out
    assert "Failed to save logs:" in result.out
    assert "Session log buffer cleared" in result.out
    assert save_file.is_file()
