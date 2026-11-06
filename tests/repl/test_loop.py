from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from rpi_agent.config import AppConfig
from rpi_agent.repl import loop as loop_module
from rpi_agent.repl.loop import repl_loop
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry


class FakePromptSession:
    """Feeds scripted input lines, then raises the terminal exception."""

    instances: list[FakePromptSession] = []

    def __init__(
        self, inputs: list[str], end: type[BaseException], **kwargs: Any
    ) -> None:
        self.inputs = list(inputs)
        self.end = end
        self.kwargs = kwargs
        self.toolbars: list[str] = []
        FakePromptSession.instances.append(self)

    async def prompt_async(
        self, message: Any, bottom_toolbar: Callable[[], str]
    ) -> str:
        self.toolbars.append(bottom_toolbar())
        if not self.inputs:
            raise self.end
        return self.inputs.pop(0)


@pytest.fixture
def run_loop(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Callable[..., tuple[str, list[str], FakePromptSession, InMemorySessionService]]:
    def _run(
        inputs: list[str], end: type[BaseException] = EOFError
    ) -> tuple[str, list[str], FakePromptSession, InMemorySessionService]:
        dispatched: list[str] = []
        FakePromptSession.instances.clear()

        async def fake_turn(
            _runner: Any, _svc: Any, prompt: str, session_id: str, *_rest: Any
        ) -> None:
            dispatched.append(f"{session_id}|{prompt}")

        monkeypatch.setattr(loop_module, "execute_agent_turn", fake_turn)
        monkeypatch.setattr(
            loop_module,
            "PromptSession",
            lambda **kw: FakePromptSession(inputs, end, **kw),
        )
        service = InMemorySessionService()
        runner = cast(Runner, object())  # execute_agent_turn is faked
        asyncio.run(repl_loop(config, repo_ctx, registry, runner, service))
        return (
            capsys.readouterr().out,
            dispatched,
            FakePromptSession.instances[0],
            service,
        )

    return _run


def test_exit_command(run_loop: Callable[..., Any]) -> None:
    out, dispatched, _, _ = run_loop(["/exit"])
    assert "RPI Agent REPL" in out
    assert "Goodbye!" in out
    assert dispatched == []


def test_quit_command(run_loop: Callable[..., Any]) -> None:
    out, *_ = run_loop(["/quit"])
    assert "Goodbye!" in out


@pytest.mark.parametrize("end", [EOFError, KeyboardInterrupt])
def test_eof_and_interrupt_exit(
    run_loop: Callable[..., Any], end: type[BaseException]
) -> None:
    out, *_ = run_loop([], end=end)
    assert "Session cancelled by user. Exiting." in out


def test_prompt_session_uses_repo_history_and_completer(
    run_loop: Callable[..., Any], repo_ctx: RepoContext
) -> None:
    _, _, session, _ = run_loop(["/exit"])
    history = session.kwargs["history"]
    assert Path(history.filename) == repo_ctx.root_path / ".rpi_agent" / "history"
    assert type(session.kwargs["completer"]).__name__ == "RpiCliCompleter"


def test_builtin_commands_do_not_dispatch(run_loop: Callable[..., Any]) -> None:
    out, dispatched, _, _ = run_loop(
        ["", "   ", "/help", "/skills", "/status", "/exit"]
    )
    assert dispatched == []
    assert "RPI Agent Slash Commands" in out
    assert "Available Skills Catalog" in out
    assert "Active Session Status" in out


def test_prompts_dispatch_and_count_turns(run_loop: Callable[..., Any]) -> None:
    _, dispatched, session, _ = run_loop(["hello", "/plan caching", "/exit"])
    prompts = [d.split("|", 1)[1] for d in dispatched]
    assert prompts == ["hello", "Please execute the 'rpi-plan' skill for: caching"]
    assert session.toolbars[0].endswith("<b>Turns:</b> 0")
    assert session.toolbars[-1].endswith("<b>Turns:</b> 2")


def test_clear_starts_new_session(run_loop: Callable[..., Any]) -> None:
    out, dispatched, session, service = run_loop(
        ["one", "/clear", "two", "/new", "/exit"]
    )
    first, second = (d.split("|", 1)[0] for d in dispatched)
    assert first != second
    assert out.count("Session context reset") == 2
    assert session.toolbars[-1].endswith("<b>Turns:</b> 0")
    assert len(service.sessions["rpi_agent_app"]["local_developer"]) == 3
