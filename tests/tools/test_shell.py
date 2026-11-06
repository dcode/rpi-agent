from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rpi_agent.tools import CodebaseTools
from rpi_agent.tools import shell as shell_module


@pytest.fixture
def yolo_tools(git_repo: Path) -> CodebaseTools:
    return CodebaseTools(git_repo, yolo=True)


def _forbid_input(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_prompt: str = "") -> str:
        raise AssertionError("should not prompt")

    monkeypatch.setattr("builtins.input", fail)


@pytest.mark.parametrize(
    "command",
    ["git status", "  GIT LOG --oneline", "pwd", "uv run ruff check src"],
)
def test_is_command_safe_true(tools: CodebaseTools, command: str) -> None:
    assert tools.is_command_safe(command)


@pytest.mark.parametrize("command", ["rm -rf /", "git push", "echo pwd"])
def test_is_command_safe_false(tools: CodebaseTools, command: str) -> None:
    assert not tools.is_command_safe(command)


def test_empty_command(tools: CodebaseTools) -> None:
    assert tools.run_shell_command("   ") == "Error: empty command requested."


def test_safe_command_runs_without_prompt(
    tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_input(monkeypatch)
    assert tools.run_shell_command("git branch --show-current").strip() == "main"


def test_unsafe_command_rejected(
    tools: CodebaseTools, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")
    result = tools.run_shell_command("touch created")
    assert result == "Command execution rejected by user: 'touch created'"
    assert not (git_repo / "created").exists()


@pytest.mark.parametrize("answer", ["y", "YES "])
def test_unsafe_command_approved(
    tools: CodebaseTools,
    git_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
) -> None:
    monkeypatch.setattr("builtins.input", lambda _prompt="": answer)
    assert tools.run_shell_command("touch created") == (
        "[Command finished with no output]"
    )
    assert (git_repo / "created").exists()


def test_yolo_skips_prompt(
    yolo_tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    _forbid_input(monkeypatch)
    assert yolo_tools.run_shell_command("echo hi") == "hi\n"


def test_runs_in_repo_root(yolo_tools: CodebaseTools, git_repo: Path) -> None:
    assert Path(yolo_tools.run_shell_command("pwd").strip()).resolve() == (
        git_repo.resolve()
    )


def test_stderr_and_exit_code(yolo_tools: CodebaseTools) -> None:
    result = yolo_tools.run_shell_command("echo out; echo err >&2; exit 3")
    assert result == "out\n\n[stderr]\nerr\n\n[Exit code: 3]"


def test_timeout(yolo_tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="sleep", timeout=120)

    monkeypatch.setattr(shell_module.subprocess, "run", timeout)
    assert yolo_tools.run_shell_command("sleep 999") == (
        "Error: command timed out after 120s: 'sleep 999'"
    )


def test_os_error(yolo_tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("no shell")

    monkeypatch.setattr(shell_module.subprocess, "run", boom)
    assert yolo_tools.run_shell_command("ls") == (
        "Error executing shell command: no shell"
    )
