from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from typing import Any

import pytest
from google.adk.sessions.sqlite_session_service import SqliteSessionService

from rpi_agent import __version__, cli
from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.constants import DEFAULT_SKILLS_DIR
from rpi_agent.exceptions import ConfigurationError
from rpi_agent.pruning import ContextPruner


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, git_repo: Path, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.chdir(git_repo)


@pytest.fixture
def repl_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []

    async def fake_repl(*args: Any) -> None:
        calls.append(args)

    monkeypatch.setattr(cli, "repl_loop", fake_repl)
    return calls


def test_parse_defaults() -> None:
    args = cli.parse_arguments([])
    assert args.config is None
    assert args.skills_dir == DEFAULT_SKILLS_DIR
    assert args.model is None and args.base_url is None and args.api_key is None
    assert args.max_retries is None and args.resume is None
    assert not any(
        (
            args.gemini,
            args.yolo,
            args.no_prune,
            args.test_discovery,
            args.test_tools,
            args.test_agent,
        )
    )


def test_parse_all_options(tmp_path: Path) -> None:
    cfg_file = tmp_path / "custom.yml"
    args = cli.parse_arguments(
        [
            "--config", str(cfg_file),
            "--skills-dir", str(tmp_path),
            "--model", "m",
            "--base-url", "u",
            "--gemini",
            "--api-key", "k",
            "--yolo",
            "--no-prune",
            "--test-discovery",
            "--test-tools",
            "--test-agent",
        ]
    )  # fmt: skip
    assert args.config == cfg_file
    assert args.skills_dir == tmp_path
    assert (args.model, args.base_url, args.api_key) == ("m", "u", "k")
    assert args.gemini and args.yolo and args.no_prune


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], None), (["--resume"], "latest"), (["--resume", "sess_abc"], "sess_abc")],
)
def test_parse_resume(argv: list[str], expected: str | None) -> None:
    assert cli.parse_arguments(argv).resume == expected


def test_parse_max_retries() -> None:
    assert cli.parse_arguments(["--max-retries", "8"]).max_retries == 8


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"rpi-agent {__version__}"


def test_main_starts_repl(repl_calls: list[tuple[Any, ...]], git_repo: Path) -> None:
    asyncio.run(cli.main(["--skills-dir", str(git_repo / "none")]))
    (call,) = repl_calls
    (
        config,
        repo_ctx,
        registry,
        runner,
        session_service,
        state_dir,
        checkpoints,
        pruner,
    ) = call
    assert config.api_key == "env-key"
    assert repo_ctx.root_path == git_repo.resolve()
    assert registry.skills == {}
    assert runner.agent.name == "rpi_root_agent"
    assert runner.session_service is session_service
    # 8 read + 6 write + run_subagent
    assert len(runner.agent.tools) == 15

    assert state_dir == git_repo.resolve() / ".rpi_agent"
    assert isinstance(session_service, SqliteSessionService)
    assert isinstance(checkpoints, CheckpointStore)
    assert checkpoints.root == state_dir / "subagents"
    assert isinstance(pruner, ContextPruner)
    assert runner.agent.before_model_callback is pruner
    assert runner.agent.model.max_retries == config.max_retries


def test_main_no_prune_disables_pruner(repl_calls: list[tuple[Any, ...]]) -> None:
    asyncio.run(cli.main(["--no-prune"]))
    runner, pruner = repl_calls[0][3], repl_calls[0][7]
    assert pruner is None
    assert runner.agent.before_model_callback is None


def test_main_closes_session_service_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []
    original_close = SqliteSessionService.close

    async def tracking_close(self: SqliteSessionService) -> None:
        closed.append(True)
        await original_close(self)

    async def failing_repl(*_args: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(SqliteSessionService, "close", tracking_close)
    monkeypatch.setattr(cli, "repl_loop", failing_repl)
    with pytest.raises(RuntimeError):
        asyncio.run(cli.main([]))
    assert closed == [True]


@pytest.mark.parametrize(
    ("flag", "target"),
    [
        ("--test-discovery", "run_discovery_diagnostic"),
        ("--test-tools", "run_tools_diagnostic"),
        ("--test-agent", "run_agent_diagnostic"),
    ],
)
def test_main_diagnostic_modes_skip_repl(
    repl_calls: list[tuple[Any, ...]],
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    target: str,
) -> None:
    called: list[str] = []
    monkeypatch.setattr(cli, target, lambda *_a, **_k: called.append(target))
    asyncio.run(cli.main([flag]))
    assert called == [target]
    assert repl_calls == []


def test_main_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(ConfigurationError):
        asyncio.run(cli.main([]))


def _patch_main(monkeypatch: pytest.MonkeyPatch, error: Exception | None) -> None:
    async def fake_main(argv: list[str] | None = None) -> None:
        if error:
            raise error

    monkeypatch.setattr(cli, "main", fake_main)


def test_run_success(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_main(monkeypatch, None)
    cli.run()


@pytest.mark.parametrize(
    ("error", "label"),
    [
        (ConfigurationError("no key"), "Configuration Error:"),
        (RuntimeError("[bold]boom"), "Fatal Error:"),
    ],
)
def test_run_reports_errors_and_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
    label: str,
) -> None:
    _patch_main(monkeypatch, error)
    with pytest.raises(SystemExit) as exc:
        cli.run()
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert label in out
    assert str(error) in out  # markup in the message is escaped, not rendered


def test_python_dash_m_entrypoint(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []
    monkeypatch.setattr(cli, "run", lambda: called.append(True))
    runpy.run_module("rpi_agent", run_name="__main__")
    assert called == [True]
