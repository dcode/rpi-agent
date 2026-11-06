from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from typing import Any

import pytest

from rpi_agent import __version__, cli
from rpi_agent.constants import DEFAULT_SKILLS_DIR
from rpi_agent.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, git_repo: Path) -> None:
    for var in ("STARK_API_KEY", "STARK_BASE_URL", "STARK_MODEL"):
        monkeypatch.delenv(var, raising=False)
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
    assert args.skills_dir == DEFAULT_SKILLS_DIR
    assert args.model is None and args.base_url is None and args.api_key is None
    assert not any(
        (
            args.yolo,
            args.no_prune,
            args.test_discovery,
            args.test_tools,
            args.test_agent,
        )
    )


def test_parse_all_options(tmp_path: Path) -> None:
    args = cli.parse_arguments(
        [
            "--skills-dir", str(tmp_path),
            "--model", "m",
            "--base-url", "u",
            "--api-key", "k",
            "--yolo",
            "--no-prune",
            "--test-discovery",
            "--test-tools",
            "--test-agent",
        ]
    )  # fmt: skip
    assert args.skills_dir == tmp_path
    assert (args.model, args.base_url, args.api_key) == ("m", "u", "k")
    assert args.yolo and args.no_prune


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.parse_arguments(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"rpi-agent {__version__}"


def test_main_starts_repl(repl_calls: list[tuple[Any, ...]], git_repo: Path) -> None:
    asyncio.run(cli.main(["--skills-dir", str(git_repo / "none")]))
    (call,) = repl_calls
    config, repo_ctx, registry, runner, session_service = call
    assert config.api_key == "env-key"
    assert repo_ctx.root_path == git_repo.resolve()
    assert registry.skills == {}
    assert runner.agent.name == "rpi_root_agent"
    assert runner.session_service is session_service
    # 6 read + 5 write + run_subagent
    assert len(runner.agent.tools) == 12


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
