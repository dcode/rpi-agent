from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from rich.console import Console

from rpi_agent import __version__
from rpi_agent.config import AppConfig
from rpi_agent.repl.display import (
    display_help,
    display_status,
    display_welcome_banner,
    get_bottom_toolbar,
)
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry


def recording_console() -> Console:
    return Console(record=True, width=200, color_system=None)


def test_welcome_banner_protected(
    config: AppConfig, repo_ctx: RepoContext, registry: SkillRegistry
) -> None:
    console = recording_console()
    display_welcome_banner(console, config, repo_ctx, registry)
    out = console.export_text()
    assert f"RPI Agent REPL v{__version__}" in out
    assert f"Repository: sample-repo ({repo_ctx.root_path})" in out
    assert f"Commit: {repo_ctx.commit_sha[:8]}" in out
    assert "test-model via http://gateway.test/v1" in out
    assert "2 skills discovered" in out
    assert "Active (Pruning tool outputs)" in out
    assert "Protected (Prompt [y/N])" in out


def test_welcome_banner_yolo_no_prune(
    make_config: Callable[..., AppConfig],
    repo_ctx: RepoContext,
    registry: SkillRegistry,
) -> None:
    console = recording_console()
    display_welcome_banner(
        console, make_config(yolo=True, prune_context=False), repo_ctx, registry
    )
    out = console.export_text()
    assert "YOLO (Auto-Approve)" in out
    assert "Token Guard: Disabled" in out


def test_help_lists_commands() -> None:
    console = recording_console()
    display_help(console)
    out = console.export_text()
    for cmd in ("/research <topic>", "/plan <topic>", "/skills", "/exit, /quit"):
        assert cmd in out


def test_status_panel(
    make_config: Callable[..., AppConfig], repo_ctx: RepoContext
) -> None:
    for kwargs, pruning, policy in (
        ({}, "Enabled", "Protected"),
        ({"yolo": True, "prune_context": False}, "Disabled", "YOLO"),
    ):
        console = recording_console()
        display_status(console, make_config(**kwargs), repo_ctx, "sess_abc", 4)
        out = console.export_text()
        assert "Session ID: sess_abc" in out
        assert "Turns: 4" in out
        assert f"Token Pruning: {pruning}" in out
        assert f"Safety Policy: {policy}" in out


def test_bottom_toolbar_escapes_html(make_config: Callable[..., AppConfig]) -> None:
    ctx = RepoContext(
        root_path=Path("/x"),
        repo_name="<repo>",
        branch_name="a&b",
        commit_sha="0" * 40,
        is_git=True,
    )
    toolbar = get_bottom_toolbar(
        ctx, make_config(model_name="m<1>"), "sess_123456789", 7
    )
    assert toolbar == (
        "<b>Repo:</b> &lt;repo&gt; | <b>Branch:</b> a&amp;b | "
        "<b>Model:</b> m&lt;1&gt; | <b>Session:</b> sess_123 | <b>Turns:</b> 7"
    )
