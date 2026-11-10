from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from google.adk.sessions import Session
from rich.console import Console

from rpi_agent import __version__
from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import AppConfig
from rpi_agent.exceptions import QuotaExhaustedError
from rpi_agent.repl.display import (
    display_help,
    display_quota_exhausted,
    display_sessions,
    display_status,
    display_welcome_banner,
    get_bottom_toolbar,
    make_retry_notifier,
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
    assert f"API Retries: up to {config.max_retries} with backoff" in out
    assert "Protected (Prompt [y/N])" in out


def test_welcome_banner_yolo_no_prune(
    make_config: Callable[..., AppConfig],
    repo_ctx: RepoContext,
    registry: SkillRegistry,
) -> None:
    console = recording_console()
    display_welcome_banner(
        console,
        make_config(yolo=True, prune_context=False, log_pane=True, log_level="INFO"),
        repo_ctx,
        registry,
    )
    out = console.export_text()
    assert "YOLO (Auto-Approve)" in out
    assert "Token Guard: Disabled" in out
    assert "INFO | Split-Pane Open" in out


def test_help_lists_commands() -> None:
    console = recording_console()
    display_help(console)
    out = console.export_text()
    for cmd in (
        "/research <topic>",
        "/plan <topic>",
        "/skills",
        "/sessions",
        "/exit, /quit",
    ):
        assert cmd in out


def test_status_panel(
    make_config: Callable[..., AppConfig], repo_ctx: RepoContext
) -> None:
    for kwargs, pruning, policy in (
        ({}, "Enabled", "Protected"),
        ({"yolo": True, "prune_context": False}, "Disabled", "YOLO"),
    ):
        console = recording_console()
        cfg = make_config(**kwargs)
        display_status(console, cfg, repo_ctx, "sess_abc", 4)
        out = console.export_text()
        assert "Session ID: sess_abc" in out
        assert "Turns: 4" in out
        assert f"Token Pruning: {pruning}" in out
        assert f"Max Retries: {cfg.max_retries}" in out
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


def test_display_sessions_empty(tmp_path: Path) -> None:
    console = recording_console()
    checkpoints = CheckpointStore(tmp_path)
    display_sessions(console, [], "sess_cur", checkpoints)
    assert "No saved sessions in this repository." in console.export_text()


def test_display_sessions_table(tmp_path: Path) -> None:
    console = recording_console()
    checkpoints = CheckpointStore(tmp_path)
    checkpoints.save("sess_new", "locator", "find a", "res 1")
    checkpoints.save("sess_new", "analyzer", "read b", "res 2")
    sessions = [
        Session(
            id="sess_new",
            app_name="rpi_agent",
            user_id="local_user",
            last_update_time=1_700_000_100.0,
        ),
        Session(
            id="sess_old",
            app_name="rpi_agent",
            user_id="local_user",
            last_update_time=1_700_000_000.0,
        ),
    ]
    display_sessions(console, sessions, "sess_new", checkpoints)
    out = console.export_text()
    assert "Saved Sessions" in out
    assert "sess_new (current)" in out
    assert "sess_old" in out
    assert out.index("sess_new (current)") < out.index("sess_old")
    assert "2" in out
    assert "0" in out
    assert "Resume a session with: rpi-agent --resume <session-id>" in out


def test_display_quota_exhausted() -> None:
    for hint, expected_reset in (
        ("24h 42m", "Quota resets in 24h 42m."),
        (None, ""),
    ):
        console = recording_console()
        err = QuotaExhaustedError("limit [5000000/5000000] exceeded", reset_hint=hint)
        display_quota_exhausted(console, err, "sess_q", 3)
        out = console.export_text()
        assert "Model Quota Exhausted" in out
        assert "limit [5000000/5000000] exceeded" in out
        if expected_reset:
            assert expected_reset in out
        else:
            assert "Quota resets in" not in out
        assert "Session sess_q is saved with 3 completed sub-agent result(s)" in out
        assert "Resume with: rpi-agent --resume sess_q" in out


def test_make_retry_notifier() -> None:
    class RateLimitError(Exception):
        pass

    console = recording_console()
    notify = make_retry_notifier(console)
    notify(2, 5, 7.6, RateLimitError("tpm"))
    assert (
        "⏳ RateLimitError from model gateway; retrying in 8s (attempt 2/5)."
        in console.export_text()
    )
