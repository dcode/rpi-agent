"""Rich and prompt_toolkit rendering helpers for the REPL."""

from __future__ import annotations

import html
from datetime import datetime

from google.adk.sessions import Session
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rpi_agent import __version__ as VERSION
from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import AppConfig
from rpi_agent.exceptions import QuotaExhaustedError
from rpi_agent.llm import RetryNotifier
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry


def display_welcome_banner(
    console: Console,
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
) -> None:
    """Renders the Rich welcome banner upon REPL initialization.

    Args:
        console: Console instance.
        config: Application configuration.
        repo_ctx: Repository metadata.
        registry: Skills registry.
    """
    safety_style = "bold red" if config.yolo else "bold green"
    safety_text = "YOLO (Auto-Approve)" if config.yolo else "Protected (Prompt [y/N])"
    prune_text = "Active (Pruning tool outputs)" if config.prune_context else "Disabled"

    content = Text()
    content.append("• Version: ", style="bold cyan")
    content.append(f"v{VERSION}\n")
    content.append("• Repository: ", style="bold cyan")
    content.append(f"{repo_ctx.repo_name} ({repo_ctx.root_path})\n")
    content.append("• Active Branch: ", style="bold cyan")
    content.append(f"{repo_ctx.branch_name} | Commit: {repo_ctx.commit_sha[:8]}\n")
    content.append("• Model Target: ", style="bold cyan")
    content.append(f"{config.model_name} via {config.base_url}\n")
    content.append("• Skills Loaded: ", style="bold cyan")
    content.append(f"{len(registry.skills)} skills discovered in {config.skills_dir}\n")
    content.append("• Token Guard: ", style="bold cyan")
    content.append(f"{prune_text} | Structure Outlining: Enabled\n", style="green")
    content.append("• API Retries: ", style="bold cyan")
    content.append(f"up to {config.max_retries} with backoff\n")
    content.append("• Logging: ", style="bold cyan")
    pane_status = (
        "Split-Pane Open" if config.log_pane else "Split-Pane Hidden (F2 or /logs)"
    )
    content.append(f"{config.log_level} | {pane_status}\n")
    content.append("• Shell Safety: ", style="bold cyan")
    content.append(safety_text, style=safety_style)
    content.append("\n\nType ")
    content.append("/help", style="bold yellow")
    content.append(" for slash commands or ")
    content.append("/exit", style="bold yellow")
    content.append(" to quit.")

    console.print(
        Panel(
            content,
            title=f"[bold blue]🤖 RPI Agent REPL v{VERSION} (Google ADK)[/bold blue]",
            subtitle="[dim]Research • Plan • Iterate • Implement • Token-Optimized[/dim]",
            border_style="blue",
        )
    )


def display_help(console: Console) -> None:
    """Renders the slash command reference table.

    Args:
        console: Console instance.
    """
    table = Table(title="RPI Agent Slash Commands", border_style="cyan")
    table.add_column("Command", style="bold yellow")
    table.add_column("Description", style="white")
    table.add_row("/research <topic>", "Run RPI Research phase on topic")
    table.add_row("/plan <topic>", "Run RPI Plan phase based on research")
    table.add_row("/iterate <feedback>", "Refine existing plan artifact")
    table.add_row("/implement <plan>", "Execute implementation plan")
    table.add_row("/skills", "List all discovered skills in ~/.agents/skills")
    table.add_row("/status", "Show active repository, model, and token guard status")
    table.add_row(
        "/logs [level|save]",
        "Toggle log pane (F2), set level (INFO/DEBUG), or save logs to disk",
    )
    table.add_row("/clear, /new", "Reset context window with new session ID")
    table.add_row("/sessions", "List saved sessions (resume with --resume <id>)")
    table.add_row("/exit, /quit", "Exit the CLI agent")
    console.print(table)


def display_status(
    console: Console,
    config: AppConfig,
    repo_ctx: RepoContext,
    session_id: str,
    turn_counter: int,
) -> None:
    """Renders the active session status panel.

    Args:
        console: Console instance.
        config: Application configuration.
        repo_ctx: Repository metadata.
        session_id: Active session ID.
        turn_counter: Current session turn count.
    """
    console.print(
        Panel(
            f"[bold]Version:[/bold] v{VERSION}\n"
            f"[bold]Repository:[/bold] {repo_ctx.repo_name}\n"
            f"[bold]Root:[/bold] {repo_ctx.root_path}\n"
            f"[bold]Branch:[/bold] {repo_ctx.branch_name}\n"
            f"[bold]Commit:[/bold] {repo_ctx.commit_sha}\n"
            f"[bold]Model:[/bold] {config.model_name}\n"
            f"[bold]Base URL:[/bold] {config.base_url}\n"
            f"[bold]Session ID:[/bold] {session_id}\n"
            f"[bold]Turns:[/bold] {turn_counter}\n"
            f"[bold]Token Pruning:[/bold] {'Enabled' if config.prune_context else 'Disabled'}\n"
            f"[bold]Max Retries:[/bold] {config.max_retries}\n"
            f"[bold]Log Level:[/bold] {config.log_level}\n"
            f"[bold]Safety Policy:[/bold] {'YOLO' if config.yolo else 'Protected'}",
            title="Active Session Status",
            border_style="cyan",
        )
    )


def get_bottom_toolbar(
    repo_ctx: RepoContext, config: AppConfig, session_id: str, turn_counter: int
) -> str:
    """Generates the HTML formatted bottom toolbar for prompt_toolkit.

    Args:
        repo_ctx: Active repository context.
        config: Session configuration.
        session_id: Active session ID.
        turn_counter: Current session turn count.

    Returns:
        HTML formatted string for bottom status toolbar.
    """
    safe_repo = html.escape(repo_ctx.repo_name)
    safe_branch = html.escape(repo_ctx.branch_name)
    safe_model = html.escape(config.model_name)
    safe_sess = html.escape(session_id[:8])
    return (
        f"<b>Repo:</b> {safe_repo} | "
        f"<b>Branch:</b> {safe_branch} | "
        f"<b>Model:</b> {safe_model} | "
        f"<b>Session:</b> {safe_sess} | "
        f"<b>Turns:</b> {turn_counter}"
    )


def display_sessions(
    console: Console,
    sessions: list[Session],
    current_session_id: str,
    checkpoints: CheckpointStore,
) -> None:
    """Renders saved sessions, most recent first.

    Args:
        console: Console instance.
        sessions: Sessions to list, already ordered.
        current_session_id: Active session, marked in the table.
        checkpoints: Store used to count saved sub-agent results per session.
    """
    if not sessions:
        console.print("[dim]No saved sessions in this repository.[/dim]")
        return

    table = Table(title="Saved Sessions", border_style="cyan")
    table.add_column("Session ID", style="bold yellow")
    table.add_column("Last Updated", style="white")
    table.add_column("Saved Sub-agent Results", style="dim", justify="right")
    for session in sessions:
        marker = " (current)" if session.id == current_session_id else ""
        updated = datetime.fromtimestamp(session.last_update_time).strftime(
            "%Y-%m-%d %H:%M"
        )
        table.add_row(
            f"{session.id}{marker}",
            updated,
            str(len(checkpoints.for_session(session.id))),
        )
    console.print(table)
    console.print("[dim]Resume a session with: rpi-agent --resume <session-id>[/dim]")


def display_quota_exhausted(
    console: Console,
    error: QuotaExhaustedError,
    session_id: str,
    saved_results: int,
) -> None:
    """Explains that quota is exhausted and how to resume once it resets.

    Args:
        console: Console instance.
        error: The quota error raised by the model client.
        session_id: Session to resume.
        saved_results: Number of sub-agent results checkpointed in the session.
    """
    reset = f"Quota resets in {error.reset_hint}." if error.reset_hint else ""
    console.print(
        Panel(
            f"{escape(str(error))}\n\n"
            f"{reset}\n"
            f"Session [bold]{session_id}[/bold] is saved with {saved_results} "
            "completed sub-agent result(s), which will be reused instead of re-run.\n\n"
            f"Resume with: [bold yellow]rpi-agent --resume {session_id}[/bold yellow]",
            title="Model Quota Exhausted",
            border_style="red",
        )
    )


def make_retry_notifier(console: Console) -> RetryNotifier:
    """Builds a callback that reports model retries as they are scheduled.

    Args:
        console: Console to print notices on.

    Returns:
        Callback accepting (attempt, max_retries, delay_seconds, error).
    """

    def notify(attempt: int, max_retries: int, delay: float, error: Exception) -> None:
        console.print(
            f"[yellow]⏳ {type(error).__name__} from model gateway; retrying in "
            f"{delay:.0f}s (attempt {attempt}/{max_retries}).[/yellow]"
        )

    return notify
