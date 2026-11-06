"""Rich and prompt_toolkit rendering helpers for the REPL."""

from __future__ import annotations

import html

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rpi_agent import __version__ as VERSION
from rpi_agent.config import AppConfig
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
            title=f"[bold blue]🤖 RPI Agent REPL v{VERSION} (Google ADK & GenAI.mil)[/bold blue]",
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
    table.add_row("/clear, /new", "Reset context window with new session ID")
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
