"""Self-check diagnostics invoked via the --test-* CLI flags."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from rpi_agent import __version__ as VERSION
from rpi_agent.agents import SubAgentManager, build_root_agent
from rpi_agent.config import AppConfig
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry
from rpi_agent.tools import CodebaseTools


def run_discovery_diagnostic(config: AppConfig, registry: SkillRegistry) -> None:
    """Runs diagnostics inspecting the skills directory and prints a Rich summary."""
    console = Console()
    count = registry.scan()
    console.print(
        f"\n[bold green]Skill Discovery Diagnostic ({count} skills found)[/bold green]"
    )
    console.print(f"Directory: [cyan]{config.skills_dir}[/cyan]\n")

    if count == 0:
        console.print(
            f"[yellow]No skills found in `{escape(str(config.skills_dir))}`.\n"
            "Ensure skills exist with the layout: ~/.agents/skills/<skill-name>/SKILL.md[/yellow]"
        )
        return

    table = Table(title="Loaded Skills", border_style="cyan")
    table.add_column("Skill Name", style="bold yellow")
    table.add_column("Description", style="white")
    table.add_column("Resources", style="dim")

    for skill in registry.skills.values():
        table.add_row(
            skill.name,
            skill.description,
            str(len(skill.resources)),
        )

    console.print(table)


def run_tools_diagnostic(repo_root: Path, yolo: bool) -> None:
    """Runs diagnostic tests on CodebaseTools operations."""
    console = Console()
    tools = CodebaseTools(repo_root, yolo=yolo)
    console.print(
        f"\n[bold green]Running CodebaseTools Diagnostic for:[/bold green] {repo_root}\n"
    )

    meta = tools.get_repo_metadata()
    console.print(Panel(Text(meta), title="Repo Metadata Check", border_style="cyan"))

    files_found = tools.find_files("*.py")
    console.print(
        Panel(Text(files_found), title="find_files (*.py)", border_style="blue")
    )

    sample_search = tools.search_code("LlmAgent", max_matches=3)
    console.print(
        Panel(
            Text(sample_search),
            title="search_code ('LlmAgent')",
            border_style="magenta",
        )
    )

    # Outline this module when running from a checkout inside the repo, else the README.
    this_file = Path(__file__).resolve()
    outline_target = (
        this_file.relative_to(tools.repo_root).as_posix()
        if this_file.is_relative_to(tools.repo_root)
        else "README.md"
    )
    outline = tools.outline_file(outline_target)
    console.print(
        Panel(
            Text(outline[:600] + "\n..."),
            title=f"outline_file ('{outline_target}')",
            border_style="green",
        )
    )


def run_agent_diagnostic(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    subagent_mgr: SubAgentManager,
    tools: list[Any],
) -> None:
    """Runs diagnostic verification on ADK agent initialization."""
    console = Console()
    console.print(
        f"\n[bold green]Running ADK Agent Setup Diagnostic (v{VERSION})...[/bold green]"
    )
    registry.scan()
    root_agent = build_root_agent(config, repo_ctx, registry, subagent_mgr, tools)
    console.print(
        Panel(
            f"Agent Name: {root_agent.name}\n"
            f"Version: {VERSION}\n"
            f"Tools Registered: {len(root_agent.tools)}\n"
            f"Token Pruning: {'Enabled' if config.prune_context else 'Disabled'}\n"
            f"Model Configuration: {config.model_name} -> {config.base_url}",
            title="Agent Initialization Verified",
            border_style="green",
        )
    )
