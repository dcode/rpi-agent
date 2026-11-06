"""Main interactive REPL loop."""

from __future__ import annotations

import uuid

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

from rpi_agent.config import AppConfig, resolve_history_file_path
from rpi_agent.constants import APP_NAME, USER_ID
from rpi_agent.repl.commands import expand_command
from rpi_agent.repl.completer import RpiCliCompleter
from rpi_agent.repl.display import (
    display_help,
    display_status,
    display_welcome_banner,
    get_bottom_toolbar,
)
from rpi_agent.repl.turn import execute_agent_turn
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry


async def repl_loop(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    runner: Runner,
    session_service: InMemorySessionService,
) -> None:
    """Main interactive REPL loop using prompt_toolkit and Rich.

    Args:
        config: Application configuration.
        repo_ctx: Repository metadata.
        registry: Skills registry.
        runner: Configured ADK Runner.
        session_service: InMemorySessionService managing conversation turns.
    """
    console = Console()
    display_welcome_banner(console, config, repo_ctx, registry)

    history_file = resolve_history_file_path(repo_ctx.root_path)
    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(history_file)),
        completer=RpiCliCompleter(repo_ctx.root_path, registry),
    )

    current_session_id = f"sess_{uuid.uuid4().hex[:12]}"
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id
    )
    turn_counter = 0

    def toolbar() -> str:
        # Closure reads the live session ID and turn count on every render.
        return get_bottom_toolbar(repo_ctx, config, current_session_id, turn_counter)

    while True:
        try:
            user_input = await session.prompt_async(
                [("class:prompt", f"rpi [{repo_ctx.repo_name}] ❯ ")],
                bottom_toolbar=toolbar,
            )
            raw = user_input.strip()
            if not raw:
                continue

            if raw in {"/exit", "/quit"}:
                console.print("[bold yellow]Exiting RPI Agent. Goodbye![/bold yellow]")
                break

            if raw == "/help":
                display_help(console)
                continue

            if raw == "/skills":
                console.print(Markdown(registry.list_skills()))
                continue

            if raw == "/status":
                display_status(
                    console, config, repo_ctx, current_session_id, turn_counter
                )
                continue

            if raw in {"/clear", "/new"}:
                current_session_id = f"sess_{uuid.uuid4().hex[:12]}"
                await session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=current_session_id
                )
                turn_counter = 0
                console.print(
                    f"[bold green]✓ Session context reset. New Session ID: {current_session_id}[/bold green]"
                )
                continue

            turn_counter += 1
            await execute_agent_turn(
                runner,
                session_service,
                expand_command(raw),
                current_session_id,
                config,
                console,
            )

        except KeyboardInterrupt, EOFError:
            console.print(
                "\n[bold yellow]Session cancelled by user. Exiting.[/bold yellow]"
            )
            break
