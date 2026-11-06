"""Main interactive REPL loop."""

from __future__ import annotations

from pathlib import Path

from google.adk.runners import Runner
from google.adk.sessions import BaseSessionService
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.markdown import Markdown

from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import AppConfig
from rpi_agent.exceptions import QuotaExhaustedError
from rpi_agent.pruning import ContextPruner
from rpi_agent.repl.commands import expand_command
from rpi_agent.repl.completer import RpiCliCompleter
from rpi_agent.repl.display import (
    display_help,
    display_quota_exhausted,
    display_sessions,
    display_status,
    display_welcome_banner,
    get_bottom_toolbar,
)
from rpi_agent.repl.turn import execute_agent_turn
from rpi_agent.repo import RepoContext
from rpi_agent.sessions import (
    count_user_turns,
    create_session,
    list_sessions,
    load_session,
)
from rpi_agent.skills import SkillRegistry


async def repl_loop(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    runner: Runner,
    session_service: BaseSessionService,
    state_dir: Path,
    checkpoints: CheckpointStore,
    pruner: ContextPruner | None = None,
) -> None:
    """Main interactive REPL loop using prompt_toolkit and Rich.

    Args:
        config: Application configuration.
        repo_ctx: Repository metadata.
        registry: Skills registry.
        runner: Configured ADK Runner.
        session_service: Persistent session store managing conversation turns.
        state_dir: Repository state directory (prompt history lives here).
        checkpoints: Store of saved sub-agent results.
        pruner: Request pruner attached to the root agent, for reporting.

    Raises:
        ConfigurationError: If `config.resume` names a session that does not exist.
    """
    console = Console()

    if config.resume:
        resumed = await load_session(session_service, config.resume)
        current_session_id = resumed.id
        turn_counter = count_user_turns(resumed)
    else:
        current_session_id = await create_session(session_service)
        turn_counter = 0

    display_welcome_banner(console, config, repo_ctx, registry)
    if config.resume:
        saved = len(checkpoints.for_session(current_session_id))
        console.print(
            f"[bold green]✓ Resumed session {current_session_id} "
            f"({turn_counter} prior prompts, {saved} saved sub-agent results).[/bold green]"
        )

    session: PromptSession[str] = PromptSession(
        history=FileHistory(str(state_dir / "history")),
        completer=RpiCliCompleter(repo_ctx.root_path, registry),
    )

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

            if raw == "/sessions":
                display_sessions(
                    console,
                    await list_sessions(session_service),
                    current_session_id,
                    checkpoints,
                )
                continue

            if raw in {"/clear", "/new"}:
                current_session_id = await create_session(session_service)
                turn_counter = 0
                console.print(
                    f"[bold green]✓ Session context reset. New Session ID: {current_session_id}[/bold green]"
                )
                continue

            turn_counter += 1
            try:
                await execute_agent_turn(
                    runner,
                    expand_command(raw),
                    current_session_id,
                    console,
                    pruner,
                )
            except QuotaExhaustedError as err:
                display_quota_exhausted(
                    console,
                    err,
                    current_session_id,
                    len(checkpoints.for_session(current_session_id)),
                )
                break

        except KeyboardInterrupt, EOFError:
            console.print(
                "\n[bold yellow]Session cancelled by user. Exiting.[/bold yellow]"
            )
            break
