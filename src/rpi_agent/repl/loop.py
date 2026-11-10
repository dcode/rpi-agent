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
from rpi_agent.config import AppConfig, normalize_log_level
from rpi_agent.constants import PANE_AUTO_OPEN_LOG_LEVELS
from rpi_agent.exceptions import ConfigurationError, QuotaExhaustedError
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
from rpi_agent.repl.log_pane import LogPaneController
from rpi_agent.repl.turn import execute_agent_turn
from rpi_agent.repo import RepoContext
from rpi_agent.sessions import (
    count_user_turns,
    create_session,
    list_sessions,
    load_session,
)
from rpi_agent.skills import SkillRegistry
from rpi_agent.telemetry import SessionLogBuffer, get_active_log_buffer, get_logger

_logger = get_logger("rpi_agent.repl")


def handle_logs_command(
    raw: str,
    console: Console,
    config: AppConfig,
    log_pane: LogPaneController,
) -> None:
    """Processes the `/logs` slash command in the REPL.

    Supported forms:
      - `/logs` or `/logs toggle`: toggle split-screen log pane visibility
      - `/logs show` / `/logs hide`: explicitly show or hide the log pane
      - `/logs <LEVEL>` or `/logs level <LEVEL>`: set log level (shows pane for INFO/DEBUG)
      - `/logs save <PATH> [LEVEL]`: save buffered logs at `LEVEL` (or active level) to disk
      - `/logs clear`: clear the in-memory session log buffer
    """
    arg = raw.removeprefix("/logs").strip()
    tokens = arg.split()
    log_buffer = log_pane.log_buffer

    if not tokens or tokens[0].lower() == "toggle":
        visible = log_pane.toggle()
        config.log_pane = visible
        state_str = "shown" if visible else "hidden"
        console.print(
            f"[bold cyan]Log pane {state_str} (level: {log_buffer.display_level}). "
            "Press F2 to toggle or F3 to focus/scroll.[/bold cyan]"
        )
        return

    action = tokens[0].lower()
    if action == "show":
        log_pane.visible = True
        config.log_pane = True
        console.print(
            f"[bold cyan]Log pane shown (level: {log_buffer.display_level}).[/bold cyan]"
        )
        return

    if action == "hide":
        log_pane.visible = False
        config.log_pane = False
        console.print("[bold cyan]Log pane hidden.[/bold cyan]")
        return

    if action == "clear":
        log_buffer.clear()
        console.print("[bold cyan]✓ Session log buffer cleared.[/bold cyan]")
        return

    if action == "save":
        if len(tokens) < 2:
            console.print(
                "[bold yellow]Usage: /logs save <path> [DEBUG|INFO|WARNING|ERROR|CRITICAL][/bold yellow]"
            )
            return
        save_path = Path(tokens[1])
        try:
            save_level = (
                normalize_log_level(tokens[2])
                if len(tokens) >= 3
                else log_buffer.display_level
            )
            written = log_buffer.save_to_file(save_path, min_level=save_level)
        except (ConfigurationError, OSError) as err:
            console.print(f"[bold red]Failed to save logs:[/bold red] {err}")
            return
        console.print(
            f"[bold green]✓ Saved {written} log entries (level >= {save_level}) "
            f"to {save_path.expanduser().resolve()}.[/bold green]"
        )
        return

    level_token = tokens[1] if action == "level" and len(tokens) >= 2 else tokens[0]
    try:
        new_level = normalize_log_level(level_token)
    except ConfigurationError as err:
        console.print(
            f"[bold red]{err}[/bold red]\n"
            "[dim]Usage: /logs [toggle|show|hide|clear|DEBUG|INFO|WARNING|ERROR|CRITICAL|save <path> [level]][/dim]"
        )
        return

    log_buffer.display_level = new_level
    config.log_level = new_level
    if new_level in PANE_AUTO_OPEN_LOG_LEVELS:
        log_pane.visible = True
        config.log_pane = True
    console.print(f"[bold green]✓ Log level set to {new_level}.[/bold green]")


async def repl_loop(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    runner: Runner,
    session_service: BaseSessionService,
    state_dir: Path,
    checkpoints: CheckpointStore,
    pruner: ContextPruner | None = None,
    log_buffer: SessionLogBuffer | None = None,
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
        log_buffer: Session log buffer for split-screen viewing and saving.

    Raises:
        ConfigurationError: If `config.resume` names a session that does not exist.
    """
    console = Console()
    active_buffer = log_buffer or get_active_log_buffer()
    if active_buffer is None:
        active_buffer = SessionLogBuffer(
            display_level=config.log_level, log_file=config.log_file
        )
    log_pane = LogPaneController(active_buffer, visible=config.log_pane)

    if config.resume:
        resumed = await load_session(session_service, config.resume)
        current_session_id = resumed.id
        turn_counter = count_user_turns(resumed)
        _logger.info(
            "session.resumed",
            session_id=current_session_id,
            turns=turn_counter,
        )
    else:
        current_session_id = await create_session(session_service)
        turn_counter = 0
        _logger.info("session.created", session_id=current_session_id)

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
    log_pane.attach(session)

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
                _logger.info("session.exit", session_id=current_session_id)
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

            if raw == "/logs" or raw.startswith("/logs "):
                handle_logs_command(raw, console, config, log_pane)
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
                _logger.info("session.reset", session_id=current_session_id)
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
