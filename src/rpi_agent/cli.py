"""Command-line entry point for the RPI agent."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from google.adk.runners import Runner
from rich.console import Console
from rich.markup import escape

from rpi_agent import __version__ as VERSION
from rpi_agent.agents import SubAgentManager, build_root_agent
from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import resolve_config, resolve_state_dir
from rpi_agent.constants import (
    APP_NAME,
    DEFAULT_BASE_URL,
    DEFAULT_CONFIG_PATH,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_SKILLS_DIR,
    GEMINI_BASE_URL,
)
from rpi_agent.diagnostics import (
    run_agent_diagnostic,
    run_discovery_diagnostic,
    run_tools_diagnostic,
)
from rpi_agent.exceptions import ConfigurationError
from rpi_agent.pruning import ContextPruner
from rpi_agent.repl import repl_loop
from rpi_agent.repl.display import make_retry_notifier
from rpi_agent.repo import RepoContext
from rpi_agent.sessions import LATEST_SESSION, open_session_service
from rpi_agent.skills import SkillRegistry
from rpi_agent.telemetry import setup_logging
from rpi_agent.tools import CodebaseTools, build_tool_sets


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parses command-line arguments for rpi-agent.

    Args:
        argv: Argument list to parse. Defaults to sys.argv[1:].

    Returns:
        Populated argparse.Namespace containing parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        prog="rpi-agent",
        description=f"RPI CLI Agent v{VERSION} using Google ADK and OpenAI-compatible gateways.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--skills-dir",
        type=Path,
        default=DEFAULT_SKILLS_DIR,
        help=f"Directory to scan for skills (default: {DEFAULT_SKILLS_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            f"Model identifier to invoke (default: OPENAI_MODEL -> "
            f"{DEFAULT_MODEL}, or {DEFAULT_GEMINI_MODEL} with --gemini)"
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help=f"Gateway URL (default: OPENAI_BASE_URL -> {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help=f"Use Google Gemini OpenAI-compatible API endpoint ({GEMINI_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gateway API Key (default: OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Bypass interactive user confirmation for shell commands",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="Disable automatic session history pruning (keeps full raw tool outputs in context)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help=(
            "Retries for rate limits and transient gateway errors "
            f"(default: RPI_AGENT_MAX_RETRIES -> {DEFAULT_MAX_RETRIES})"
        ),
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=LATEST_SESSION,
        default=None,
        metavar="SESSION_ID",
        help="Resume a saved session (default: the most recent one)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help=(
            f"Logging verbosity level: DEBUG, INFO, WARNING, ERROR, CRITICAL "
            f"(default: RPI_AGENT_LOG_LEVEL -> {DEFAULT_LOG_LEVEL})"
        ),
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Optional file path to persist structured JSONL logs to disk",
    )
    parser.add_argument(
        "--log-pane",
        action="store_true",
        help="Open the split-screen session log pane in the REPL",
    )
    parser.add_argument(
        "--test-discovery",
        action="store_true",
        help="Run discovery diagnostics across the skills directory and exit",
    )
    parser.add_argument(
        "--test-tools",
        action="store_true",
        help="Run diagnostic checks on codebase and shell tools and exit",
    )
    parser.add_argument(
        "--test-agent",
        action="store_true",
        help="Run diagnostic checks on Agent and SubAgentManager setup and exit",
    )
    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> None:
    """Main application setup and execution entry point."""
    args = parse_arguments(argv)
    config = resolve_config(args)
    setup_logging(log_level=config.log_level, log_file=config.log_file)
    repo_ctx = RepoContext.capture()
    registry = SkillRegistry(config.skills_dir)
    registry.scan()

    if config.test_discovery:
        run_discovery_diagnostic(config, registry)
        return

    if config.test_tools:
        run_tools_diagnostic(repo_ctx.root_path, config.yolo)
        return

    state_dir = resolve_state_dir(repo_ctx.root_path)
    checkpoints = CheckpointStore(state_dir / "subagents")
    on_retry = make_retry_notifier(Console())

    codebase_tools = CodebaseTools(repo_ctx.root_path, yolo=config.yolo)
    read_tools, write_tools = build_tool_sets(codebase_tools, registry)
    all_tools = [*read_tools, *write_tools]
    subagent_mgr = SubAgentManager(
        config, repo_ctx, registry, read_tools, checkpoints, on_retry
    )

    if config.test_agent:
        run_agent_diagnostic(config, repo_ctx, registry, subagent_mgr, all_tools)
        return

    pruner = ContextPruner() if config.prune_context else None
    root_agent = build_root_agent(
        config, repo_ctx, registry, subagent_mgr, all_tools, pruner, on_retry
    )
    session_service = open_session_service(state_dir)
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    try:
        await repl_loop(
            config,
            repo_ctx,
            registry,
            runner,
            session_service,
            state_dir,
            checkpoints,
            pruner,
        )
    finally:
        await session_service.close()


def run() -> None:
    """Synchronous console-script entry point with top-level error reporting."""
    console = Console()
    try:
        asyncio.run(main())
    except ConfigurationError as config_err:
        console.print(
            f"[bold red]Configuration Error:[/bold red] {escape(str(config_err))}"
        )
        sys.exit(1)
    except Exception as fatal_error:  # noqa: BLE001
        console.print(f"[bold red]Fatal Error:[/bold red] {escape(str(fatal_error))}")
        sys.exit(1)


if __name__ == "__main__":
    run()
