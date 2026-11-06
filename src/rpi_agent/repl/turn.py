"""Executes a single agent turn with live tool-call rendering."""

from __future__ import annotations

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from rpi_agent.config import AppConfig
from rpi_agent.constants import USER_ID
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.pruning import prune_session_history


async def execute_agent_turn(
    runner: Runner,
    session_service: InMemorySessionService,
    user_input: str,
    session_id: str,
    config: AppConfig,
    console: Console,
) -> None:
    """Executes a turn through the ADK Runner, displaying live tool calls and responses.

    Args:
        runner: Initialized ADK Runner.
        session_service: InMemorySessionService instance.
        user_input: User prompt or expanded command text.
        session_id: Active session identifier.
        config: Application configuration.
        console: Rich console instance.
    """
    content = types.Content(role="user", parts=[types.Part(text=user_input)])
    collected_text: list[str] = []

    status = console.status("[bold cyan]Thinking...", spinner="dots")
    status.start()

    try:
        async for event in runner.run_async(
            user_id=USER_ID, session_id=session_id, new_message=content
        ):
            calls = event.get_function_calls()
            if calls:
                status.stop()
                for call in calls:
                    args_repr = ", ".join(
                        f"{k}={repr(v)[:55]}..." if len(repr(v)) > 55 else f"{k}={v!r}"
                        for k, v in (call.args or {}).items()
                    )
                    clean_name = escape(call.name or "<unnamed>")
                    clean_args = escape(args_repr)
                    console.print(
                        Panel(
                            Text.from_markup(
                                f"[bold yellow]⚡ Tool Execution:[/bold yellow] "
                                f"[cyan]{clean_name}[/cyan]({clean_args})"
                            ),
                            border_style="yellow",
                            padding=(0, 1),
                        )
                    )
                status.update("[bold cyan]Executing tool & reasoning...")
                status.start()

            responses = event.get_function_responses()
            if responses:
                status.update("[bold cyan]Processing tool output...")

            if event.is_final_response():
                if event.content and event.content.parts:
                    for part in event.content.parts:
                        if part.text:
                            collected_text.append(part.text)

        final_response = "".join(collected_text).strip()
        if final_response:
            console.print()
            console.print(Markdown(final_response))
            console.print()
        else:
            console.print(
                "[dim italic]Turn completed with no direct response.[/dim italic]"
            )

        # Automatically prune historical raw tool outputs to conserve token budget
        if config.prune_context:
            pruned = await prune_session_history(session_service, session_id)
            if pruned > 0:
                console.print(
                    f"[dim green]🍃 Context Optimized: pruned ~{pruned:,} characters of raw tool output from history.[/dim green]"
                )

    except (ValueError, KeyError, CodebaseToolError) as err:
        console.print(f"[bold red]Execution Error:[/bold red] {escape(str(err))}")
    finally:
        status.stop()
