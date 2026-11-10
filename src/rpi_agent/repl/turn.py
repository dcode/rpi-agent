"""Executes a single agent turn with live tool-call rendering."""

from __future__ import annotations

import openai
from google.adk.runners import Runner
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from rpi_agent.constants import USER_ID
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.pruning import ContextPruner
from rpi_agent.telemetry import get_logger, get_tracer

_logger = get_logger("rpi_agent.turn")


async def execute_agent_turn(
    runner: Runner,
    user_input: str,
    session_id: str,
    console: Console,
    pruner: ContextPruner | None = None,
) -> None:
    """Executes a turn through the ADK Runner, displaying live tool calls and responses.

    Args:
        runner: Initialized ADK Runner.
        user_input: User prompt or expanded command text.
        session_id: Active session identifier.
        console: Rich console instance.
        pruner: Request pruner attached to the root agent, for reporting.

    Raises:
        QuotaExhaustedError: Propagated so the REPL can stop and offer a resume.
    """
    if pruner is not None:
        pruner.last_pruned_chars = 0

    content = types.Content(role="user", parts=[types.Part(text=user_input)])
    collected_text: list[str] = []

    status = console.status("[bold cyan]Thinking...", spinner="dots")
    status.start()
    tracer = get_tracer("rpi_agent.turn")

    try:
        with tracer.start_as_current_span(
            "agent.turn",
            attributes={"session.id": session_id, "user.id": USER_ID},
        ):
            _logger.info(
                "turn.start",
                session_id=session_id,
                prompt_chars=len(user_input),
            )
            async for event in runner.run_async(
                user_id=USER_ID, session_id=session_id, new_message=content
            ):
                calls = event.get_function_calls()
                if calls:
                    status.stop()
                    for call in calls:
                        args_repr = ", ".join(
                            f"{k}={repr(v)[:55]}..."
                            if len(repr(v)) > 55
                            else f"{k}={v!r}"
                            for k, v in (call.args or {}).items()
                        )
                        clean_name = escape(call.name or "<unnamed>")
                        clean_args = escape(args_repr)
                        _logger.info(
                            "tool.call",
                            tool=call.name or "<unnamed>",
                            session_id=session_id,
                            args=args_repr,
                        )
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
                    _logger.debug(
                        "tool.response",
                        count=len(responses),
                        session_id=session_id,
                    )
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

            if pruner is not None and pruner.last_pruned_chars > 0:
                _logger.info(
                    "context.pruned",
                    pruned_chars=pruner.last_pruned_chars,
                    session_id=session_id,
                )
                console.print(
                    f"[dim green]🍃 Context Optimized: omitted ~{pruner.last_pruned_chars:,} "
                    "characters of earlier tool output from model requests.[/dim green]"
                )

            _logger.info(
                "turn.completed",
                session_id=session_id,
                response_chars=len(final_response),
            )

    except openai.OpenAIError as err:
        _logger.error(
            "turn.api_error",
            session_id=session_id,
            error_type=type(err).__name__,
            error=str(err),
        )
        console.print(
            f"[bold red]Model API Error:[/bold red] {escape(str(err))}\n"
            "[dim]Retries exhausted. The session is saved; send the prompt again to "
            "retry.[/dim]"
        )
    except (ValueError, KeyError, CodebaseToolError) as err:
        _logger.error(
            "turn.execution_error",
            session_id=session_id,
            error_type=type(err).__name__,
            error=str(err),
        )
        console.print(f"[bold red]Execution Error:[/bold red] {escape(str(err))}")
    finally:
        status.stop()
