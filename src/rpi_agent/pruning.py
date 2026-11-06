"""Context & session pruning engine."""

from __future__ import annotations

from google.adk.sessions import InMemorySessionService

from rpi_agent.constants import APP_NAME, MAX_TOOL_RESULT_PRESERVE_CHARS, USER_ID


async def prune_session_history(
    session_service: InMemorySessionService,
    session_id: str,
) -> int:
    """Prunes bulky tool responses from historical turns in the session.

    Replaces verbose raw file dumps and tool responses in completed turns with
    a lightweight summary stub. Retains the agent's synthesized findings and user
    prompts intact, preventing token compounding on subsequent turns.

    Args:
        session_service: In-memory session store.
        session_id: Target session identifier.

    Returns:
        Approximate number of characters pruned from session history.
    """
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if not session or not hasattr(session, "events"):
        return 0

    pruned_chars = 0
    # Keep the most recent events intact; prune older intermediate tool responses
    events = session.events
    if len(events) <= 2:
        return 0

    # Prune historical events before the last completed turn
    for event in events[:-2]:
        if not hasattr(event, "content") or not event.content:
            continue
        if not hasattr(event.content, "parts") or not event.content.parts:
            continue

        for part in event.content.parts:
            # Check for function responses (tool outputs)
            if hasattr(part, "function_response") and part.function_response:
                resp = part.function_response.response
                if isinstance(resp, dict):
                    output_text = str(resp.get("result", ""))
                    if len(output_text) > MAX_TOOL_RESULT_PRESERVE_CHARS:
                        pruned_chars += (
                            len(output_text) - MAX_TOOL_RESULT_PRESERVE_CHARS
                        )
                        resp["result"] = (
                            f"[Tool output pruned for token optimization: {len(output_text)} chars. "
                            f"Synthesized findings recorded in subsequent response. "
                            f"Re-read with read_file if verification is required.]"
                        )

    return pruned_chars
