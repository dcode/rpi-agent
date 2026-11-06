from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from google.adk.sessions import InMemorySessionService

from rpi_agent.constants import APP_NAME, MAX_TOOL_RESULT_PRESERVE_CHARS, USER_ID
from rpi_agent.pruning import prune_session_history


class FakeSessionService:
    def __init__(self, session: Any) -> None:
        self.session = session
        self.requested: dict[str, str] = {}

    async def get_session(self, **kwargs: str) -> Any:
        self.requested = kwargs
        return self.session


def _prune(session: Any) -> tuple[int, FakeSessionService]:
    service = FakeSessionService(session)
    pruned = asyncio.run(
        prune_session_history(cast(InMemorySessionService, service), "sess")
    )
    return pruned, service


def tool_event(response: Any) -> SimpleNamespace:
    part = SimpleNamespace(function_response=SimpleNamespace(response=response))
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))


def text_event(text: str) -> SimpleNamespace:
    part = SimpleNamespace(text=text, function_response=None)
    return SimpleNamespace(content=SimpleNamespace(parts=[part]))


def test_missing_session() -> None:
    pruned, service = _prune(None)
    assert pruned == 0
    assert service.requested == {
        "app_name": APP_NAME,
        "user_id": USER_ID,
        "session_id": "sess",
    }


def test_session_without_events_attribute() -> None:
    assert _prune(SimpleNamespace())[0] == 0


def test_too_few_events_is_noop() -> None:
    big = {"result": "x" * 10_000}
    session = SimpleNamespace(events=[tool_event(big), tool_event(big)])
    assert _prune(session)[0] == 0
    assert big["result"] == "x" * 10_000


def test_prunes_large_historical_tool_output_only() -> None:
    size = MAX_TOOL_RESULT_PRESERVE_CHARS + 400
    old_big = {"result": "x" * size}
    old_small = {"result": "small"}
    recent_big = {"result": "y" * size}
    session = SimpleNamespace(
        events=[
            tool_event(old_big),
            tool_event(old_small),
            text_event("findings"),
            tool_event(recent_big),
            text_event("final answer"),
        ]
    )

    assert _prune(session)[0] == 400
    assert old_big["result"].startswith(
        f"[Tool output pruned for token optimization: {size} chars."
    )
    assert old_small["result"] == "small"
    assert recent_big["result"] == "y" * size


def test_skips_events_without_content_or_parts_and_non_dict_responses() -> None:
    non_dict = "z" * 5_000
    session = SimpleNamespace(
        events=[
            SimpleNamespace(),
            SimpleNamespace(content=None),
            SimpleNamespace(content=SimpleNamespace()),
            SimpleNamespace(content=SimpleNamespace(parts=[])),
            tool_event(non_dict),
            tool_event({"other": "x" * 5_000}),
            text_event("a"),
            text_event("b"),
        ]
    )
    assert _prune(session)[0] == 0


def test_with_real_session_service() -> None:
    async def scenario() -> int:
        service = InMemorySessionService()
        await service.create_session(
            app_name=APP_NAME, user_id=USER_ID, session_id="real"
        )
        return await prune_session_history(service, "real")

    assert asyncio.run(scenario()) == 0
