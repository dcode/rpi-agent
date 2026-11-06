"""ADK test doubles."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class FakeEvent:
    """Minimal stand-in for `google.adk.events.Event`."""

    text: str | None = None
    final: bool = False
    calls: list[Any] = field(default_factory=list)
    responses: list[Any] = field(default_factory=list)

    @property
    def content(self) -> Any:
        if self.text is None:
            return None
        return SimpleNamespace(parts=[SimpleNamespace(text=self.text)])

    def get_function_calls(self) -> list[Any]:
        return self.calls

    def get_function_responses(self) -> list[Any]:
        return self.responses

    def is_final_response(self) -> bool:
        return self.final


class FakeRunner:
    """Stand-in for `google.adk.runners.Runner` that replays canned events."""

    def __init__(
        self, events: list[FakeEvent] | None = None, error: Exception | None = None
    ) -> None:
        self.events = events or []
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def run_async(self, **kwargs: Any) -> AsyncIterator[FakeEvent]:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        for event in self.events:
            yield event
