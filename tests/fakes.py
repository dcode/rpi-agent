"""ADK test doubles."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class FakeEvent:
    """Minimal stand-in for `google.adk.events.Event`.

    `text` is shorthand for a single text part; `parts` gives full control,
    where None models a non-text part (e.g. a function call or thought).
    """

    text: str | None = None
    final: bool = False
    calls: list[Any] = field(default_factory=list)
    responses: list[Any] = field(default_factory=list)
    parts: list[str | None] | None = None

    @property
    def content(self) -> Any:
        if self.parts is not None:
            texts = self.parts
        elif self.text is not None:
            texts = [self.text]
        else:
            return None
        return SimpleNamespace(parts=[SimpleNamespace(text=t) for t in texts])

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


class FakePromptSession:
    """Feeds scripted input lines, then raises the terminal exception."""

    instances: list[FakePromptSession] = []

    def __init__(
        self, inputs: list[str], end: type[BaseException], **kwargs: Any
    ) -> None:
        self.inputs = list(inputs)
        self.end = end
        self.kwargs = kwargs
        self.toolbars: list[str] = []
        FakePromptSession.instances.append(self)

    async def prompt_async(
        self, message: Any, bottom_toolbar: Callable[[], str]
    ) -> str:
        self.toolbars.append(bottom_toolbar())
        if not self.inputs:
            raise self.end
        return self.inputs.pop(0)
