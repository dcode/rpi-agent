from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, Window

from rpi_agent.repl.log_pane import LogPaneController
from rpi_agent.telemetry import SessionLogBuffer


class FakeLayout:
    def __init__(self, container: Any) -> None:
        self.container = container
        self.focused: Any = None

    def has_focus(self, target: Any) -> bool:
        return self.focused is target

    def focus(self, target: Any) -> None:
        self.focused = target


class FakeApp:
    def __init__(self, layout: FakeLayout) -> None:
        self.layout = layout
        self.invalidated = 0

    def invalidate(self) -> None:
        self.invalidated += 1


class FakeSession:
    def __init__(self, with_kb: bool = False) -> None:
        self.default_buffer = object()
        self.layout = FakeLayout(HSplit([Window()]))
        self.app = FakeApp(self.layout)
        self.key_bindings: Any = KeyBindings() if with_kb else None


def emit_lines(buf: SessionLogBuffer, count: int) -> None:
    for i in range(count):
        buf.emit(
            logging.LogRecord("app", logging.INFO, "", i + 1, f"line {i}", (), None)
        )


def test_log_pane_visibility_and_focus_without_session() -> None:
    buf = SessionLogBuffer(display_level="INFO")
    pane = LogPaneController(buf, visible=False, height=6)

    assert pane.visible is False
    assert pane.is_focused() is False
    assert pane.focus_logs() is False
    assert pane.visible is True
    assert pane.focus_prompt() is False
    assert pane.toggle() is False


def test_log_pane_attach_focus_and_scrolling() -> None:
    buf = SessionLogBuffer(display_level="INFO")
    pane = LogPaneController(buf, visible=False, height=5)
    session = FakeSession(with_kb=True)

    pane.attach(session)
    # Attaching again does not duplicate the container
    pane.attach(session)
    assert session.layout.container.children[0] is pane.container

    # Also test attach when session has no prior key_bindings
    session_no_kb = FakeSession(with_kb=False)
    pane.attach(session_no_kb)
    assert session_no_kb.key_bindings is not None

    # Emit 10 log lines
    emit_lines(buf, 10)
    assert "line 9" in pane.buffer.text
    assert pane.buffer.document.cursor_position_row == 9

    # Focus the log pane via toggle_focus()
    assert pane.toggle_focus() is True
    assert pane.visible is True
    assert pane.is_focused() is True
    assert "FOCUSED" in pane._get_title()

    # While focused, new log entries do not jump the cursor
    pane.scroll_to_top()
    assert pane.buffer.document.cursor_position_row == 0
    emit_lines(buf, 1)
    assert pane.buffer.document.cursor_position_row == 0

    # Scroll down and up
    pane.scroll_lines(3)
    assert pane.buffer.document.cursor_position_row == 3
    pane.scroll_lines(-1)
    assert pane.buffer.document.cursor_position_row == 2

    # Scroll to bottom re-enables auto-scroll
    pane.scroll_to_bottom()
    assert (
        pane.buffer.document.cursor_position_row == pane.buffer.document.line_count - 1
    )

    # Toggle focus back to prompt
    assert pane.toggle_focus() is False
    assert pane.is_focused() is False
    assert "F2: hide" in pane._get_title()

    # If focused and hidden via visible = False, focus returns to prompt
    pane.focus_logs()
    assert pane.is_focused() is True
    pane.visible = False
    assert pane.is_focused() is False


def test_log_pane_key_bindings_handlers() -> None:
    buf = SessionLogBuffer(display_level="INFO")
    emit_lines(buf, 20)
    pane = LogPaneController(buf, visible=True, height=5)
    session = FakeSession()
    pane.attach(session)

    kb = pane.build_key_bindings()
    by_key = {b.keys[0]: b for b in kb.bindings}
    dummy_event: Any = SimpleNamespace()

    # F2 toggles visibility
    by_key["f2"].handler(dummy_event)
    assert pane.visible is False
    by_key["f2"].handler(dummy_event)
    assert pane.visible is True

    # F3 focuses logs
    by_key["f3"].handler(dummy_event)
    assert pane.is_focused() is True

    # Home / g scrolls to top
    by_key["home"].handler(dummy_event)
    assert pane.buffer.document.cursor_position_row == 0

    # Down / j scrolls down 1
    by_key["down"].handler(dummy_event)
    assert pane.buffer.document.cursor_position_row == 1

    # PageDown scrolls down by height - 1 (4)
    by_key["pagedown"].handler(dummy_event)
    assert pane.buffer.document.cursor_position_row == 5

    # PageUp scrolls up by height - 1 (4)
    by_key["pageup"].handler(dummy_event)
    assert pane.buffer.document.cursor_position_row == 1

    # Up / k scrolls up 1
    by_key["up"].handler(dummy_event)
    assert pane.buffer.document.cursor_position_row == 0

    # End / G scrolls to bottom
    by_key["end"].handler(dummy_event)
    assert (
        pane.buffer.document.cursor_position_row == pane.buffer.document.line_count - 1
    )

    # Escape returns focus to prompt
    by_key["escape"].handler(dummy_event)
    assert pane.is_focused() is False
