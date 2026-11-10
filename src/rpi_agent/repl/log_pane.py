"""Split-screen scrollable log pane for the interactive prompt_toolkit REPL."""

from __future__ import annotations

from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent, merge_key_bindings
from prompt_toolkit.layout.containers import ConditionalContainer, HSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.widgets import Frame

from rpi_agent.telemetry import SessionLogBuffer

DEFAULT_PANE_HEIGHT: int = 10


class LogPaneController:
    """Manages a split-screen, scrollable log pane attached to a PromptSession."""

    def __init__(
        self,
        log_buffer: SessionLogBuffer,
        visible: bool = False,
        height: int = DEFAULT_PANE_HEIGHT,
    ) -> None:
        """Initializes the split-screen log pane controller.

        Args:
            log_buffer: Active session log buffer providing structured log entries.
            visible: Whether the split-screen log pane starts open.
            height: Preferred line height of the log pane window.
        """
        self.log_buffer = log_buffer
        self._visible = visible
        self.height = height
        self._auto_scroll = True
        self._session: Any = None

        self.buffer = Buffer(name="LOG_PANE_BUFFER", read_only=True)
        self.control = BufferControl(buffer=self.buffer, focusable=True)
        self.window = Window(
            content=self.control,
            height=Dimension(min=4, max=max(12, height + 4), preferred=height),
            wrap_lines=True,
            right_margins=[ScrollbarMargin(display_arrows=True)],
        )
        self.frame = Frame(self.window, title=self._get_title)
        self.container = ConditionalContainer(
            self.frame,
            filter=Condition(lambda: self._visible),
        )

        self.log_buffer.add_listener(self.refresh)
        self.refresh()

    @property
    def visible(self) -> bool:
        """Whether the split-screen log pane is currently shown."""
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._visible = bool(value)
        if not self._visible and self.is_focused():
            self.focus_prompt()
        self._invalidate()

    def toggle(self) -> bool:
        """Toggles the split-screen log pane visibility and returns the new state."""
        self.visible = not self._visible
        return self._visible

    def is_focused(self) -> bool:
        """Returns True if the log pane currently has keyboard focus."""
        if self._session is None or getattr(self._session, "app", None) is None:
            return False
        return bool(self._session.app.layout.has_focus(self.window))

    def focus_logs(self) -> bool:
        """Moves keyboard focus to the log pane (opening it if hidden)."""
        if not self._visible:
            self._visible = True
        if (
            self._session is not None
            and getattr(self._session, "app", None) is not None
        ):
            self._session.app.layout.focus(self.window)
            self._invalidate()
            return True
        return False

    def focus_prompt(self) -> bool:
        """Returns keyboard focus to the main REPL input buffer."""
        if (
            self._session is not None
            and getattr(self._session, "app", None) is not None
            and getattr(self._session, "default_buffer", None) is not None
        ):
            self._session.app.layout.focus(self._session.default_buffer)
            self._invalidate()
            return True
        return False

    def toggle_focus(self) -> bool:
        """Toggles keyboard focus between the REPL input and the log pane."""
        if self.is_focused():
            self.focus_prompt()
            return False
        return self.focus_logs()

    def scroll_lines(self, delta: int) -> None:
        """Scrolls the log buffer cursor vertically by `delta` lines.

        Scrolling up disables auto-scroll so the user can inspect earlier logs;
        scrolling back to the last line re-enables auto-scroll.
        """
        doc = self.buffer.document
        target_row = max(0, min(doc.line_count - 1, doc.cursor_position_row + delta))
        new_pos = doc.translate_row_col_to_index(target_row, 0)
        self.buffer.cursor_position = new_pos
        self._auto_scroll = target_row >= doc.line_count - 1
        self._invalidate()

    def scroll_to_top(self) -> None:
        """Scrolls to the oldest log entry in the pane."""
        self.buffer.cursor_position = 0
        self._auto_scroll = False
        self._invalidate()

    def scroll_to_bottom(self) -> None:
        """Scrolls to the newest log entry in the pane and re-enables auto-scroll."""
        text = self.buffer.text
        self.buffer.cursor_position = len(text)
        self._auto_scroll = True
        self._invalidate()

    def refresh(self) -> None:
        """Synchronizes the log pane buffer with the current `SessionLogBuffer` contents."""
        text = self.log_buffer.format_for_display()
        if self._auto_scroll and not self.is_focused():
            cursor_pos = len(text)
        else:
            cursor_pos = min(self.buffer.cursor_position, len(text))
        self.buffer.set_document(
            Document(text=text, cursor_position=cursor_pos),
            bypass_readonly=True,
        )
        self._invalidate()

    def build_key_bindings(self) -> KeyBindings:
        """Creates key bindings for toggling, focusing, and scrolling the log pane."""
        kb = KeyBindings()

        @kb.add("f2")
        def _toggle_pane(_event: KeyPressEvent) -> None:
            self.toggle()

        @kb.add("f3")
        def _toggle_focus(_event: KeyPressEvent) -> None:
            self.toggle_focus()

        pane_focused = Condition(self.is_focused)

        @kb.add("escape", filter=pane_focused)
        @kb.add("q", filter=pane_focused)
        def _return_to_prompt(_event: KeyPressEvent) -> None:
            self.focus_prompt()

        @kb.add("up", filter=pane_focused)
        @kb.add("k", filter=pane_focused)
        def _line_up(_event: KeyPressEvent) -> None:
            self.scroll_lines(-1)

        @kb.add("down", filter=pane_focused)
        @kb.add("j", filter=pane_focused)
        def _line_down(_event: KeyPressEvent) -> None:
            self.scroll_lines(1)

        @kb.add("pageup", filter=pane_focused)
        def _page_up(_event: KeyPressEvent) -> None:
            self.scroll_lines(-max(1, self.height - 1))

        @kb.add("pagedown", filter=pane_focused)
        def _page_down(_event: KeyPressEvent) -> None:
            self.scroll_lines(max(1, self.height - 1))

        @kb.add("home", filter=pane_focused)
        @kb.add("g", filter=pane_focused)
        def _top(_event: KeyPressEvent) -> None:
            self.scroll_to_top()

        @kb.add("end", filter=pane_focused)
        @kb.add("G", filter=pane_focused)
        def _bottom(_event: KeyPressEvent) -> None:
            self.scroll_to_bottom()

        return kb

    def attach(self, session: Any) -> None:
        """Attaches the split-screen log pane and key bindings to a `PromptSession`."""
        self._session = session
        layout = getattr(session, "layout", None)
        if layout is not None and isinstance(
            getattr(layout, "container", None), HSplit
        ):
            children = layout.container.children
            if self.container not in children:
                children.insert(0, self.container)

        kb = self.build_key_bindings()
        existing_kb = getattr(session, "key_bindings", None)
        if existing_kb is not None:
            session.key_bindings = merge_key_bindings([existing_kb, kb])
        elif hasattr(session, "key_bindings"):
            session.key_bindings = kb

    def _get_title(self) -> str:
        focus_hint = (
            "FOCUSED - Up/Dn/PgUp/PgDn: scroll | Esc/F3: prompt"
            if self.is_focused()
            else "F2: hide | F3: focus/scroll"
        )
        count = len(self.log_buffer.filter_entries())
        return f"Session Logs [{self.log_buffer.display_level}] ({count} entries | {focus_hint})"

    def _invalidate(self) -> None:
        if (
            self._session is not None
            and getattr(self._session, "app", None) is not None
        ):
            self._session.app.invalidate()
