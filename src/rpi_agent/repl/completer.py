"""Tab-completion for the interactive REPL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from rpi_agent.constants import IGNORED_DIRECTORIES
from rpi_agent.skills import SkillRegistry


class RpiCliCompleter(Completer):
    """Provides tab-completion for slash commands and workspace file paths."""

    def __init__(self, repo_root: Path, registry: SkillRegistry) -> None:
        """Initializes RpiCliCompleter.

        Args:
            repo_root: Root path of the current Git repository.
            registry: Skills registry.
        """
        self.repo_root = repo_root
        self.registry = registry
        self.slash_commands: list[tuple[str, str]] = [
            ("/help", "Show CLI usage and available commands"),
            ("/skills", "List all discovered skills in ~/.agents/skills"),
            ("/status", "Display active repository, model, and gateway configuration"),
            ("/research", "Trigger the rpi-research workflow for a topic"),
            ("/plan", "Trigger the rpi-plan workflow"),
            ("/iterate", "Refine an existing plan with feedback"),
            ("/implement", "Execute an approved implementation plan"),
            ("/clear", "Reset conversational session context"),
            ("/new", "Start a fresh session with clean context"),
            ("/exit", "Exit the RPI agent"),
            ("/quit", "Exit the RPI agent"),
        ]

    def get_completions(self, document: Document, complete_event: Any) -> Any:
        """Yields auto-completions based on current line buffer."""
        text = document.text_before_cursor
        # WORD=True splits on whitespace only, keeping "src/rpi" as one path token.
        word_before_cursor = document.get_word_before_cursor(WORD=True)

        if text.startswith("/"):
            for cmd, desc in self.slash_commands:
                if cmd.startswith(text):
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=f"{cmd:<12} {desc}",
                    )
            return

        if "/" in word_before_cursor or word_before_cursor.startswith("."):
            path_prefix = word_before_cursor
            target_dir = self.repo_root
            search_prefix = path_prefix

            if "/" in path_prefix:
                dir_part, search_prefix = path_prefix.rsplit("/", 1)
                target_dir = (self.repo_root / dir_part).resolve()

            if target_dir.exists() and target_dir.is_dir():
                try:
                    for child in sorted(target_dir.iterdir()):
                        if child.name in IGNORED_DIRECTORIES:
                            continue
                        if child.name.startswith(search_prefix):
                            display_name = (
                                f"{child.name}/" if child.is_dir() else child.name
                            )
                            yield Completion(
                                child.name,
                                start_position=-len(search_prefix),
                                display=display_name,
                            )
                except OSError:
                    pass
