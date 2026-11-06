"""Shared base for repository-scoped tool mixins."""

from __future__ import annotations

from pathlib import Path

from rpi_agent.exceptions import CodebaseToolError


class RepoScopedTool:
    """Holds repository root state and enforces path containment for tools."""

    def __init__(self, repo_root: Path, yolo: bool = False) -> None:
        """Initializes the tool surface.

        Args:
            repo_root: Root path of the current Git repository or workspace.
            yolo: If True, bypasses interactive prompts for shell commands.
        """
        self.repo_root = repo_root.resolve()
        self.yolo = yolo

    def _resolve_safe_path(self, target_rel_or_abs: str) -> Path:
        """Resolves path and enforces containment within repo root.

        Args:
            target_rel_or_abs: Input file path.

        Returns:
            Resolved absolute Path guaranteed inside repo_root.

        Raises:
            CodebaseToolError: If path attempts to escape repository root.
        """
        path = (self.repo_root / target_rel_or_abs).resolve()
        if not path.is_relative_to(self.repo_root):
            raise CodebaseToolError(
                f"Security violation: path '{target_rel_or_abs}' resolves outside repository root."
            )
        return path
