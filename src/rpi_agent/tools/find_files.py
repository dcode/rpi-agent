"""Glob-based file discovery tool."""

from __future__ import annotations

import os
from pathlib import Path

from rpi_agent.constants import IGNORED_DIRECTORIES, MAX_FIND_FILES_DEFAULT
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool


class FindFilesTool(RepoScopedTool):
    """Provides the `find_files` tool."""

    def find_files(self, pattern: str = "*", path: str = ".") -> str:
        """Finds files matching a glob pattern, ignoring vendor and cache directories.

        Args:
            pattern: Glob pattern to evaluate (e.g. '*.py', '*service*').
            path: Relative search start directory. Defaults to repository root.

        Returns:
            Formatted list of matching relative file paths.
        """
        try:
            start_dir = self._resolve_safe_path(path)
            if not start_dir.is_dir():
                return f"Error: '{path}' is not a valid directory."

            matches: list[str] = []
            for root, dirs, files in os.walk(start_dir):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
                for file_name in files:
                    if Path(file_name).match(pattern):
                        full_path = Path(root) / file_name
                        matches.append(full_path.relative_to(self.repo_root).as_posix())

            if not matches:
                return f"No files matching '{pattern}' found under '{path}'."

            matches.sort()
            count = len(matches)
            trimmed = matches[:MAX_FIND_FILES_DEFAULT]
            suffix = (
                f"\n[... and {count - len(trimmed)} more files]"
                if count > len(trimmed)
                else ""
            )
            return (
                f"Found {count} files matching '{pattern}':\n"
                + "\n".join(trimmed)
                + suffix
            )
        except (OSError, CodebaseToolError) as err:
            return f"Error in find_files: {err}"
