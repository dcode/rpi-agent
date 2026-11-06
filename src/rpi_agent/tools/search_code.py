"""Regex code search tool."""

from __future__ import annotations

import os
import re
from pathlib import Path

from rpi_agent.constants import IGNORED_DIRECTORIES, MAX_SEARCH_MATCHES_DEFAULT
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool


class SearchCodeTool(RepoScopedTool):
    """Provides the `search_code` tool."""

    def search_code(
        self,
        query: str,
        path: str = ".",
        case_sensitive: bool = False,
        max_matches: int = MAX_SEARCH_MATCHES_DEFAULT,
    ) -> str:
        """Searches for regex/string patterns across codebase files.

        Args:
            query: Regex or text query to locate.
            path: Directory to search inside. Defaults to '.'.
            case_sensitive: Whether to match case.
            max_matches: Maximum lines to return before truncating.

        Returns:
            List of matching lines formatted as `file_path:line_number: content`.
        """
        try:
            start_dir = self._resolve_safe_path(path)
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(query, flags)

            results: list[str] = []
            match_count = 0

            for root, dirs, files in os.walk(start_dir):
                dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES]
                for file_name in files:
                    file_path = Path(root) / file_name
                    if file_path.suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".pdf",
                        ".lock",
                        ".bin",
                        ".exe",
                    }:
                        continue

                    try:
                        with file_path.open(
                            "r", encoding="utf-8", errors="ignore"
                        ) as f:
                            for idx, line in enumerate(f, start=1):
                                if pattern.search(line):
                                    rel = file_path.relative_to(
                                        self.repo_root
                                    ).as_posix()
                                    results.append(f"{rel}:{idx}: {line.rstrip()}")
                                    match_count += 1
                                    if match_count >= max_matches:
                                        results.append(
                                            f"\n[Truncated after reaching {max_matches} matches. "
                                            "Narrow your query path or refine regex if needed.]"
                                        )
                                        return "\n".join(results)
                    except OSError, UnicodeDecodeError:
                        continue

            if not results:
                return f"No occurrences matching query '{query}' found."
            return "\n".join(results)
        except (re.error, OSError, CodebaseToolError) as err:
            return f"Error in search_code: {err}"
