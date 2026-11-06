"""Line-numbered file reading tool."""

from __future__ import annotations

from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool


class ReadFileTool(RepoScopedTool):
    """Provides the `read_file` tool."""

    def read_file(self, file_path: str, start_line: int = 1, end_line: int = -1) -> str:
        """Reads a file with 1-indexed line numbers and boundary checks.

        Args:
            file_path: Target relative file path.
            start_line: 1-indexed starting line number.
            end_line: 1-indexed ending line number (-1 reads to end).

        Returns:
            Line-numbered text output of the target segment.
        """
        try:
            target = self._resolve_safe_path(file_path)
            if not target.is_file():
                return f"Error: '{file_path}' does not exist or is not a file."

            with target.open("r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            total_lines = len(lines)
            if total_lines == 0:
                return f"File '{file_path}' is empty (0 lines)."

            actual_start = max(1, start_line)
            actual_end = total_lines if end_line == -1 else min(end_line, total_lines)

            if actual_start > total_lines:
                return f"Error: start_line {start_line} exceeds file length ({total_lines})."

            formatted: list[str] = [
                f"# File: {file_path} (Lines {actual_start}-{actual_end} of {total_lines})"
            ]
            for idx in range(actual_start, actual_end + 1):
                formatted.append(f"{idx:5d} | {lines[idx - 1].rstrip()}")

            return "\n".join(formatted)
        except (OSError, CodebaseToolError) as err:
            return f"Error reading file '{file_path}': {err}"
