"""File creation and surgical replacement tools."""

from __future__ import annotations

from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool


class WriteFileTool(RepoScopedTool):
    """Provides the `write_file` and `replace_in_file` tools."""

    def write_file(self, file_path: str, content: str) -> str:
        """Writes or creates a file in the workspace, ensuring parent directories exist.

        Args:
            file_path: Target relative file path.
            content: Raw text content to write.

        Returns:
            Success confirmation string detailing written bytes and lines.
        """
        try:
            target = self._resolve_safe_path(file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            line_count = len(content.splitlines())
            return (
                f"Successfully wrote {len(content.encode('utf-8'))} bytes "
                f"({line_count} lines) to '{file_path}'."
            )
        except (OSError, CodebaseToolError) as err:
            return f"Error writing file '{file_path}': {err}"

    def replace_in_file(self, file_path: str, old_str: str, new_str: str) -> str:
        """Surgically replaces a unique substring within an existing file.

        Args:
            file_path: Target relative file path.
            old_str: Exact string segment to find and replace.
            new_str: Replacement string.

        Returns:
            Confirmation of surgical replacement.
        """
        try:
            target = self._resolve_safe_path(file_path)
            if not target.is_file():
                return f"Error: file '{file_path}' does not exist."

            content = target.read_text(encoding="utf-8")
            occurrences = content.count(old_str)

            if occurrences == 0:
                return (
                    f"Error: old_str not found in '{file_path}'. No replacement made."
                )
            if occurrences > 1:
                return (
                    f"Error: old_str matches {occurrences} times in '{file_path}'. "
                    "Replacement requires unique target context."
                )

            updated = content.replace(old_str, new_str, 1)
            target.write_text(updated, encoding="utf-8")
            return f"Successfully applied surgical replacement in '{file_path}'."
        except (OSError, UnicodeDecodeError, CodebaseToolError) as err:
            return f"Error replacing in file '{file_path}': {err}"
