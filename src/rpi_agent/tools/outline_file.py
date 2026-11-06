"""Token-conscious structural outlining tool."""

from __future__ import annotations

import re

from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool


class OutlineFileTool(RepoScopedTool):
    """Provides the `outline_file` tool."""

    def outline_file(self, file_path: str) -> str:
        """Extracts high-level structure and declarations from a file without loading raw code.

        Supports:
        - Python: classes, methods, functions (def, async def)
        - Ansible: task names, play hosts, roles, handlers
        - Terraform: resources, data blocks, modules, variables, outputs
        - Shell: functions
        - Markdown: section headers (#, ##, ###)

        Args:
            file_path: Relative path to the file to outline.

        Returns:
            Line-numbered structural skeleton of the file.
        """
        try:
            target = self._resolve_safe_path(file_path)
            if not target.is_file():
                return f"Error: '{file_path}' does not exist or is not a file."

            suffix = target.suffix.lower()

            patterns: list[tuple[str, re.Pattern[str]]] = []

            if suffix == ".py":
                patterns.append(
                    (
                        "Python",
                        re.compile(
                            r"^\s*(?:async\s+def|def|class)\s+([a-zA-Z_][a-zA-Z_0-9]*)"
                        ),
                    )
                )
            elif suffix in {".tf", ".tofu"}:
                patterns.append(
                    (
                        "Terraform",
                        re.compile(
                            r'^(resource|data|module|variable|output)\s+"([^"]+)"(?:\s+"([^"]+)")?'
                        ),
                    )
                )
            elif (
                suffix in {".yml", ".yaml"}
                or "tasks" in file_path
                or "playbook" in file_path
            ):
                patterns.append(
                    (
                        "Ansible/YAML",
                        re.compile(
                            r"^\s*-\s*name:\s*(.+)|^\s*(hosts|tasks|roles|handlers):"
                        ),
                    )
                )
            elif suffix in {".sh", ".bash", ".zsh"}:
                patterns.append(
                    ("Shell", re.compile(r"^(?:function\s+)?([a-zA-Z0-9_-]+)\s*\(\)"))
                )
            elif suffix in {".md", ".markdown"}:
                patterns.append(("Markdown", re.compile(r"^(#{1,4}\s+.+)")))

            # Default generic fallback for other formats
            if not patterns:
                patterns.append(
                    (
                        "Generic",
                        re.compile(
                            r"^(?:class|def|fn|pub fn|func|struct|enum|interface)\s+([a-zA-Z0-9_]+)"
                        ),
                    )
                )

            declarations: list[str] = []
            with target.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    stripped = line.rstrip()
                    for _, regex in patterns:
                        if regex.search(stripped):
                            declarations.append(f"{line_no:5d} | {stripped}")
                            break

            if not declarations:
                # If no declarations matched, return the first 15 lines as preview
                with target.open("r", encoding="utf-8", errors="replace") as f:
                    preview = [
                        f"{i:5d} | {f.readline().rstrip()}" for i in range(1, 16)
                    ]
                return (
                    f"No specific structural declarations identified in '{file_path}'. "
                    f"File header preview (lines 1-15):\n" + "\n".join(preview)
                )

            return (
                f"### Structural Outline of `{file_path}` ({len(declarations)} declarations):\n"
                + "\n".join(declarations)
                + "\n\n(Use `read_file` with start_line and end_line to inspect specific implementation blocks)"
            )
        except (OSError, CodebaseToolError) as err:
            return f"Error outlining file '{file_path}': {err}"
