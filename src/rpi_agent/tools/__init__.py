"""Codebase & structural tooling surface exposed to ADK agents.

Each tool lives in its own module as a mixin over `RepoScopedTool`; they are
composed here into `CodebaseTools`, whose bound methods are registered with ADK.
"""

from __future__ import annotations

from typing import Any

from rpi_agent.skills import SkillRegistry
from rpi_agent.tools.ctags import CtagsTool
from rpi_agent.tools.find_files import FindFilesTool
from rpi_agent.tools.outline_file import OutlineFileTool
from rpi_agent.tools.read_file import ReadFileTool
from rpi_agent.tools.repo_metadata import RepoMetadataTool
from rpi_agent.tools.search_code import SearchCodeTool
from rpi_agent.tools.shell import ShellTool
from rpi_agent.tools.write_file import WriteFileTool


class CodebaseTools(
    CtagsTool,
    FindFilesTool,
    OutlineFileTool,
    SearchCodeTool,
    ReadFileTool,
    WriteFileTool,
    RepoMetadataTool,
    ShellTool,
):
    """Repository-scoped file operations, structural indexers, and policy-guarded shell executor."""


def build_tool_sets(
    codebase_tools: CodebaseTools, registry: SkillRegistry
) -> tuple[list[Any], list[Any]]:
    """Groups tool callables into read-only and mutating sets.

    Args:
        codebase_tools: Repository-scoped tool surface.
        registry: Active skills registry.

    Returns:
        Tuple of (read_tools, write_tools). Read tools are safe for sub-agents.
    """
    read_tools = [
        codebase_tools.find_files,
        codebase_tools.outline_file,
        codebase_tools.search_symbols,
        codebase_tools.list_symbols,
        codebase_tools.search_code,
        codebase_tools.read_file,
        codebase_tools.get_repo_metadata,
        registry.load_skill_resource,
    ]

    write_tools = [
        codebase_tools.generate_ctags,
        codebase_tools.write_file,
        codebase_tools.replace_in_file,
        codebase_tools.run_shell_command,
        registry.load_skill,
        registry.list_skills,
    ]

    return read_tools, write_tools


__all__ = ["CodebaseTools", "build_tool_sets"]
