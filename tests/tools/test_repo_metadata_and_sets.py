from __future__ import annotations

from datetime import datetime
from pathlib import Path

from rpi_agent.skills import SkillRegistry
from rpi_agent.tools import CodebaseTools, build_tool_sets


def test_get_repo_metadata(tools: CodebaseTools, git_repo: Path) -> None:
    lines = dict(line.split(": ", 1) for line in tools.get_repo_metadata().splitlines())
    assert lines["Repository"] == "sample-repo"
    assert lines["Root Path"] == str(git_repo.resolve())
    assert lines["Active Branch"] == "main"
    assert len(lines["Commit SHA"]) == 40
    assert lines["Git Working Tree"] == "True"
    assert datetime.fromisoformat(lines["Timestamp"]).tzinfo is not None


def test_codebase_tools_composes_every_tool(tools: CodebaseTools) -> None:
    for name in (
        "find_files",
        "outline_file",
        "search_symbols",
        "list_symbols",
        "generate_ctags",
        "search_code",
        "read_file",
        "write_file",
        "replace_in_file",
        "get_repo_metadata",
        "is_command_safe",
        "run_shell_command",
    ):
        assert callable(getattr(tools, name))


def test_build_tool_sets_partitions_read_and_write(
    tools: CodebaseTools, registry: SkillRegistry
) -> None:
    read_tools, write_tools = build_tool_sets(tools, registry)
    read_names = [t.__name__ for t in read_tools]
    write_names = [t.__name__ for t in write_tools]

    assert read_names == [
        "find_files",
        "outline_file",
        "search_symbols",
        "list_symbols",
        "search_code",
        "read_file",
        "get_repo_metadata",
        "load_skill_resource",
    ]
    assert write_names == [
        "generate_ctags",
        "write_file",
        "replace_in_file",
        "run_shell_command",
        "load_skill",
        "list_skills",
    ]
    # Sub-agents receive read_tools only; nothing mutating may leak in.
    assert not {
        "generate_ctags",
        "write_file",
        "replace_in_file",
        "run_shell_command",
    } & set(read_names)
