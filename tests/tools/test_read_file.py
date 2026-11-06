from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.tools import CodebaseTools


def test_reads_whole_file(tools: CodebaseTools) -> None:
    result = tools.read_file("README.md")
    assert result.splitlines() == [
        "# File: README.md (Lines 1-5 of 5)",
        "    1 | # Title",
        "    2 | ",
        "    3 | ## Usage",
        "    4 | ",
        "    5 | text",
    ]


def test_reads_range(tools: CodebaseTools) -> None:
    result = tools.read_file("README.md", start_line=3, end_line=3)
    assert result.splitlines() == [
        "# File: README.md (Lines 3-3 of 5)",
        "    3 | ## Usage",
    ]


def test_clamps_range(tools: CodebaseTools) -> None:
    result = tools.read_file("README.md", start_line=0, end_line=99)
    assert result.startswith("# File: README.md (Lines 1-5 of 5)")


def test_start_beyond_end(tools: CodebaseTools) -> None:
    assert tools.read_file("README.md", start_line=10) == (
        "Error: start_line 10 exceeds file length (5)."
    )


def test_empty_file(tools: CodebaseTools, git_repo: Path) -> None:
    (git_repo / "empty.txt").write_text("")
    assert tools.read_file("empty.txt") == "File 'empty.txt' is empty (0 lines)."


def test_missing_file(tools: CodebaseTools) -> None:
    assert tools.read_file("nope") == "Error: 'nope' does not exist or is not a file."


def test_escape(tools: CodebaseTools) -> None:
    assert tools.read_file("../x").startswith("Error reading file '../x': Security")


def test_os_error(tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("denied")

    monkeypatch.setattr(Path, "open", boom)
    assert tools.read_file("README.md") == "Error reading file 'README.md': denied"
