from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.tools import CodebaseTools
from rpi_agent.tools import find_files as find_files_module


def test_finds_matches_and_skips_ignored_dirs(tools: CodebaseTools) -> None:
    result = tools.find_files("*.py")
    assert result == "Found 1 files matching '*.py':\npkg/app.py"


def test_default_pattern_lists_everything(tools: CodebaseTools) -> None:
    result = tools.find_files()
    assert "README.md" in result
    assert "pkg/app.py" in result
    assert ".git/" not in result


def test_scoped_to_subdirectory(tools: CodebaseTools) -> None:
    assert "pkg/app.py" in tools.find_files("*.py", path="pkg")


def test_no_matches(tools: CodebaseTools) -> None:
    assert tools.find_files("*.rs") == "No files matching '*.rs' found under '.'."


def test_path_not_a_directory(tools: CodebaseTools) -> None:
    assert tools.find_files(path="README.md") == (
        "Error: 'README.md' is not a valid directory."
    )


def test_path_escape_is_reported(tools: CodebaseTools) -> None:
    assert tools.find_files(path="..").startswith("Error in find_files: Security")


def test_results_truncated(
    tools: CodebaseTools, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(find_files_module, "MAX_FIND_FILES_DEFAULT", 2)
    for name in ("a", "b", "c"):
        (git_repo / f"{name}.txt").write_text("")

    result = tools.find_files("*.txt")
    assert result.startswith("Found 3 files matching '*.txt':\na.txt\nb.txt")
    assert result.endswith("[... and 1 more files]")
    assert "c.txt" not in result


def test_os_error_is_reported(
    tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(find_files_module.os, "walk", boom)
    assert tools.find_files() == "Error in find_files: disk gone"
