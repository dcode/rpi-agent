from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.tools import CodebaseTools


def test_finds_matches_case_insensitive_by_default(tools: CodebaseTools) -> None:
    result = tools.search_code("llmagent")
    assert result == "pkg/app.py:5:         return LlmAgent"


def test_case_sensitive(tools: CodebaseTools) -> None:
    assert tools.search_code("llmagent", case_sensitive=True) == (
        "No occurrences matching query 'llmagent' found."
    )


def test_regex(tools: CodebaseTools) -> None:
    result = tools.search_code(r"^(class|async def) ")
    assert result.splitlines() == [
        "pkg/app.py:3: class Service:",
        "pkg/app.py:7: async def main():",
    ]


def test_scoped_path(tools: CodebaseTools, git_repo: Path) -> None:
    (git_repo / "other.txt").write_text("LlmAgent\n")
    assert "other.txt" not in tools.search_code("LlmAgent", path="pkg")
    assert "other.txt" in tools.search_code("LlmAgent")


def test_skips_binary_extensions(tools: CodebaseTools, git_repo: Path) -> None:
    (git_repo / "image.png").write_text("LlmAgent\n")
    (git_repo / "uv.lock").write_text("LlmAgent\n")
    result = tools.search_code("LlmAgent")
    assert "image.png" not in result
    assert "uv.lock" not in result


def test_truncates_at_max_matches(tools: CodebaseTools, git_repo: Path) -> None:
    (git_repo / "many.txt").write_text("hit\n" * 10)
    result = tools.search_code("hit", max_matches=3)
    lines = result.splitlines()
    assert lines[:3] == ["many.txt:1: hit", "many.txt:2: hit", "many.txt:3: hit"]
    assert "[Truncated after reaching 3 matches." in result


def test_invalid_regex(tools: CodebaseTools) -> None:
    assert tools.search_code("(").startswith("Error in search_code:")


def test_escape(tools: CodebaseTools) -> None:
    assert tools.search_code("x", path="..").startswith(
        "Error in search_code: Security"
    )


def test_unreadable_file_is_skipped(
    tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("denied")

    monkeypatch.setattr(Path, "open", boom)
    assert tools.search_code("LlmAgent") == (
        "No occurrences matching query 'LlmAgent' found."
    )
