from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.tools import CodebaseTools


def test_write_creates_parents(tools: CodebaseTools, git_repo: Path) -> None:
    result = tools.write_file("new/dir/file.txt", "héllo\nworld\n")
    assert result == "Successfully wrote 13 bytes (2 lines) to 'new/dir/file.txt'."
    assert (git_repo / "new/dir/file.txt").read_text(encoding="utf-8") == (
        "héllo\nworld\n"
    )


def test_write_overwrites(tools: CodebaseTools, git_repo: Path) -> None:
    tools.write_file("README.md", "replaced")
    assert (git_repo / "README.md").read_text() == "replaced"


def test_write_escape(tools: CodebaseTools, tmp_path: Path) -> None:
    result = tools.write_file("../evil.txt", "x")
    assert result.startswith("Error writing file '../evil.txt': Security")
    assert not (tmp_path / "evil.txt").exists()


def test_write_os_error(tools: CodebaseTools) -> None:
    # Writing to an existing directory path fails with IsADirectoryError.
    assert tools.write_file("pkg", "x").startswith("Error writing file 'pkg':")


def test_replace_unique(tools: CodebaseTools, git_repo: Path) -> None:
    assert tools.replace_in_file("README.md", "## Usage", "## How") == (
        "Successfully applied surgical replacement in 'README.md'."
    )
    assert "## How" in (git_repo / "README.md").read_text()


def test_replace_not_found(tools: CodebaseTools) -> None:
    assert tools.replace_in_file("README.md", "zzz", "y") == (
        "Error: old_str not found in 'README.md'. No replacement made."
    )


def test_replace_ambiguous(tools: CodebaseTools, git_repo: Path) -> None:
    before = (git_repo / "README.md").read_text()
    result = tools.replace_in_file("README.md", "#", "*")
    assert result.startswith("Error: old_str matches 3 times in 'README.md'.")
    assert (git_repo / "README.md").read_text() == before


def test_replace_missing_file(tools: CodebaseTools) -> None:
    assert tools.replace_in_file("nope", "a", "b") == (
        "Error: file 'nope' does not exist."
    )


def test_replace_escape(tools: CodebaseTools) -> None:
    assert tools.replace_in_file("../x", "a", "b").startswith(
        "Error replacing in file '../x': Security"
    )


def test_replace_os_error(
    tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only")

    monkeypatch.setattr(Path, "write_text", boom)
    assert tools.replace_in_file("README.md", "## Usage", "x") == (
        "Error replacing in file 'README.md': read-only"
    )


@pytest.mark.xfail(
    reason="Bug: UnicodeDecodeError escapes replace_in_file instead of an error string",
    raises=UnicodeDecodeError,
)
def test_replace_non_utf8_file(tools: CodebaseTools, git_repo: Path) -> None:
    (git_repo / "binary.bin").write_bytes(b"\xff\xfe")
    assert tools.replace_in_file("binary.bin", "a", "b").startswith("Error")
