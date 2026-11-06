from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool


def test_init_resolves_root(tmp_path: Path) -> None:
    tool = RepoScopedTool(tmp_path / "." / "", yolo=True)
    assert tool.repo_root == tmp_path.resolve()
    assert tool.yolo is True


def test_resolve_relative_path(tmp_path: Path) -> None:
    tool = RepoScopedTool(tmp_path)
    assert tool._resolve_safe_path("a/b.txt") == tmp_path.resolve() / "a" / "b.txt"


def test_resolve_absolute_path_inside_root(tmp_path: Path) -> None:
    tool = RepoScopedTool(tmp_path)
    inside = tmp_path / "x.txt"
    assert tool._resolve_safe_path(str(inside)) == inside.resolve()


@pytest.mark.parametrize("escape", ["../outside", "/etc/passwd", "a/../../b"])
def test_resolve_rejects_escape(tmp_path: Path, escape: str) -> None:
    tool = RepoScopedTool(tmp_path / "root")
    with pytest.raises(CodebaseToolError, match="Security violation"):
        tool._resolve_safe_path(escape)


def test_resolve_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(tmp_path)
    with pytest.raises(CodebaseToolError):
        RepoScopedTool(root)._resolve_safe_path("link/secret")
