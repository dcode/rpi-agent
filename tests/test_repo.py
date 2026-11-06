from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from rpi_agent import repo as repo_module
from rpi_agent.repo import RepoContext


def test_capture_git_repo(git_repo: Path) -> None:
    ctx = RepoContext.capture(git_repo)
    assert ctx.is_git is True
    assert ctx.root_path == git_repo.resolve()
    assert ctx.repo_name == "sample-repo"
    assert ctx.branch_name == "main"
    assert len(ctx.commit_sha) == 40


def test_capture_from_subdirectory_finds_root(git_repo: Path) -> None:
    ctx = RepoContext.capture(git_repo / "pkg")
    assert ctx.root_path == git_repo.resolve()


def test_capture_defaults_to_cwd(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(git_repo)
    assert RepoContext.capture().root_path == git_repo.resolve()


def test_capture_detached_head(git_repo: Path) -> None:
    subprocess.run(["git", "checkout", "-q", "--detach"], cwd=git_repo, check=True)
    assert RepoContext.capture(git_repo).branch_name == "[Detached HEAD]"


def test_capture_repo_without_commits(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    ctx = RepoContext.capture(tmp_path)
    assert ctx.is_git is True
    assert ctx.commit_sha == "[Uncommitted]"


def test_capture_non_git_directory(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    # Stop git from discovering any enclosing repository above tmp_path.
    ctx = _capture_isolated(plain)
    assert ctx.is_git is False
    assert ctx.root_path == plain.resolve()
    assert ctx.repo_name == "plain"
    assert ctx.branch_name == "[Non-Git]"
    assert ctx.commit_sha == "[No-Commit]"


def test_capture_when_git_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("git not installed")

    monkeypatch.setattr(repo_module.subprocess, "run", boom)
    ctx = RepoContext.capture(tmp_path)
    assert ctx.is_git is False


def _capture_isolated(path: Path) -> RepoContext:
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GIT_CEILING_DIRECTORIES", str(path.parent))
        return RepoContext.capture(path)
