"""Shared fixtures."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from rpi_agent.config import AppConfig
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry
from rpi_agent.tools import CodebaseTools


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A committed Git repository with a small mixed-language source tree."""
    repo = tmp_path / "sample-repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")

    (repo / "pkg").mkdir()
    (repo / "pkg" / "app.py").write_text(
        "import os\n\nclass Service:\n    def run(self):\n        return LlmAgent\n\n"
        "async def main():\n    pass\n",
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# Title\n\n## Usage\n\ntext\n", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.py").write_text("LlmAgent\n", encoding="utf-8")

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture
def repo_ctx(git_repo: Path) -> RepoContext:
    return RepoContext.capture(git_repo)


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    """A skills directory with one resource-bearing skill and one bare skill."""
    root = tmp_path / "skills"
    research = root / "rpi-research"
    (research / "templates").mkdir(parents=True)
    (research / "SKILL.md").write_text(
        "---\nname: rpi-research\ndescription: Research a topic\n---\n\nDo research.\n",
        encoding="utf-8",
    )
    (research / "templates" / "doc.md").write_text("TEMPLATE", encoding="utf-8")

    bare = root / "bare"
    bare.mkdir()
    (bare / "SKILL.md").write_text("Just instructions.\n", encoding="utf-8")
    return root


@pytest.fixture
def registry(skills_dir: Path) -> SkillRegistry:
    reg = SkillRegistry(skills_dir)
    reg.scan()
    return reg


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., AppConfig]:
    def _make(**overrides: Any) -> AppConfig:
        values: dict[str, Any] = {
            "api_key": "test-key",
            "base_url": "http://gateway.test/v1",
            "model_name": "test-model",
            "skills_dir": tmp_path / "skills",
        }
        values.update(overrides)
        return AppConfig(**values)

    return _make


@pytest.fixture
def config(make_config: Callable[..., AppConfig]) -> AppConfig:
    return make_config()


@pytest.fixture
def tools(git_repo: Path) -> CodebaseTools:
    return CodebaseTools(git_repo)
