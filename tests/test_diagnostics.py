from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from rpi_agent import diagnostics
from rpi_agent.agents import SubAgentManager
from rpi_agent.config import AppConfig
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry


def test_discovery_with_skills(
    make_config: Callable[..., AppConfig],
    skills_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = make_config(skills_dir=skills_dir)
    diagnostics.run_discovery_diagnostic(config, SkillRegistry(skills_dir))
    out = capsys.readouterr().out
    assert "Skill Discovery Diagnostic (2 skills found)" in out
    assert "rpi-research" in out
    assert "Research a topic" in out


def test_discovery_without_skills(
    make_config: Callable[..., AppConfig],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty = tmp_path / "empty"
    diagnostics.run_discovery_diagnostic(
        make_config(skills_dir=empty), SkillRegistry(empty)
    )
    out = capsys.readouterr().out
    assert "(0 skills found)" in out
    assert "No skills found" in out


def test_tools_diagnostic_outside_package_outlines_readme(
    git_repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    diagnostics.run_tools_diagnostic(git_repo, yolo=False)
    out = capsys.readouterr().out
    assert "Repo Metadata Check" in out
    assert "pkg/app.py" in out
    assert "search_code ('LlmAgent')" in out
    assert "outline_file ('README.md')" in out
    assert "# Title" in out


def test_tools_diagnostic_inside_package_outlines_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    project_root = Path(diagnostics.__file__).resolve().parents[2]
    diagnostics.run_tools_diagnostic(project_root, yolo=False)
    out = capsys.readouterr().out
    assert "outline_file ('src/rpi_agent/diagnostics.py')" in out
    assert "run_tools_diagnostic" in out


def test_agent_diagnostic(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager = SubAgentManager(config, repo_ctx, registry, [])
    diagnostics.run_agent_diagnostic(config, repo_ctx, registry, manager, [print])
    out = capsys.readouterr().out
    assert "Agent Name: rpi_root_agent" in out
    assert "Tools Registered: 2" in out
    assert "test-model -> http://gateway.test/v1" in out
