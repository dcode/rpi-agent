from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import Any, cast

import pytest
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm

from rpi_agent import agents as agents_module
from rpi_agent.agents import SubAgentManager, build_root_agent, get_llm_client
from rpi_agent.config import AppConfig
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry
from tests.fakes import FakeEvent, FakeRunner


def read_tool(path: str) -> str:
    """A read-only stand-in tool."""
    return path


@pytest.fixture
def manager(
    config: AppConfig, repo_ctx: RepoContext, registry: SkillRegistry
) -> SubAgentManager:
    return SubAgentManager(config, repo_ctx, registry, [read_tool])


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replaces Runner with a FakeRunner and records the agent it was given."""
    state: dict[str, Any] = {"events": [], "error": None}

    def fake_runner(**kwargs: Any) -> FakeRunner:
        state["agent"] = kwargs["agent"]
        state["runner"] = FakeRunner(state["events"], state["error"])
        return state["runner"]

    monkeypatch.setattr(agents_module, "Runner", fake_runner)
    return state


def test_get_llm_client(config: AppConfig) -> None:
    client = get_llm_client(config)
    assert isinstance(client, LiteLlm)
    assert client.model == "openai/test-model"


def test_subagent_returns_final_text(
    manager: SubAgentManager, captured: dict[str, Any]
) -> None:
    captured["events"] += [
        FakeEvent(text="ignored, not final"),
        FakeEvent(final=True),
        FakeEvent(text="Found ", final=True),
        FakeEvent(text="it.", final=True),
    ]
    result = asyncio.run(manager.run_subagent_async("rpi-research", "where?"))
    assert result == "Found it."

    agent: LlmAgent = captured["agent"]
    assert agent.name == "subagent_rpi_research"
    assert agent.tools == [read_tool]
    assert isinstance(agent.instruction, str)
    assert "Do research." in agent.instruction
    assert "CRITICAL: You are read-only." in agent.instruction
    assert "Branch: main" in agent.instruction

    call = captured["runner"].calls[0]
    assert call["session_id"].startswith("sub_")
    assert call["new_message"].parts[0].text == "where?"


def test_subagent_unknown_skill_uses_generic_instruction(
    manager: SubAgentManager, captured: dict[str, Any]
) -> None:
    asyncio.run(manager.run_subagent_async("unknown", "task"))
    assert "Execute this task objectively: unknown" in captured["agent"].instruction


@pytest.mark.parametrize(
    ("skill_name", "agent_name"),
    [("ok-name", "subagent_ok_name"), ("a.b/c d", "subagent_a_b_c_d")],
)
def test_subagent_name_is_sanitised(
    manager: SubAgentManager,
    captured: dict[str, Any],
    skill_name: str,
    agent_name: str,
) -> None:
    asyncio.run(manager.run_subagent_async(skill_name, "task"))
    assert captured["agent"].name == agent_name


def test_subagent_no_output(manager: SubAgentManager, captured: dict[str, Any]) -> None:
    assert asyncio.run(manager.run_subagent_async("bare", "task")) == (
        "Subagent for 'bare' completed with no output."
    )


@pytest.mark.parametrize(
    "error", [ValueError("bad"), KeyError("k"), CodebaseToolError("tool")]
)
def test_subagent_handled_errors(
    manager: SubAgentManager, captured: dict[str, Any], error: Exception
) -> None:
    captured["error"] = error
    result = asyncio.run(manager.run_subagent_async("bare", "task"))
    assert result.startswith("Subagent 'bare' execution failed:")


def test_build_root_agent(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    manager: SubAgentManager,
) -> None:
    agent = build_root_agent(config, repo_ctx, registry, manager, [read_tool])
    assert agent.name == "rpi_root_agent"
    assert isinstance(agent.model, LiteLlm)

    names = [getattr(t, "__name__", None) for t in agent.tools]
    assert names == ["read_tool", "run_subagent"]

    instruction = agent.instruction
    assert isinstance(instruction, str)
    assert f"Working Directory: {repo_ctx.root_path}" in instruction
    assert f"Current Commit: {repo_ctx.commit_sha}" in instruction
    assert "### Available Skills Catalog" in instruction
    assert "rpi-research" in instruction


def test_root_agent_run_subagent_delegates(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    manager: SubAgentManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []

    async def fake_run(skill_name: str, task_prompt: str) -> str:
        seen.append((skill_name, task_prompt))
        return "delegated"

    monkeypatch.setattr(manager, "run_subagent_async", fake_run)
    agent = build_root_agent(config, repo_ctx, registry, manager, [])
    run_subagent = cast(Callable[[str, str], Coroutine[Any, Any, str]], agent.tools[-1])

    assert asyncio.run(run_subagent("skill", "prompt")) == "delegated"
    assert seen == [("skill", "prompt")]
