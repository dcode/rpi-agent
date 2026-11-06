from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import litellm
import pytest
from google.adk.agents import LlmAgent

from rpi_agent import agents as agents_module
from rpi_agent.agents import SubAgentManager, build_root_agent, get_llm_client
from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import AppConfig
from rpi_agent.exceptions import CodebaseToolError, QuotaExhaustedError
from rpi_agent.llm import RetryingLiteLlm
from rpi_agent.pruning import ContextPruner
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


def test_get_llm_client(make_config: Callable[..., AppConfig]) -> None:
    def notify(*_args: object) -> None:
        pass

    client = get_llm_client(make_config(max_retries=7), notify)
    assert isinstance(client, RetryingLiteLlm)
    assert client.model == "openai/test-model"
    assert client.max_retries == 7
    assert client._on_retry is notify
    assert client._additional_args["api_base"] == "http://gateway.test/v1"


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


def test_subagent_skips_non_text_parts(
    manager: SubAgentManager, captured: dict[str, Any]
) -> None:
    captured["events"] += [FakeEvent(parts=["Found ", None, "", "it."], final=True)]
    assert asyncio.run(manager.run_subagent_async("bare", "task")) == "Found it."


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
    assert isinstance(agent.model, RetryingLiteLlm)
    assert agent.before_model_callback is None

    names = [getattr(t, "__name__", None) for t in agent.tools]
    assert names == ["read_tool", "run_subagent"]

    instruction = agent.instruction
    assert isinstance(instruction, str)
    assert f"Working Directory: {repo_ctx.root_path}" in instruction
    assert f"Current Commit: {repo_ctx.commit_sha}" in instruction
    assert "### Available Skills Catalog" in instruction
    assert "rpi-research" in instruction
    assert "including from interrupted turns" in instruction


def test_build_root_agent_attaches_pruner(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    manager: SubAgentManager,
) -> None:
    pruner = ContextPruner()
    agent = build_root_agent(config, repo_ctx, registry, manager, [], pruner)
    assert agent.before_model_callback is pruner


def test_root_agent_run_subagent_delegates_with_session_id(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    manager: SubAgentManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str, str | None]] = []

    async def fake_run(
        skill_name: str, task_prompt: str, session_id: str | None = None
    ) -> str:
        seen.append((skill_name, task_prompt, session_id))
        return "delegated"

    monkeypatch.setattr(manager, "run_subagent_async", fake_run)
    agent = build_root_agent(config, repo_ctx, registry, manager, [])
    run_subagent = cast(
        Callable[[str, str, Any], Coroutine[Any, Any, str]], agent.tools[-1]
    )
    tool_context = SimpleNamespace(session=SimpleNamespace(id="sess_root"))

    assert asyncio.run(run_subagent("skill", "prompt", tool_context)) == "delegated"
    assert seen == [("skill", "prompt", "sess_root")]


# ------------------------------------------------------------------------------
# Failure handling and checkpoints
# ------------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "subagents")


@pytest.fixture
def checkpointed(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    store: CheckpointStore,
) -> SubAgentManager:
    return SubAgentManager(config, repo_ctx, registry, [read_tool], store)


def test_subagent_api_error_returned_as_text(
    manager: SubAgentManager, captured: dict[str, Any]
) -> None:
    captured["error"] = litellm.RateLimitError(
        "TPM limit", llm_provider="openai", model="m"
    )
    result = asyncio.run(manager.run_subagent_async("bare", "task"))
    assert result.startswith(
        "Subagent 'bare' failed with a model API error after retries:"
    )
    assert "TPM limit" in result


def test_subagent_quota_exhaustion_propagates(
    checkpointed: SubAgentManager, captured: dict[str, Any], store: CheckpointStore
) -> None:
    captured["error"] = QuotaExhaustedError("daily", reset_hint="1h")
    with pytest.raises(QuotaExhaustedError):
        asyncio.run(checkpointed.run_subagent_async("bare", "task", "sess_1"))
    assert store.for_session("sess_1") == []


def test_successful_result_is_checkpointed(
    checkpointed: SubAgentManager, captured: dict[str, Any], store: CheckpointStore
) -> None:
    captured["events"] += [FakeEvent(text="found it", final=True)]
    assert (
        asyncio.run(checkpointed.run_subagent_async("locator", "find auth", "sess_1"))
        == "found it"
    )

    (saved,) = store.for_session("sess_1")
    assert (saved.skill_name, saved.task_prompt, saved.result) == (
        "locator",
        "find auth",
        "found it",
    )


def test_checkpointed_result_is_reused_without_running(
    checkpointed: SubAgentManager, captured: dict[str, Any], store: CheckpointStore
) -> None:
    saved = store.save("sess_1", "locator", "find auth", "cached findings")
    result = asyncio.run(
        checkpointed.run_subagent_async("locator", "find auth", "sess_1")
    )
    assert result == (
        f"[Reused saved sub-agent result from {saved.created}; same skill and task "
        "in this session.]\n\ncached findings"
    )
    assert "agent" not in captured  # no sub-agent was constructed


def test_different_task_is_not_reused(
    checkpointed: SubAgentManager, captured: dict[str, Any], store: CheckpointStore
) -> None:
    store.save("sess_1", "locator", "find auth", "cached")
    captured["events"] += [FakeEvent(text="fresh", final=True)]
    assert (
        asyncio.run(
            checkpointed.run_subagent_async("locator", "find billing", "sess_1")
        )
        == "fresh"
    )


def test_empty_or_failed_results_are_not_checkpointed(
    checkpointed: SubAgentManager, captured: dict[str, Any], store: CheckpointStore
) -> None:
    asyncio.run(checkpointed.run_subagent_async("locator", "empty", "sess_1"))
    captured["error"] = ValueError("bad")
    asyncio.run(checkpointed.run_subagent_async("locator", "fails", "sess_1"))
    assert store.for_session("sess_1") == []


def test_no_checkpointing_without_session_id(
    checkpointed: SubAgentManager, captured: dict[str, Any], store: CheckpointStore
) -> None:
    captured["events"] += [FakeEvent(text="result", final=True)]
    asyncio.run(checkpointed.run_subagent_async("locator", "task"))
    assert not store.root.exists()


def test_subagent_uses_retry_notifier(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    captured: dict[str, Any],
) -> None:
    def notify(*_args: object) -> None:
        pass

    mgr = SubAgentManager(config, repo_ctx, registry, [], on_retry=notify)
    asyncio.run(mgr.run_subagent_async("bare", "task"))
    assert captured["agent"].model._on_retry is notify
