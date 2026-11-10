"""Model factory, sub-agent delegation, and root agent construction."""

from __future__ import annotations

import re
import uuid
from typing import Any

import openai
from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from rpi_agent.checkpoints import CheckpointStore
from rpi_agent.config import AppConfig
from rpi_agent.constants import APP_NAME, USER_ID
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.llm import RetryingLiteLlm, RetryNotifier
from rpi_agent.pruning import ContextPruner
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry
from rpi_agent.telemetry import get_logger, get_tracer

_logger = get_logger("rpi_agent.agents")


def get_llm_client(
    config: AppConfig, on_retry: RetryNotifier | None = None
) -> RetryingLiteLlm:
    """Configures and returns the LiteLLM client for the OpenAI-compatible gateway.

    Args:
        config: Application configuration with API key, base URL, and retries.
        on_retry: Optional callback invoked before each retry wait.

    Returns:
        Configured RetryingLiteLlm targeting the OpenAI-compatible gateway.
    """
    model_identifier = f"openai/{config.model_name}"
    return RetryingLiteLlm(
        model=model_identifier,
        max_retries=config.max_retries,
        on_retry=on_retry,
        api_base=config.base_url,
        api_key=config.api_key,
    )


class SubAgentManager:
    """Manages spawning isolated ADK sub-agents for focused tasks and sub-recipes."""

    def __init__(
        self,
        config: AppConfig,
        repo_ctx: RepoContext,
        registry: SkillRegistry,
        read_tools: list[Any],
        checkpoints: CheckpointStore | None = None,
        on_retry: RetryNotifier | None = None,
    ) -> None:
        """Initializes SubAgentManager.

        Args:
            config: Active session application configuration.
            repo_ctx: Live Git repository context.
            registry: Active skills registry.
            read_tools: List of read-only tool functions available to sub-agents.
            checkpoints: Store for saving and reusing completed sub-agent results.
            on_retry: Optional callback invoked before each model retry wait.
        """
        self.config = config
        self.repo_ctx = repo_ctx
        self.registry = registry
        self.read_tools = read_tools
        self.checkpoints = checkpoints
        self.on_retry = on_retry

    async def run_subagent_async(
        self, skill_name: str, task_prompt: str, session_id: str | None = None
    ) -> str:
        """Spawns an isolated ADK agent session to execute a focused skill task.

        Completed results are checkpointed per parent session; an identical
        request in the same session reuses the saved result.

        Args:
            skill_name: Name of the skill to execute (e.g. 'rpi-codebase-locator').
            task_prompt: Specific task or query for the sub-agent to investigate.
            session_id: Parent session ID used to scope checkpoints.

        Returns:
            The synthesized final text output from the isolated sub-agent session.

        Raises:
            QuotaExhaustedError: If the gateway quota is exhausted; propagated so
                the whole run stops cleanly with completed checkpoints saved.
        """
        store = self.checkpoints if session_id else None
        if store is not None and session_id is not None:
            saved = store.load(session_id, skill_name, task_prompt)
            if saved is not None:
                _logger.info(
                    "subagent.checkpoint_reused",
                    skill_name=skill_name,
                    session_id=session_id,
                    created=saved.created,
                )
                return (
                    f"[Reused saved sub-agent result from {saved.created}; "
                    f"same skill and task in this session.]\n\n{saved.result}"
                )

        tracer = get_tracer("rpi_agent.agents")
        with tracer.start_as_current_span(
            "subagent.run",
            attributes={
                "subagent.skill_name": skill_name,
                "session.id": session_id or "",
            },
        ):
            _logger.info(
                "subagent.start",
                skill_name=skill_name,
                session_id=session_id,
            )
            try:
                skill = self.registry.skills.get(skill_name)
                skill_instructions = (
                    skill.instructions
                    if skill
                    else f"Execute this task objectively: {skill_name}"
                )

                # Always a valid identifier: fixed alpha prefix plus [a-zA-Z0-9_] only.
                sanitized_name = re.sub(r"[^a-zA-Z0-9_]", "_", f"subagent_{skill_name}")

                subagent_instruction = (
                    f"You are a specialized sub-agent executing skill '{skill_name}'.\n"
                    f"Repository: {self.repo_ctx.repo_name} | Branch: {self.repo_ctx.branch_name}\n"
                    f"CRITICAL: You are read-only. Answer the task concisely using provided tools.\n\n"
                    f"### Skill Guidance:\n{skill_instructions}"
                )

                client = get_llm_client(self.config, self.on_retry)
                sub_agent = LlmAgent(
                    name=sanitized_name,
                    model=client,
                    instruction=subagent_instruction,
                    tools=self.read_tools,
                )

                session_service = InMemorySessionService()
                sub_session_id = f"sub_{uuid.uuid4().hex[:12]}"
                await session_service.create_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=sub_session_id
                )

                runner = Runner(
                    agent=sub_agent,
                    app_name=APP_NAME,
                    session_service=session_service,
                )

                content = types.Content(
                    role="user", parts=[types.Part(text=task_prompt)]
                )
                collected_response: list[str] = []

                async for event in runner.run_async(
                    user_id=USER_ID,
                    session_id=sub_session_id,
                    new_message=content,
                ):
                    if event.is_final_response():
                        if event.content and event.content.parts:
                            for part in event.content.parts:
                                if part.text:
                                    collected_response.append(part.text)

            except openai.OpenAIError as err:
                _logger.error(
                    "subagent.api_error",
                    skill_name=skill_name,
                    error=str(err),
                )
                # Retries are exhausted by now; let the root agent decide how to proceed.
                return (
                    f"Subagent '{skill_name}' failed with a model API error after "
                    f"retries: {err}"
                )
            except (ValueError, KeyError, CodebaseToolError) as err:
                _logger.error(
                    "subagent.error",
                    skill_name=skill_name,
                    error=str(err),
                )
                return f"Subagent '{skill_name}' execution failed: {err}"

            final_text = "".join(collected_response).strip()
            if not final_text:
                _logger.warning("subagent.empty_output", skill_name=skill_name)
                return f"Subagent for '{skill_name}' completed with no output."

            if store is not None and session_id is not None:
                store.save(session_id, skill_name, task_prompt, final_text)
            _logger.info(
                "subagent.completed",
                skill_name=skill_name,
                output_chars=len(final_text),
            )
            return final_text


def build_root_agent(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    subagent_mgr: SubAgentManager,
    tools: list[Any],
    pruner: ContextPruner | None = None,
    on_retry: RetryNotifier | None = None,
) -> LlmAgent:
    """Constructs the root LlmAgent with injected skills and repository context.

    Args:
        config: Application configuration state.
        repo_ctx: Live Git repository context.
        registry: Loaded skills registry.
        subagent_mgr: Manager for delegating tasks to sub-agents.
        tools: Complete list of tools available to the root agent.
        pruner: Optional request pruner run before every model call.
        on_retry: Optional callback invoked before each model retry wait.

    Returns:
        Configured ADK LlmAgent ready to be bound to Runner.
    """
    client = get_llm_client(config, on_retry)

    async def run_subagent(
        skill_name: str, task_prompt: str, tool_context: ToolContext
    ) -> str:
        """Spawns an isolated read-only sub-agent to execute a sub-skill pass.

        Results are saved per session; repeating the same skill and task reuses
        the saved result instead of re-running the sub-agent.
        """
        return await subagent_mgr.run_subagent_async(
            skill_name, task_prompt, session_id=tool_context.session.id
        )

    all_tools = [*tools, run_subagent]

    system_instruction = (
        "You are an expert autonomous software engineer and codebase researcher running locally.\n"
        "You operate within the root of a Git repository using the RPI (Research-Plan-Iterate-Implement) "
        "methodology and local skills.\n\n"
        f"Repository Name: {repo_ctx.repo_name}\n"
        f"Working Directory: {repo_ctx.root_path}\n"
        f"Active Branch: {repo_ctx.branch_name}\n"
        f"Current Commit: {repo_ctx.commit_sha}\n\n"
        "### Workflow & Operating Directives:\n"
        "1. When exploring large codebases, use 'search_symbols', 'list_symbols' (or 'generate_ctags' "
        "to build an index if none exists), or 'outline_file' first to understand symbol locations "
        "before reading entire files.\n"
        "2. When reading source code, prefer targeted 'read_file' ranges (start_line, end_line) "
        "to verify assumptions without dumping unneeded tokens.\n"
        "3. When asked to research a topic, load the 'rpi-research' skill and follow its workflow.\n"
        "4. Use 'run_subagent' to spawn isolated sub-agents for locator, analyzer, or pattern passes. "
        "Before spawning one, check this conversation for results of earlier sub-agent calls "
        "(including from interrupted turns) and build on them instead of repeating that work.\n"
        "5. Store research artifacts in `./thoughts/research/YYYY-MM-DD-HHmm-<topic>.md`.\n"
        "6. Store planning artifacts in `./thoughts/plans/YYYY-MM-DD-HHmm-<description>.md`.\n"
        "7. Before taking code mutation action on synthesized findings, verify critical assumptions "
        "with targeted reads.\n\n"
        f"{registry.list_skills()}\n"
    )

    return LlmAgent(
        name="rpi_root_agent",
        model=client,
        instruction=system_instruction,
        tools=all_tools,
        before_model_callback=pruner,
    )
