"""Model factory, sub-agent delegation, and root agent construction."""

from __future__ import annotations

import re
import uuid
from typing import Any

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from rpi_agent.config import AppConfig
from rpi_agent.constants import APP_NAME, USER_ID
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.repo import RepoContext
from rpi_agent.skills import SkillRegistry


def get_llm_client(config: AppConfig) -> LiteLlm:
    """Configures and returns the LiteLLM client for the GenAI.mil gateway.

    Args:
        config: Application configuration with API key and base URL.

    Returns:
        Configured LiteLlm instance targeting the OpenAI-compatible gateway.
    """
    model_identifier = f"openai/{config.model_name}"
    return LiteLlm(
        model=model_identifier,
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
    ) -> None:
        """Initializes SubAgentManager.

        Args:
            config: Active session application configuration.
            repo_ctx: Live Git repository context.
            registry: Active skills registry.
            read_tools: List of read-only tool functions available to sub-agents.
        """
        self.config = config
        self.repo_ctx = repo_ctx
        self.registry = registry
        self.read_tools = read_tools

    async def run_subagent_async(self, skill_name: str, task_prompt: str) -> str:
        """Spawns an isolated ADK agent session to execute a focused skill task.

        Args:
            skill_name: Name of the skill to execute (e.g. 'rpi-codebase-locator').
            task_prompt: Specific task or query for the sub-agent to investigate.

        Returns:
            The synthesized final text output from the isolated sub-agent session.
        """
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

            client = get_llm_client(self.config)
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

            content = types.Content(role="user", parts=[types.Part(text=task_prompt)])
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

            final_text = "".join(collected_response).strip()
            return (
                final_text
                if final_text
                else f"Subagent for '{skill_name}' completed with no output."
            )
        except (ValueError, KeyError, CodebaseToolError) as err:
            return f"Subagent '{skill_name}' execution failed: {err}"


def build_root_agent(
    config: AppConfig,
    repo_ctx: RepoContext,
    registry: SkillRegistry,
    subagent_mgr: SubAgentManager,
    tools: list[Any],
) -> LlmAgent:
    """Constructs the root LlmAgent with injected skills and repository context.

    Args:
        config: Application configuration state.
        repo_ctx: Live Git repository context.
        registry: Loaded skills registry.
        subagent_mgr: Manager for delegating tasks to sub-agents.
        tools: Complete list of tools available to the root agent.

    Returns:
        Configured ADK LlmAgent ready to be bound to Runner.
    """
    client = get_llm_client(config)

    async def run_subagent(skill_name: str, task_prompt: str) -> str:
        """Spawns an isolated read-only sub-agent to execute a sub-skill pass."""
        return await subagent_mgr.run_subagent_async(skill_name, task_prompt)

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
        "1. When exploring large codebases, use 'outline_file' first to understand class, function, "
        "Ansible task, or Terraform resource locations before reading entire files.\n"
        "2. When reading source code, prefer targeted 'read_file' ranges (start_line, end_line) "
        "to verify assumptions without dumping unneeded tokens.\n"
        "3. When asked to research a topic, load the 'rpi-research' skill and follow its workflow.\n"
        "4. Use 'run_subagent' to spawn isolated sub-agents for locator, analyzer, or pattern passes.\n"
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
    )
