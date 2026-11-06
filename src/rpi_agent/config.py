"""Session configuration model and resolution from CLI args and environment."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from rpi_agent.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    STATE_DIR_NAME,
)
from rpi_agent.exceptions import ConfigurationError


@dataclass(slots=True)
class AppConfig:
    """Configuration state for the RPI agent session.

    Attributes:
        api_key: Active API key for the model gateway.
        base_url: Active base URL for the OpenAI-compatible gateway.
        model_name: Target model identifier.
        skills_dir: Directory path containing SKILL.md packages.
        yolo: If True, bypasses interactive confirmation for shell commands.
        prune_context: If True, prunes verbose tool outputs from conversation history.
        max_retries: Retries for transient model API errors (rate limits, 5xx).
        resume: Session ID to resume, "latest", or None to start a new session.
        test_discovery: Runs skill discovery diagnostics and exits.
        test_tools: Runs codebase tool sanity diagnostics and exits.
        test_agent: Runs agent setup diagnostics and exits.
    """

    api_key: str
    base_url: str
    model_name: str
    skills_dir: Path
    yolo: bool = False
    prune_context: bool = True
    max_retries: int = DEFAULT_MAX_RETRIES
    resume: str | None = None
    test_discovery: bool = False
    test_tools: bool = False
    test_agent: bool = False


def resolve_config(args: argparse.Namespace) -> AppConfig:
    """Resolves session configuration using CLI arguments, custom env vars, and defaults.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Validated AppConfig object.

    Raises:
        ConfigurationError: If no valid API key can be resolved.
    """
    api_key = (
        args.api_key
        or os.environ.get("STARK_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise ConfigurationError(
            "Missing API Key. Please provide --api-key, set STARK_API_KEY, "
            "or set OPENAI_API_KEY in your environment."
        )

    base_url = (
        args.base_url
        or os.environ.get("STARK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or DEFAULT_BASE_URL
    )

    model_name = (
        args.model
        or os.environ.get("STARK_MODEL")
        or os.environ.get("OPENAI_MODEL")
        or DEFAULT_MODEL
    )

    skills_dir = args.skills_dir.expanduser().resolve()

    max_retries = args.max_retries
    if max_retries is None:
        raw_retries = os.environ.get("RPI_AGENT_MAX_RETRIES")
        try:
            max_retries = (
                int(raw_retries) if raw_retries is not None else DEFAULT_MAX_RETRIES
            )
        except ValueError as err:
            raise ConfigurationError(
                f"RPI_AGENT_MAX_RETRIES must be an integer, got '{raw_retries}'."
            ) from err
    if max_retries < 0:
        raise ConfigurationError(f"max retries must be >= 0, got {max_retries}.")

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        skills_dir=skills_dir,
        yolo=args.yolo,
        prune_context=not args.no_prune,
        max_retries=max_retries,
        resume=args.resume,
        test_discovery=args.test_discovery,
        test_tools=args.test_tools,
        test_agent=args.test_agent,
    )


def resolve_state_dir(repo_root: Path) -> Path:
    """Resolves and creates the repository-scoped state directory.

    Holds prompt history, the session database, and sub-agent checkpoints.

    Args:
        repo_root: Path to repository root.

    Returns:
        `<repo_root>/.rpi_agent`, or `~/.rpi_agent/<repo name>` if the repository
        is not writable.
    """
    state_dir = repo_root / STATE_DIR_NAME
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        state_dir = Path.home() / STATE_DIR_NAME / repo_root.name
        state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir
