"""Session configuration model and resolution from CLI args and environment."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from rpi_agent.constants import DEFAULT_BASE_URL, DEFAULT_MODEL
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

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        skills_dir=skills_dir,
        yolo=args.yolo,
        prune_context=not args.no_prune,
        test_discovery=args.test_discovery,
        test_tools=args.test_tools,
        test_agent=args.test_agent,
    )


def resolve_history_file_path(repo_root: Path) -> Path:
    """Resolves and ensures the repository-isolated history file path exists.

    Args:
        repo_root: Path to repository root.

    Returns:
        Path to the isolated history file at `<repo_root>/.rpi_agent/history`.
    """
    history_dir = repo_root / ".rpi_agent"
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return Path.home() / ".rpi_agent_history"
    return history_dir / "history"
