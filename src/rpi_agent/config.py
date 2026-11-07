"""Session configuration model and resolution from CLI args, env, and config file."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rpi_agent.constants import (
    DEFAULT_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_SKILLS_DIR,
    GEMINI_BASE_URL,
    STATE_DIR_NAME,
)
from rpi_agent.exceptions import ConfigurationError

_STRING_CONFIG_KEYS: frozenset[str] = frozenset(
    {"api_key", "base_url", "model", "skills_dir"}
)
_BOOL_CONFIG_KEYS: frozenset[str] = frozenset(
    {"gemini", "yolo", "no_prune", "prune_context"}
)
_INT_CONFIG_KEYS: frozenset[str] = frozenset({"max_retries"})
_ALLOWED_CONFIG_KEYS: frozenset[str] = (
    _STRING_CONFIG_KEYS | _BOOL_CONFIG_KEYS | _INT_CONFIG_KEYS
)


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


def load_user_config(config_path: Path | None = None) -> dict[str, Any]:
    """Loads and validates user configuration from a YAML file.

    When `config_path` is None, looks for `~/.config/rpi-agent.yml` and returns
    an empty dict if it does not exist. When `config_path` is explicitly given,
    raises `ConfigurationError` if the file is missing.

    Args:
        config_path: Explicit path to a YAML config file, or None for default.

    Returns:
        Normalized mapping of validated configuration options.

    Raises:
        ConfigurationError: If the file cannot be read, is not a valid YAML
            mapping, or contains unknown or invalidly typed options.
    """
    if config_path is not None:
        target = config_path.expanduser()
        if not target.is_file():
            raise ConfigurationError(f"Configuration file not found: {target}")
    else:
        target = Path.home() / ".config" / "rpi-agent.yml"
        if not target.is_file():
            return {}

    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as err:
        raise ConfigurationError(
            f"Failed to load configuration file {target}: {err}"
        ) from err

    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigurationError(
            f"Configuration file {target} must contain a YAML mapping (key: value pairs)."
        )

    normalized: dict[str, Any] = {}
    for raw_key, value in raw.items():
        if not isinstance(raw_key, str):
            raise ConfigurationError(
                f"Configuration keys in {target} must be strings, got {raw_key!r}."
            )
        key = raw_key.strip().replace("-", "_")
        if key == "model_name":
            key = "model"
        if key not in _ALLOWED_CONFIG_KEYS:
            raise ConfigurationError(
                f"Unknown configuration option '{raw_key}' in {target}."
            )
        if key in _STRING_CONFIG_KEYS:
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(
                    f"Configuration option '{raw_key}' in {target} must be a non-empty string."
                )
            normalized[key] = value.strip()
        elif key in _BOOL_CONFIG_KEYS:
            if not isinstance(value, bool):
                raise ConfigurationError(
                    f"Configuration option '{raw_key}' in {target} must be true or false."
                )
            normalized[key] = value
        else:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConfigurationError(
                    f"Configuration option '{raw_key}' in {target} must be an integer."
                )
            normalized[key] = value

    return normalized


def resolve_config(args: argparse.Namespace) -> AppConfig:
    """Resolves session configuration from CLI args, environment, and user config file.

    Precedence order: CLI arguments > environment variables > `~/.config/rpi-agent.yml`
    > built-in defaults.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Validated AppConfig object.

    Raises:
        ConfigurationError: If no valid API key can be resolved or any option is invalid.
    """
    file_config = load_user_config(getattr(args, "config", None))

    api_key = (
        args.api_key or os.environ.get("OPENAI_API_KEY") or file_config.get("api_key")
    )
    if not api_key:
        raise ConfigurationError(
            "Missing API Key. Please provide --api-key, set OPENAI_API_KEY "
            "in your environment, or set api_key in ~/.config/rpi-agent.yml."
        )

    cli_gemini = bool(getattr(args, "gemini", False))
    cfg_gemini = bool(file_config.get("gemini", False))
    use_gemini = cli_gemini or cfg_gemini

    if args.base_url:
        base_url = args.base_url
    elif cli_gemini:
        base_url = GEMINI_BASE_URL
    elif env_base_url := os.environ.get("OPENAI_BASE_URL"):
        base_url = env_base_url
    elif cfg_base_url := file_config.get("base_url"):
        base_url = cfg_base_url
    elif cfg_gemini:
        base_url = GEMINI_BASE_URL
    else:
        base_url = DEFAULT_BASE_URL

    default_model = DEFAULT_GEMINI_MODEL if use_gemini else DEFAULT_MODEL
    model_name = (
        args.model
        or os.environ.get("OPENAI_MODEL")
        or file_config.get("model")
        or default_model
    )

    if args.skills_dir == DEFAULT_SKILLS_DIR and "skills_dir" in file_config:
        skills_dir = Path(file_config["skills_dir"]).expanduser().resolve()
    else:
        skills_dir = args.skills_dir.expanduser().resolve()

    yolo = bool(args.yolo or file_config.get("yolo", False))

    if args.no_prune:
        prune_context = False
    elif "prune_context" in file_config:
        prune_context = bool(file_config["prune_context"])
    elif "no_prune" in file_config:
        prune_context = not bool(file_config["no_prune"])
    else:
        prune_context = True

    max_retries = args.max_retries
    if max_retries is None:
        raw_retries = os.environ.get("RPI_AGENT_MAX_RETRIES")
        if raw_retries is not None:
            try:
                max_retries = int(raw_retries)
            except ValueError as err:
                raise ConfigurationError(
                    f"RPI_AGENT_MAX_RETRIES must be an integer, got '{raw_retries}'."
                ) from err
        elif "max_retries" in file_config:
            max_retries = file_config["max_retries"]
        else:
            max_retries = DEFAULT_MAX_RETRIES
    if max_retries < 0:
        raise ConfigurationError(f"max retries must be >= 0, got {max_retries}.")

    return AppConfig(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        skills_dir=skills_dir,
        yolo=yolo,
        prune_context=prune_context,
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
