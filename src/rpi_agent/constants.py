"""Constants & defaults shared across the RPI agent."""

from __future__ import annotations

from pathlib import Path

APP_NAME: str = "rpi_agent_app"
USER_ID: str = "local_developer"
DEFAULT_MODEL: str = "gpt-4o"
DEFAULT_GEMINI_MODEL: str = "gemini-3.8-flash"
DEFAULT_BASE_URL: str = "https://api.openai.com/v1"
GEMINI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_CONFIG_PATH: Path = Path.home() / ".config" / "rpi-agent.yml"
DEFAULT_SKILLS_DIR: Path = Path.home() / ".agents" / "skills"
STATE_DIR_NAME: str = ".rpi_agent"
DEFAULT_LOG_LEVEL: str = "WARNING"
VALID_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
DEFAULT_LOG_BUFFER_MAX_RECORDS: int = 2000

# Model API Retry Policy
DEFAULT_MAX_RETRIES: int = 5
RETRY_BASE_DELAY_SECONDS: float = 2.0
RETRY_MAX_DELAY_SECONDS: float = 60.0
# A Retry-After longer than this is treated as non-retryable within a turn.
MAX_RETRY_AFTER_SECONDS: float = 300.0

# Token Optimization Thresholds
MAX_TOOL_RESULT_PRESERVE_CHARS: int = 600
MAX_SEARCH_MATCHES_DEFAULT: int = 40
MAX_FIND_FILES_DEFAULT: int = 100
MAX_CTAGS_CONTEXT_LINES: int = 12

CTAGS_CANDIDATE_FILES: tuple[str, ...] = (
    "tags",
    ".tags",
    ".git/tags",
)
CTAGS_BINARIES: tuple[str, ...] = (
    "ctags",
    "universal-ctags",
    "uctags",
    "exuberant-ctags",
    "ctags-exuberant",
)

IGNORED_DIRECTORIES: set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    ".thoughts",
    ".rpi_agent",
}

SAFE_READ_COMMAND_PREFIXES: tuple[str, ...] = (
    "git status",
    "git branch",
    "git log",
    "git rev-parse",
    "git diff",
    "git show",
    "date",
    "pwd",
    "cargo check",
    "npm run typecheck",
    "uv run ruff check",
)
