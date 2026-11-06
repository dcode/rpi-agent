"""Constants & defaults shared across the RPI agent."""

from __future__ import annotations

from pathlib import Path

APP_NAME: str = "rpi_agent_app"
USER_ID: str = "local_developer"
DEFAULT_MODEL: str = "gemini-3.8-flash-withtools"
DEFAULT_BASE_URL: str = "https://api.genai.mil/v1"
DEFAULT_SKILLS_DIR: Path = Path.home() / ".agents" / "skills"
STATE_DIR_NAME: str = ".rpi_agent"

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
