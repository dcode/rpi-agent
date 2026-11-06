"""Exception types raised by the RPI agent."""


class ConfigurationError(Exception):
    """Raised when environment or CLI configuration is invalid or missing."""


class SkillNotFoundError(Exception):
    """Raised when a requested skill cannot be resolved on disk."""


class CodebaseToolError(Exception):
    """Raised when a filesystem, git, or command execution tool fails."""


class QuotaExhaustedError(Exception):
    """Raised when the gateway reports a quota that retrying cannot recover.

    Attributes:
        reset_hint: Human-readable time until the quota resets, if reported.
    """

    def __init__(self, message: str, reset_hint: str | None = None) -> None:
        super().__init__(message)
        self.reset_hint = reset_hint
