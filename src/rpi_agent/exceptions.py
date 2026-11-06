"""Exception types raised by the RPI agent."""


class ConfigurationError(Exception):
    """Raised when environment or CLI configuration is invalid or missing."""


class SkillNotFoundError(Exception):
    """Raised when a requested skill cannot be resolved on disk."""


class CodebaseToolError(Exception):
    """Raised when a filesystem, git, or command execution tool fails."""
