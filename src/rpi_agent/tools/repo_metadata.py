"""Live repository metadata tool."""

from __future__ import annotations

from datetime import UTC, datetime

from rpi_agent.repo import RepoContext
from rpi_agent.tools.base import RepoScopedTool


class RepoMetadataTool(RepoScopedTool):
    """Provides the `get_repo_metadata` tool."""

    def get_repo_metadata(self) -> str:
        """Inspects and returns live repository metadata for frontmatter or headers.

        Returns:
            Formatted key-value block of Git metadata and ISO timestamp.
        """
        repo_ctx = RepoContext.capture(self.repo_root)
        now = datetime.now(UTC).isoformat()
        return (
            f"Repository: {repo_ctx.repo_name}\n"
            f"Root Path: {repo_ctx.root_path}\n"
            f"Active Branch: {repo_ctx.branch_name}\n"
            f"Commit SHA: {repo_ctx.commit_sha}\n"
            f"Timestamp: {now}\n"
            f"Git Working Tree: {repo_ctx.is_git}"
        )
