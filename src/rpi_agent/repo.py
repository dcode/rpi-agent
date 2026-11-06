"""Git repository context capture."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class RepoContext:
    """Metadata representing the current working directory's Git repository.

    Attributes:
        root_path: Absolute filesystem path to the repository root.
        repo_name: Clean directory name of the repository.
        branch_name: Current active Git branch or detached head identifier.
        commit_sha: Full HEAD commit SHA.
        is_git: Whether current directory is inside a valid Git working tree.
    """

    root_path: Path
    repo_name: str
    branch_name: str
    commit_sha: str
    is_git: bool

    @classmethod
    def capture(cls, target_path: Path | None = None) -> RepoContext:
        """Captures Git repository metadata for a given path or current working directory.

        Args:
            target_path: Directory to inspect. Defaults to Path.cwd().

        Returns:
            RepoContext populated with live Git metadata or fallback markers.
        """
        cwd = (target_path or Path.cwd()).resolve()

        def _run_git(args: list[str]) -> str | None:
            try:
                result = subprocess.run(
                    ["git", *args],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    return result.stdout.strip()
            except OSError, subprocess.SubprocessError:
                pass
            return None

        toplevel = _run_git(["rev-parse", "--show-toplevel"])
        if toplevel:
            root = Path(toplevel).resolve()
            repo_name = root.name
            branch = _run_git(["branch", "--show-current"]) or "[Detached HEAD]"
            commit = _run_git(["rev-parse", "HEAD"]) or "[Uncommitted]"
            return cls(
                root_path=root,
                repo_name=repo_name,
                branch_name=branch,
                commit_sha=commit,
                is_git=True,
            )

        return cls(
            root_path=cwd,
            repo_name=cwd.name,
            branch_name="[Non-Git]",
            commit_sha="[No-Commit]",
            is_git=False,
        )
