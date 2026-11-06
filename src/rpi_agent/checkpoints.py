"""On-disk checkpoints of completed sub-agent results.

Each successful sub-agent run is saved as a Markdown file with frontmatter under
`<state_dir>/subagents/<session_id>/`. When the same skill is asked the same task
again in that session (e.g. after resuming once quota resets), the saved result
is reused instead of spending tokens to recompute it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import frontmatter
import yaml


@dataclass(slots=True)
class Checkpoint:
    """A saved sub-agent result.

    Attributes:
        path: File the checkpoint is stored in.
        skill_name: Skill the sub-agent executed.
        task_prompt: Task the sub-agent was given.
        result: Final text the sub-agent returned.
        created: ISO-8601 UTC timestamp of when it was saved.
    """

    path: Path
    skill_name: str
    task_prompt: str
    result: str
    created: str


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", value) or "_"


def checkpoint_key(skill_name: str, task_prompt: str) -> str:
    """Stable identifier for a (skill, task) pair."""
    digest = hashlib.sha256(f"{skill_name}\0{task_prompt}".encode())
    return digest.hexdigest()[:16]


class CheckpointStore:
    """Saves and looks up sub-agent results per session."""

    def __init__(self, root: Path) -> None:
        """Initializes the store.

        Args:
            root: Directory holding one subdirectory per session.
        """
        self.root = root

    def session_dir(self, session_id: str) -> Path:
        """Directory holding checkpoints for a session."""
        return self.root / _safe_component(session_id)

    def load(
        self, session_id: str, skill_name: str, task_prompt: str
    ) -> Checkpoint | None:
        """Returns the saved result for this skill and task, if any."""
        key = checkpoint_key(skill_name, task_prompt)
        for path in sorted(self.session_dir(session_id).glob(f"*-{key}.md")):
            checkpoint = self._read(path)
            if (
                checkpoint is not None
                and checkpoint.skill_name == skill_name
                and checkpoint.task_prompt == task_prompt
            ):
                return checkpoint
        return None

    def save(
        self, session_id: str, skill_name: str, task_prompt: str, result: str
    ) -> Checkpoint:
        """Persists a sub-agent result, numbered in completion order."""
        directory = self.session_dir(session_id)
        directory.mkdir(parents=True, exist_ok=True)

        sequence = len(list(directory.glob("*.md"))) + 1
        key = checkpoint_key(skill_name, task_prompt)
        path = directory / f"{sequence:03d}-{_safe_component(skill_name)}-{key}.md"
        created = datetime.now(UTC).isoformat(timespec="seconds")

        post = frontmatter.Post(
            result,
            skill=skill_name,
            task=task_prompt,
            created=created,
        )
        path.write_text(frontmatter.dumps(post) + "\n", encoding="utf-8")
        return Checkpoint(path, skill_name, task_prompt, result, created)

    def for_session(self, session_id: str) -> list[Checkpoint]:
        """Returns a session's readable checkpoints in completion order."""
        checkpoints = (
            self._read(path)
            for path in sorted(self.session_dir(session_id).glob("*.md"))
        )
        return [c for c in checkpoints if c is not None]

    @staticmethod
    def _read(path: Path) -> Checkpoint | None:
        try:
            post = frontmatter.load(path)
        except OSError, ValueError, yaml.YAMLError:
            return None
        return Checkpoint(
            path=path,
            skill_name=str(post.metadata.get("skill", "")),
            task_prompt=str(post.metadata.get("task", "")),
            result=post.content,
            created=str(post.metadata.get("created", "")),
        )
