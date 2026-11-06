"""Skill discovery engine for SKILL.md packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import frontmatter

from rpi_agent.exceptions import SkillNotFoundError


@dataclass(slots=True)
class SkillMetadata:
    """In-memory representation of an individual discovered skill.

    Attributes:
        name: Unique name of the skill.
        description: Concise functional overview of the skill.
        skill_file_path: Path to the underlying SKILL.md file.
        instructions: Raw Markdown instructions body.
        resources: Relative file paths of auxiliary scripts, prompts, or templates.
    """

    name: str
    description: str
    skill_file_path: Path
    instructions: str
    resources: list[str] = field(default_factory=list)


class SkillRegistry:
    """Scans, parses, and provides access to local SKILL.md specifications."""

    def __init__(self, skills_dir: Path) -> None:
        """Initializes the registry.

        Args:
            skills_dir: Base directory containing skill packages.
        """
        self.skills_dir = skills_dir
        self._skills: dict[str, SkillMetadata] = {}

    @property
    def skills(self) -> dict[str, SkillMetadata]:
        """Returns discovered skills dictionary keyed by skill name."""
        return self._skills

    def scan(self) -> int:
        """Scans the configured skills directory for SKILL.md files.

        Returns:
            Number of successfully loaded skills.
        """
        self._skills.clear()
        if not self.skills_dir.exists() or not self.skills_dir.is_dir():
            return 0

        for skill_file in sorted(self.skills_dir.glob("*/SKILL.md")):
            try:
                post = frontmatter.load(skill_file)
                skill_dir = skill_file.parent
                skill_name = str(post.metadata.get("name") or skill_dir.name).strip()
                description = str(
                    post.metadata.get("description") or "No description provided."
                ).strip()

                resources: list[str] = []
                for entry in sorted(skill_dir.rglob("*")):
                    if entry.is_file() and entry.name != "SKILL.md":
                        try:
                            rel_path = entry.relative_to(skill_dir).as_posix()
                            resources.append(rel_path)
                        except ValueError:
                            continue

                self._skills[skill_name] = SkillMetadata(
                    name=skill_name,
                    description=description,
                    skill_file_path=skill_file,
                    instructions=post.content.strip(),
                    resources=resources,
                )
            except OSError, ValueError, KeyError:
                continue

        return len(self._skills)

    def list_skills(self) -> str:
        """Returns a Markdown catalog of all available skills.

        Returns:
            Markdown formatted catalog of skill names, descriptions, and resources.
        """
        if not self._skills:
            return "No skills currently discovered in registry."

        lines: list[str] = ["### Available Skills Catalog\n"]
        for skill in self._skills.values():
            res_info = (
                f" (Resources: {len(skill.resources)})" if skill.resources else ""
            )
            lines.append(f"- **`{skill.name}`**: {skill.description}{res_info}")
        return "\n".join(lines)

    def load_skill(self, skill_name: str) -> str:
        """Loads and returns the full instructions of a specific skill.

        Args:
            skill_name: Name of the skill to retrieve.

        Returns:
            Markdown body containing detailed skill instructions.

        Raises:
            SkillNotFoundError: If the requested skill does not exist.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            raise SkillNotFoundError(
                f"Skill '{skill_name}' not found. Available skills: {', '.join(self._skills.keys())}"
            )
        return (
            f"--- Skill: {skill.name} ---\n"
            f"Description: {skill.description}\n"
            f"Location: {skill.skill_file_path}\n\n"
            f"{skill.instructions}"
        )

    def load_skill_resource(self, skill_name: str, resource_rel_path: str) -> str:
        """Reads an auxiliary file or template packaged alongside a skill.

        Args:
            skill_name: Name of the parent skill package.
            resource_rel_path: Relative path to the auxiliary file within the skill dir.

        Returns:
            Raw text content of the target auxiliary resource.

        Raises:
            SkillNotFoundError: If skill or resource is missing or invalid.
        """
        skill = self._skills.get(skill_name)
        if not skill:
            raise SkillNotFoundError(f"Parent skill '{skill_name}' not found.")

        target = (skill.skill_file_path.parent / resource_rel_path).resolve()
        skill_root = skill.skill_file_path.parent.resolve()

        if not target.is_relative_to(skill_root) or not target.is_file():
            raise SkillNotFoundError(
                f"Resource '{resource_rel_path}' not found in skill '{skill_name}'."
            )

        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as err:
            raise SkillNotFoundError(f"Failed to read skill resource: {err}") from err
