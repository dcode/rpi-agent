from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.exceptions import SkillNotFoundError
from rpi_agent.skills import SkillRegistry


def test_scan_missing_dir_returns_zero(tmp_path: Path) -> None:
    reg = SkillRegistry(tmp_path / "nope")
    assert reg.scan() == 0
    assert reg.skills == {}


def test_scan_path_is_file_returns_zero(tmp_path: Path) -> None:
    file = tmp_path / "file"
    file.write_text("x")
    assert SkillRegistry(file).scan() == 0


def test_scan_loads_metadata_and_resources(registry: SkillRegistry) -> None:
    assert set(registry.skills) == {"rpi-research", "bare"}

    research = registry.skills["rpi-research"]
    assert research.description == "Research a topic"
    assert research.instructions == "Do research."
    assert research.resources == ["templates/doc.md"]


def test_scan_falls_back_to_dir_name_and_default_description(
    registry: SkillRegistry,
) -> None:
    bare = registry.skills["bare"]
    assert bare.description == "No description provided."
    assert bare.resources == []


def test_rescan_clears_previous_results(
    registry: SkillRegistry, skills_dir: Path
) -> None:
    (skills_dir / "bare" / "SKILL.md").unlink()
    assert registry.scan() == 1
    assert "bare" not in registry.skills


def test_scan_skips_unreadable_skill(skills_dir: Path) -> None:
    broken = skills_dir / "broken"
    broken.mkdir()
    (broken / "SKILL.md").write_bytes(b"\xff\xfe invalid utf-8")
    reg = SkillRegistry(skills_dir)
    assert reg.scan() == 2
    assert "broken" not in reg.skills


@pytest.mark.xfail(
    reason="Bug: yaml.YAMLError from malformed frontmatter is not caught by scan()",
    raises=Exception,
)
def test_scan_skips_malformed_frontmatter(skills_dir: Path) -> None:
    bad = skills_dir / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("---\nname: [unclosed\n---\nbody\n")
    assert SkillRegistry(skills_dir).scan() == 2


def test_list_skills_empty(tmp_path: Path) -> None:
    assert SkillRegistry(tmp_path).list_skills() == (
        "No skills currently discovered in registry."
    )


def test_list_skills_catalog(registry: SkillRegistry) -> None:
    catalog = registry.list_skills()
    assert catalog.startswith("### Available Skills Catalog")
    assert "- **`rpi-research`**: Research a topic (Resources: 1)" in catalog
    assert "- **`bare`**: No description provided." in catalog
    assert "bare`**: No description provided. (Resources" not in catalog


def test_load_skill(registry: SkillRegistry) -> None:
    text = registry.load_skill("rpi-research")
    assert text.startswith("--- Skill: rpi-research ---")
    assert "Description: Research a topic" in text
    assert text.endswith("Do research.")


def test_load_skill_missing_lists_available(registry: SkillRegistry) -> None:
    with pytest.raises(SkillNotFoundError, match="bare, rpi-research"):
        registry.load_skill("nope")


def test_load_skill_resource(registry: SkillRegistry) -> None:
    assert registry.load_skill_resource("rpi-research", "templates/doc.md") == (
        "TEMPLATE"
    )


def test_load_skill_resource_missing_skill(registry: SkillRegistry) -> None:
    with pytest.raises(SkillNotFoundError, match="Parent skill 'nope'"):
        registry.load_skill_resource("nope", "x")


@pytest.mark.parametrize("rel_path", ["missing.md", "../bare/SKILL.md", "templates"])
def test_load_skill_resource_rejects_missing_or_escaping(
    registry: SkillRegistry, rel_path: str
) -> None:
    with pytest.raises(SkillNotFoundError, match="not found in skill"):
        registry.load_skill_resource("rpi-research", rel_path)


def test_load_skill_resource_read_error(
    registry: SkillRegistry, skills_dir: Path
) -> None:
    (skills_dir / "rpi-research" / "binary.bin").write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(SkillNotFoundError, match="Failed to read skill resource"):
        registry.load_skill_resource("rpi-research", "binary.bin")
