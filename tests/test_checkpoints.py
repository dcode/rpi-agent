from __future__ import annotations

from pathlib import Path

import frontmatter
import pytest

from rpi_agent.checkpoints import CheckpointStore, checkpoint_key


@pytest.fixture
def store(tmp_path: Path) -> CheckpointStore:
    return CheckpointStore(tmp_path / "subagents")


def test_key_is_stable_and_distinguishes_inputs() -> None:
    key = checkpoint_key("locator", "find auth")
    assert key == checkpoint_key("locator", "find auth")
    assert len(key) == 16
    assert key != checkpoint_key("locator", "find auth ")
    assert key != checkpoint_key("analyzer", "find auth")
    # The separator prevents ("ab", "c") colliding with ("a", "bc").
    assert checkpoint_key("ab", "c") != checkpoint_key("a", "bc")


def test_load_missing_returns_none(store: CheckpointStore) -> None:
    assert store.load("sess_1", "locator", "task") is None
    assert store.for_session("sess_1") == []


def test_save_then_load_roundtrip(store: CheckpointStore) -> None:
    saved = store.save(
        "sess_1", "rpi-codebase-locator", "find auth", "## Found\nauth.py"
    )
    loaded = store.load("sess_1", "rpi-codebase-locator", "find auth")

    assert loaded is not None
    assert loaded.result == "## Found\nauth.py"
    assert loaded.created == saved.created
    assert loaded.path == saved.path


def test_files_are_numbered_readable_markdown(store: CheckpointStore) -> None:
    first = store.save("sess_1", "rpi-codebase-locator", "t1", "r1")
    second = store.save("sess_1", "rpi codebase/analyzer", "t2", "r2")

    assert (
        first.path.name
        == f"001-rpi-codebase-locator-{checkpoint_key('rpi-codebase-locator', 't1')}.md"
    )
    assert second.path.name.startswith("002-rpi_codebase_analyzer-")

    post = frontmatter.load(first.path)
    assert post.metadata["skill"] == "rpi-codebase-locator"
    assert post.metadata["task"] == "t1"
    assert post.content == "r1"
    assert [c.result for c in store.for_session("sess_1")] == ["r1", "r2"]


def test_sessions_are_isolated(store: CheckpointStore) -> None:
    store.save("sess_1", "locator", "task", "one")
    assert store.load("sess_2", "locator", "task") is None


def test_session_id_is_sanitised_for_paths(store: CheckpointStore) -> None:
    assert store.session_dir("../../etc") == store.root / "______etc"
    assert store.session_dir("") == store.root / "_"


def test_task_mismatch_with_same_key_is_ignored(store: CheckpointStore) -> None:
    saved = store.save("sess_1", "locator", "task", "result")
    # Simulate a hash collision: same filename, different recorded task.
    post = frontmatter.load(saved.path)
    post.metadata["task"] = "other task"
    saved.path.write_text(frontmatter.dumps(post))
    assert store.load("sess_1", "locator", "task") is None


def test_unreadable_checkpoints_are_skipped(store: CheckpointStore) -> None:
    store.save("sess_1", "locator", "task", "good")
    bad = store.session_dir("sess_1") / "002-bad-0000000000000000.md"
    bad.write_text("---\nskill: [unclosed\n---\n")
    assert [c.result for c in store.for_session("sess_1")] == ["good"]
