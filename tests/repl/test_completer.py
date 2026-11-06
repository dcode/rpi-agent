from __future__ import annotations

from pathlib import Path

import pytest
from prompt_toolkit.completion import CompleteEvent
from prompt_toolkit.document import Document

from rpi_agent.repl.completer import RpiCliCompleter
from rpi_agent.skills import SkillRegistry


@pytest.fixture
def completer(git_repo: Path, registry: SkillRegistry) -> RpiCliCompleter:
    return RpiCliCompleter(git_repo, registry)


def complete(completer: RpiCliCompleter, text: str) -> list[tuple[str, int]]:
    return [
        (c.text, c.start_position)
        for c in completer.get_completions(Document(text), CompleteEvent())
    ]


def test_slash_prefix_matches(completer: RpiCliCompleter) -> None:
    assert complete(completer, "/s") == [("/skills", -2), ("/status", -2)]


def test_slash_lists_all_commands(completer: RpiCliCompleter) -> None:
    assert len(complete(completer, "/")) == len(completer.slash_commands)


def test_slash_display_includes_description(completer: RpiCliCompleter) -> None:
    (completion,) = completer.get_completions(Document("/hel"), CompleteEvent())
    assert "Show CLI usage" in completion.display_text


def test_slash_no_match(completer: RpiCliCompleter) -> None:
    assert complete(completer, "/zzz") == []


def test_plain_text_has_no_completions(completer: RpiCliCompleter) -> None:
    assert complete(completer, "hello wor") == []


def test_dot_slash_lists_repo_root_skipping_ignored(
    completer: RpiCliCompleter,
) -> None:
    names = [text for text, _ in complete(completer, "look at ./")]
    assert names == ["README.md", "pkg"]


def test_missing_directory_yields_nothing(
    completer: RpiCliCompleter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(completer, "repo_root", completer.repo_root / "gone")
    assert complete(completer, "see ./") == []


def test_os_error_is_swallowed(
    completer: RpiCliCompleter, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_self: Path) -> None:
        raise OSError("denied")

    monkeypatch.setattr(Path, "iterdir", boom)
    assert complete(completer, "see ./") == []


def test_nested_path_completion(completer: RpiCliCompleter) -> None:
    assert complete(completer, "open pkg/ap") == [("app.py", -2)]


def test_trailing_slash_lists_that_directory(completer: RpiCliCompleter) -> None:
    assert complete(completer, "open pkg/") == [("app.py", 0)]


def test_dot_relative_prefix(completer: RpiCliCompleter) -> None:
    assert complete(completer, "look at ./pk") == [("pkg", -2)]
    assert complete(completer, "look at ./pkg/a") == [("app.py", -1)]


def test_bare_word_has_no_path_completions(completer: RpiCliCompleter) -> None:
    assert complete(completer, "open pk") == []


def test_dotfile_prefix_completes_from_root(
    completer: RpiCliCompleter, git_repo: Path
) -> None:
    (git_repo / ".envrc").write_text("")
    (git_repo / ".editorconfig").write_text("")
    assert complete(completer, "source .en") == [(".envrc", -3)]
    # Ignored directories stay hidden even when the prefix matches them.
    assert complete(completer, "see .gi") == []
