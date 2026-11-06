from __future__ import annotations

from pathlib import Path

import pytest

from rpi_agent.tools import CodebaseTools


def _outline(tools: CodebaseTools, repo: Path, rel: str, content: str) -> str:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return tools.outline_file(rel)


def test_python(tools: CodebaseTools) -> None:
    result = tools.outline_file("pkg/app.py")
    assert result.startswith("### Structural Outline of `pkg/app.py`")
    assert "    3 | class Service:" in result
    assert "    7 | async def main():" in result
    assert "import os" not in result
    assert "read_file" in result


@pytest.mark.xfail(
    reason="Bug: '^' only anchors the def branch, so indented methods are skipped",
    raises=AssertionError,
)
def test_python_includes_methods(tools: CodebaseTools) -> None:
    assert "    4 |     def run(self):" in tools.outline_file("pkg/app.py")


@pytest.mark.xfail(
    reason="Bug: the class branch is unanchored and matches inside any line",
    raises=AssertionError,
)
def test_python_ignores_class_in_comments(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(tools, git_repo, "c.py", "x = 1  # subclass Foo\n")
    assert result.startswith("No specific structural declarations")


def test_terraform(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(
        tools,
        git_repo,
        "main.tf",
        'resource "aws_s3_bucket" "b" {\n  x = 1\n}\nvariable "region" {}\n',
    )
    assert '    1 | resource "aws_s3_bucket" "b" {' in result
    assert '    4 | variable "region" {}' in result
    assert "x = 1" not in result


def test_yaml(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(
        tools,
        git_repo,
        "site.yml",
        "- hosts: all\n  tasks:\n    - name: Install\n      apt: {}\n",
    )
    assert "    2 |   tasks:" in result
    assert "    3 |     - name: Install" in result
    assert "apt" not in result


def test_ansible_heuristic_by_path(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(tools, git_repo, "roles/x/tasks/main", "- name: Step\n")
    assert "    1 | - name: Step" in result


def test_shell(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(
        tools,
        git_repo,
        "run.sh",
        "#!/bin/sh\nsetup() {\n  :\n}\nfunction teardown() {\n}\n",
    )
    assert "    2 | setup() {" in result
    assert "    5 | function teardown() {" in result


def test_markdown(tools: CodebaseTools) -> None:
    result = tools.outline_file("README.md")
    assert "    1 | # Title" in result
    assert "    3 | ## Usage" in result
    assert "text" not in result.split("\n\n(Use")[0]


def test_generic_fallback(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(
        tools, git_repo, "lib.rs", "use std;\npub fn go() {}\nstruct S;\n"
    )
    assert "    2 | pub fn go() {}" in result
    assert "    3 | struct S;" in result


def test_no_declarations_returns_preview(tools: CodebaseTools, git_repo: Path) -> None:
    result = _outline(tools, git_repo, "notes.txt", "alpha\nbeta\n")
    assert result.startswith(
        "No specific structural declarations identified in 'notes.txt'."
    )
    assert "    1 | alpha" in result
    assert "    2 | beta" in result
    assert "   15 | " in result


def test_missing_file(tools: CodebaseTools) -> None:
    assert tools.outline_file("nope.py") == (
        "Error: 'nope.py' does not exist or is not a file."
    )


def test_directory_is_not_a_file(tools: CodebaseTools) -> None:
    assert "is not a file" in tools.outline_file("pkg")


def test_escape(tools: CodebaseTools) -> None:
    assert tools.outline_file("../x.py").startswith(
        "Error outlining file '../x.py': Security"
    )


def test_os_error(tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("denied")

    monkeypatch.setattr(Path, "open", boom)
    assert tools.outline_file("pkg/app.py") == (
        "Error outlining file 'pkg/app.py': denied"
    )
