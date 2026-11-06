from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from rpi_agent.tools import CodebaseTools
from rpi_agent.tools import ctags as ctags_module
from rpi_agent.tools.ctags import TagEntry, _extract_excmd

SAMPLE_TAGS = """\
!_TAG_FILE_FORMAT\t2\t/extended format/
!_TAG_KIND_DESCRIPTION!Python\tc,class\t/classes/
!_TAG_KIND_DESCRIPTION!Python\tf,function\t/functions/
!_TAG_KIND_DESCRIPTION\tz,customkind\t/Generic header kind/
!_TAG_KIND_DESCRIPTION!badheader
!_TAG_KIND_DESCRIPTION!empty\t,

\t
malformed_too_few_fields\tpkg/app.py
\tpkg/app.py\t1;"\tc
empty_file\t\t1;"\tc
outside_escape\t../outside.py\t1;"\tc\tline:1\tlanguage:Python
Service\tpkg/app.py\t3;"\tc\tline:3\tlanguage:Python\troles:def\tend:5
run\tpkg/app.py\t/^    def run(self):$/;"\tkind:method\tlanguage:Python\tscope:class:Service\troles:def\tsignature:(self)
main\tpkg/app.py\t?async def main():$?;"\tf\tlanguage:Python
helper\tpkg/extra.py\t2;/^def helper():$/;"\tfunction\textra_bare\tlanguage:Python\tclass:Mod
GLOBAL_VAR\tpkg/extra.py\t1;"\tK:variable\tline:1\tlanguage:Python\troles:assigned
second_fn\tpkg/sub/mod.py\t/^def second_fn():$/;"\tf\tlanguage:Python\t\t
custom_sym\tpkg/app.py\traw_excmd\tz
unknown_sym\tpkg/app.py\t1;"\tQ
"""


@pytest.fixture
def tagged_repo(git_repo: Path) -> Path:
    """Populates `git_repo` with sample source files and a `tags` file."""
    (git_repo / "pkg" / "extra.py").write_text(
        "GLOBAL_VAR = 42\ndef helper():\n    return GLOBAL_VAR\n",
        encoding="utf-8",
    )
    (git_repo / "pkg" / "sub").mkdir(parents=True)
    (git_repo / "pkg" / "sub" / "mod.py").write_text(
        "# mod\ndef second_fn():\n    pass\n",
        encoding="utf-8",
    )
    (git_repo / "tags").write_text(SAMPLE_TAGS, encoding="utf-8")
    return git_repo


def test_missing_tags_file_messages(tools: CodebaseTools) -> None:
    search_msg = tools.search_symbols("Service")
    assert "No Ctags file" in search_msg
    assert "generate_ctags" in search_msg

    list_msg = tools.list_symbols(".")
    assert "No Ctags file" in list_msg
    assert "generate_ctags" in list_msg


def test_discovers_dot_tags_and_git_tags(tools: CodebaseTools, git_repo: Path) -> None:
    dot_tags = git_repo / ".tags"
    dot_tags.write_text('Service\tpkg/app.py\t3;"\tc\n', encoding="utf-8")
    assert "pkg/app.py:3: [class] Service" in tools.search_symbols("Service")
    dot_tags.unlink()

    git_tags = git_repo / ".git" / "tags"
    git_tags.write_text('Service\tpkg/app.py\t3;"\tc\n', encoding="utf-8")
    assert "pkg/app.py:3: [class] Service" in tools.search_symbols("Service")


def test_search_symbols_basic_and_case_sensitivity(
    tools: CodebaseTools, tagged_repo: Path
) -> None:
    res = tools.search_symbols("service")
    assert "pkg/app.py:3: [class, Python] Service" in res
    assert "outside_escape" not in res

    assert (
        tools.search_symbols("service", case_sensitive=True)
        == "No symbols matching query 'service' found in ctags index."
    )
    assert "Service" in tools.search_symbols("Service", case_sensitive=True)


def test_search_symbols_filters_kind_language_and_path(
    tools: CodebaseTools, tagged_repo: Path
) -> None:
    assert "Service" in tools.search_symbols(".*", kind="c")
    assert "Service" in tools.search_symbols(".*", kind="class")

    assigned = tools.search_symbols("GLOBAL_VAR", kind="variable", language="Python")
    assert "pkg/extra.py:1: [variable, Python, roles:assigned] GLOBAL_VAR" in assigned

    sub_only = tools.search_symbols(".*", path="pkg/sub")
    assert "second_fn" in sub_only
    assert "Service" not in sub_only

    file_only = tools.search_symbols(".*", path="pkg/extra.py")
    assert "helper" in file_only
    assert "Service" not in file_only


def test_search_symbols_resolves_pattern_lines_and_context(
    tools: CodebaseTools, tagged_repo: Path
) -> None:
    with_ctx = tools.search_symbols("^Service$", include_context=True)
    assert "pkg/app.py:3: [class, Python] Service" in with_ctx
    assert "    3 | class Service:" in with_ctx
    assert "    4 |     def run(self):" in with_ctx

    run_ctx = tools.search_symbols("^run$", include_context=True)
    assert "pkg/app.py:4: [method, Python, scope:class:Service] run" in run_ctx
    assert "    4 |     def run(self):" in run_ctx

    fn_match = tools.search_symbols("second_fn", language="Python")
    assert "pkg/sub/mod.py:2: [function, Python] second_fn" in fn_match


def test_search_symbols_truncates_at_max_matches(
    tools: CodebaseTools, tagged_repo: Path
) -> None:
    res = tools.search_symbols(".*", max_matches=2)
    assert "[Truncated after reaching 2 symbol matches." in res


def test_search_symbols_errors(
    tools: CodebaseTools, tagged_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tools.search_symbols("(").startswith("Error in search_symbols:")
    assert tools.search_symbols("x", path="../outside").startswith(
        "Error in search_symbols: Security"
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk error")

    monkeypatch.setattr(Path, "open", boom)
    assert tools.search_symbols("Service") == "Error in search_symbols: disk error"


def test_list_symbols_directory_and_file(
    tools: CodebaseTools, tagged_repo: Path
) -> None:
    all_syms = tools.list_symbols(".")
    assert all_syms.startswith("### Indexed Symbols under `.` (8 symbols):")
    assert "pkg/extra.py:2: [function, Python, scope:class:Mod] helper" in all_syms
    assert "pkg/app.py: [customkind] custom_sym" in all_syms
    assert "pkg/app.py:1: [Q] unknown_sym" in all_syms

    py_only = tools.list_symbols("pkg/sub", language="Python", max_matches=2)
    assert "### Indexed Symbols under `pkg/sub` (1 symbols):" in py_only

    py_all = tools.list_symbols("pkg", language="Python", max_matches=2)
    assert "### Indexed Symbols under `pkg` (6 symbols):" in py_all
    assert "[... and 4 more symbols]" in py_all

    assert (
        tools.list_symbols("pkg", kind="nonexistent")
        == "No indexed symbols found under 'pkg'."
    )


def test_list_symbols_errors(
    tools: CodebaseTools, tagged_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tools.list_symbols("nope") == "Error: 'nope' does not exist."
    assert tools.list_symbols("../escape").startswith("Error in list_symbols: Security")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("cannot read tags")

    monkeypatch.setattr(Path, "open", boom)
    assert tools.list_symbols(".") == "Error in list_symbols: cannot read tags"


def test_resolve_line_and_snippet_edge_cases(
    tools: CodebaseTools, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_entry = TagEntry(
        name="ghost",
        file_path="deleted.py",
        line=None,
        end_line=None,
        pattern="def ghost():",
        kind_code="",
        kind_name="",
        language="",
        scope="",
        roles="",
    )
    assert tools._resolve_line_and_snippet(missing_entry, include_context=True) == (
        None,
        "",
    )
    assert missing_entry.format_summary() == "deleted.py: [symbol] ghost"

    unmatched = TagEntry(
        name="missing_pat",
        file_path="pkg/app.py",
        line=None,
        end_line=None,
        pattern="never_in_file_xyz",
        kind_code="f",
        kind_name="function",
        language="Python",
        scope="",
        roles="",
    )
    assert tools._resolve_line_and_snippet(unmatched, include_context=True) == (
        None,
        "",
    )

    entry_with_line = TagEntry(
        name="Service",
        file_path="pkg/app.py",
        line=3,
        end_line=None,
        pattern="",
        kind_code="c",
        kind_name="class",
        language="Python",
        scope="",
        roles="",
    )

    def fail_read(*_args: object, **_kwargs: object) -> str:
        raise OSError("unreadable")

    monkeypatch.setattr(Path, "read_text", fail_read)
    assert tools._resolve_line_and_snippet(entry_with_line, include_context=True) == (
        3,
        "",
    )


def test_extract_excmd_escaped_slashes() -> None:
    line, pat = _extract_excmd(r"/^path\/to\/file\?$/;")
    assert line is None
    assert "path" in pat

    line2, pat2 = _extract_excmd(r'/^a\/b\?c$/;"')
    assert line2 is None
    assert pat2 == "a/b?c"

    line3, pat3 = _extract_excmd(r'/no_anchors/;"')
    assert line3 is None
    assert pat3 == "no_anchors"


# ------------------------------------------------------------------------------
# generate_ctags & binary detection tests
# ------------------------------------------------------------------------------


def test_generate_ctags_skips_when_already_present(
    tools: CodebaseTools, tagged_repo: Path
) -> None:
    res = tools.generate_ctags()
    assert res == (
        "Ctags file 'tags' already exists (8 indexed symbols). "
        "Pass `force=True` to regenerate."
    )


def test_generate_ctags_universal_and_exuberant(
    tools: CodebaseTools, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_cmds: list[list[str]] = []

    def fake_which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name == "universal-ctags" else None

    def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        seen_cmds.append(cmd)
        if "--version" in cmd:
            return SimpleNamespace(
                returncode=0,
                stdout="Universal Ctags 6.2.1 (derived from Exuberant Ctags)",
                stderr="",
            )
        # Write a valid tags file when indexing
        out_idx = cmd.index("-f") + 1
        Path(cmd[out_idx]).write_text(
            'Service\tpkg/app.py\t3;"\tc\tline:3\tlanguage:Python\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ctags_module.shutil, "which", fake_which)
    monkeypatch.setattr(ctags_module.subprocess, "run", fake_run)

    msg = tools.generate_ctags()
    assert (
        msg == "Generated 'tags' using Universal Ctags (1 symbols indexed under '.')."
    )
    assert "--fields=+lKnsne" in seen_cmds[-1]

    # Force regeneration on a subdirectory with Exuberant Ctags
    def fake_run_exuberant(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        seen_cmds.append(cmd)
        if "--version" in cmd:
            return SimpleNamespace(
                returncode=0,
                stdout="Exuberant Ctags 5.8, Copyright (C) Darren Hiebert",
                stderr="",
            )
        out_idx = cmd.index("-f") + 1
        Path(cmd[out_idx]).write_text(
            'Service\tpkg/app.py\t3;"\tc\tline:3\tlanguage:Python\n'
            'main\tpkg/app.py\t7;"\tf\tline:7\tlanguage:Python\n',
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ctags_module.subprocess, "run", fake_run_exuberant)
    msg_force = tools.generate_ctags(path="pkg", force=True)
    assert msg_force == (
        "Generated 'tags' using Exuberant Ctags (2 symbols indexed under 'pkg')."
    )
    assert "--fields=+lKns" in seen_cmds[-1]
    assert seen_cmds[-1][-1] == "pkg"


def test_detect_ctags_rejects_bsd_and_handles_probe_errors(
    tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_which(name: str) -> str | None:
        if name == "ctags":
            return "/usr/bin/ctags"
        if name == "uctags":
            return "/usr/local/bin/uctags"
        return None

    def fake_run(cmd: list[str], **_kwargs: object) -> SimpleNamespace:
        if cmd[0] == "/usr/bin/ctags":
            # BSD ctags fails on --version withoutUniversal/Exuberant banner
            return SimpleNamespace(
                returncode=1, stdout="", stderr="ctags: illegal option -- -"
            )
        raise OSError("cannot exec")

    monkeypatch.setattr(ctags_module.shutil, "which", fake_which)
    monkeypatch.setattr(ctags_module.subprocess, "run", fake_run)

    res = tools.generate_ctags()
    assert res.startswith(
        "Error: No compatible Universal Ctags or Exuberant Ctags binary found"
    )


def test_generate_ctags_error_paths(
    tools: CodebaseTools, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tools.generate_ctags(path="missing_dir") == (
        "Error: 'missing_dir' does not exist."
    )
    assert tools.generate_ctags(path="../outside").startswith(
        "Error in generate_ctags: Security"
    )

    monkeypatch.setattr(
        tools, "_detect_ctags_binary", lambda: ("/usr/bin/ctags", "universal")
    )

    # Non-zero exit from ctags
    monkeypatch.setattr(
        ctags_module.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(
            returncode=2, stdout="", stderr="permission denied"
        ),
    )
    assert tools.generate_ctags() == "Error generating ctags index: permission denied"

    # TimeoutExpired
    def raise_timeout(*_a: object, **_k: object) -> None:
        raise subprocess.TimeoutExpired(cmd="ctags", timeout=60)

    monkeypatch.setattr(ctags_module.subprocess, "run", raise_timeout)
    assert tools.generate_ctags() == (
        "Error generating ctags index: ctags command timed out after 60s."
    )

    # Generic OSError
    def raise_os_error(*_a: object, **_k: object) -> None:
        raise OSError("spawn failed")

    monkeypatch.setattr(ctags_module.subprocess, "run", raise_os_error)
    assert tools.generate_ctags() == "Error in generate_ctags: spawn failed"
