"""Universal and Exuberant Ctags symbol search, listing, and generation tools."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from rpi_agent.constants import (
    CTAGS_BINARIES,
    CTAGS_CANDIDATE_FILES,
    IGNORED_DIRECTORIES,
    MAX_CTAGS_CONTEXT_LINES,
    MAX_FIND_FILES_DEFAULT,
    MAX_SEARCH_MATCHES_DEFAULT,
)
from rpi_agent.exceptions import CodebaseToolError
from rpi_agent.tools.base import RepoScopedTool

_DEFAULT_KIND_NAMES: dict[str, str] = {
    "c": "class",
    "d": "define",
    "e": "enumerator",
    "f": "function",
    "g": "enumeration",
    "i": "import",
    "l": "local",
    "m": "member",
    "n": "namespace",
    "p": "prototype",
    "s": "struct",
    "t": "typedef",
    "u": "union",
    "v": "variable",
}

_SCOPE_FIELD_KEYS: frozenset[str] = frozenset(
    {
        "scope",
        "class",
        "struct",
        "interface",
        "enum",
        "function",
        "method",
        "module",
        "namespace",
    }
)


@dataclass(frozen=True, slots=True)
class TagEntry:
    """Parsed Ctags symbol record."""

    name: str
    file_path: str
    line: int | None
    end_line: int | None
    pattern: str
    kind_code: str
    kind_name: str
    language: str
    scope: str
    roles: str

    def matches_kind(self, kind_filter: str) -> bool:
        """Returns True if `kind_filter` matches either the short code or full name."""
        if not kind_filter:
            return True
        target = kind_filter.strip().lower()
        return target in {self.kind_code.lower(), self.kind_name.lower()}

    def matches_language(self, language_filter: str) -> bool:
        """Returns True if `language_filter` matches the tag's language."""
        if not language_filter:
            return True
        return self.language.lower() == language_filter.strip().lower()

    def format_summary(self, resolved_line: int | None = None) -> str:
        """Formats the tag into a single compact reference line."""
        line_num = resolved_line if resolved_line is not None else self.line
        loc = f"{self.file_path}:{line_num}" if line_num is not None else self.file_path
        kind_label = self.kind_name or self.kind_code or "symbol"
        meta_parts: list[str] = [kind_label]
        if self.language:
            meta_parts.append(self.language)
        if self.scope:
            meta_parts.append(f"scope:{self.scope}")
        if self.roles and self.roles != "def":
            meta_parts.append(f"roles:{self.roles}")
        meta = ", ".join(meta_parts)
        return f"{loc}: [{meta}] {self.name}"


def _extract_excmd(raw_excmd: str) -> tuple[int | None, str]:
    """Splits a ctags excmd field into an optional 1-indexed line number and pattern."""
    cleaned = raw_excmd[:-2] if raw_excmd.endswith(';"') else raw_excmd
    if cleaned.isdigit():
        return int(cleaned), ""

    # Combined excmd format: `42;/^pattern$/`
    semi_idx = cleaned.find(";")
    if semi_idx > 0 and cleaned[:semi_idx].isdigit():
        return int(cleaned[:semi_idx]), cleaned[semi_idx + 1 :]

    if len(cleaned) >= 2 and cleaned[0] in {"/", "?"} and cleaned[-1] == cleaned[0]:
        body = cleaned[1:-1]
        if body.startswith("^"):
            body = body[1:]
        if body.endswith("$"):
            body = body[:-1]
        body = body.replace(r"\/", "/").replace(r"\?", "?")
        return None, body

    return None, cleaned


class CtagsTool(RepoScopedTool):
    """Provides Ctags symbol lookup (`search_symbols`), listing (`list_symbols`), and index creation (`generate_ctags`)."""

    def _find_tags_file(self) -> Path | None:
        """Locates the first available Ctags file inside the repository."""
        for rel_candidate in CTAGS_CANDIDATE_FILES:
            candidate = self._resolve_safe_path(rel_candidate)
            if candidate.is_file():
                return candidate
        return None

    def _detect_ctags_binary(self) -> tuple[str, str] | None:
        """Finds a compatible Universal Ctags or Exuberant Ctags executable.

        Returns:
            Tuple of (executable_path, implementation_flavor) where flavor is
            'universal' or 'exuberant', or None if no compatible binary is found.
        """
        for binary_name in CTAGS_BINARIES:
            executable = shutil.which(binary_name)
            if not executable:
                continue
            try:
                proc = subprocess.run(
                    [executable, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except OSError, subprocess.SubprocessError:
                continue

            version_text = f"{proc.stdout}\n{proc.stderr}"
            if "Universal Ctags" in version_text:
                return executable, "universal"
            if "Exuberant Ctags" in version_text:
                return executable, "exuberant"

        return None

    def generate_ctags(self, path: str = ".", force: bool = False) -> str:
        """Generates a Ctags index file (`tags`) if one does not already exist.

        Verifies that the installed `ctags` binary is Universal Ctags (preferred)
        or Exuberant Ctags (compatible extended format) before indexing, rejecting
        incompatible BSD/POSIX `ctags` implementations.

        Args:
            path: Relative directory or file path to index. Defaults to '.'.
            force: If True, regenerates the tags file even if one already exists.

        Returns:
            Status message describing the existing or newly generated tags file.
        """
        try:
            target_scope = self._resolve_safe_path(path)
            if not target_scope.exists():
                return f"Error: '{path}' does not exist."

            existing = self._find_tags_file()
            if existing is not None and not force:
                rel_existing = existing.relative_to(self.repo_root).as_posix()
                entries = self._parse_tags_file(existing)
                return (
                    f"Ctags file '{rel_existing}' already exists "
                    f"({len(entries)} indexed symbols). Pass `force=True` to regenerate."
                )

            detected = self._detect_ctags_binary()
            if detected is None:
                return (
                    "Error: No compatible Universal Ctags or Exuberant Ctags binary found "
                    "on PATH (checked: "
                    + ", ".join(CTAGS_BINARIES)
                    + "). BSD/POSIX ctags is not supported."
                )

            executable, flavor = detected
            output_file = (
                existing if existing is not None else self._resolve_safe_path("tags")
            )
            rel_output = output_file.relative_to(self.repo_root).as_posix()
            rel_target = (
                "."
                if target_scope == self.repo_root
                else target_scope.relative_to(self.repo_root).as_posix()
            )

            fields_flag = (
                "--fields=+lKnsne" if flavor == "universal" else "--fields=+lKns"
            )
            cmd = [
                executable,
                "-R",
                "--excmd=number",
                fields_flag,
                *[f"--exclude={ignored}" for ignored in sorted(IGNORED_DIRECTORIES)],
                "-f",
                str(output_file),
                rel_target,
            ]

            result = subprocess.run(
                cmd,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or f"exit code {result.returncode}"
                return f"Error generating ctags index: {stderr}"

            entries = self._parse_tags_file(output_file)
            flavor_label = (
                "Universal Ctags" if flavor == "universal" else "Exuberant Ctags"
            )
            return (
                f"Generated '{rel_output}' using {flavor_label} "
                f"({len(entries)} symbols indexed under '{rel_target}')."
            )
        except subprocess.TimeoutExpired:
            return "Error generating ctags index: ctags command timed out after 60s."
        except (OSError, subprocess.SubprocessError, CodebaseToolError) as err:
            return f"Error in generate_ctags: {err}"

    def _parse_tags_file(self, tags_file: Path) -> list[TagEntry]:
        """Parses a Universal or Exuberant Ctags file into validated repository-scoped TagEntry items."""
        kind_header_map: dict[tuple[str, str], str] = {}
        global_kind_map: dict[str, str] = dict(_DEFAULT_KIND_NAMES)
        entries: list[TagEntry] = []

        with tags_file.open("r", encoding="utf-8", errors="replace") as f:
            for raw_line in f:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue

                if line.startswith("!_TAG_"):
                    # Parse `!_TAG_KIND_DESCRIPTION!<Lang>\t<letter>,<name>\t/desc/`
                    if line.startswith("!_TAG_KIND_DESCRIPTION"):
                        parts = line.split("\t")
                        if len(parts) >= 2 and "," in parts[1]:
                            header_key = parts[0]
                            lang = (
                                header_key.split("!", 2)[2].lower()
                                if header_key.count("!") >= 2
                                else ""
                            )
                            letter, kind_full = parts[1].split(",", 1)
                            if letter and kind_full:
                                global_kind_map[letter] = kind_full
                                if lang:
                                    kind_header_map[(lang, letter)] = kind_full
                    continue

                parts = line.split("\t")
                if len(parts) < 3:
                    continue

                tag_name, raw_file, raw_excmd = parts[0], parts[1], parts[2]
                if not tag_name or not raw_file:
                    continue

                try:
                    resolved_file = self._resolve_safe_path(raw_file)
                except CodebaseToolError:
                    continue

                rel_posix = resolved_file.relative_to(self.repo_root).as_posix()
                line_no, pattern = _extract_excmd(raw_excmd)
                end_line: int | None = None
                kind_code = ""
                kind_name = ""
                language = ""
                scope = ""
                roles = ""

                for ext in parts[3:]:
                    if not ext:
                        continue
                    if ":" not in ext:
                        if not kind_code:
                            kind_code = ext
                            if len(ext) > 1:
                                kind_name = ext
                        continue

                    key, val = ext.split(":", 1)
                    if key in {"kind", "K"}:
                        kind_code = kind_code or val
                        kind_name = val
                    elif key == "line" and val.isdigit():
                        line_no = int(val)
                    elif key == "end" and val.isdigit():
                        end_line = int(val)
                    elif key == "language":
                        language = val
                    elif key == "roles":
                        roles = val
                    elif key in _SCOPE_FIELD_KEYS and not scope:
                        scope = val if key == "scope" else f"{key}:{val}"

                if not kind_name and kind_code:
                    kind_name = (
                        kind_header_map.get((language.lower(), kind_code))
                        or global_kind_map.get(kind_code)
                        or kind_code
                    )

                entries.append(
                    TagEntry(
                        name=tag_name,
                        file_path=rel_posix,
                        line=line_no,
                        end_line=end_line,
                        pattern=pattern,
                        kind_code=kind_code,
                        kind_name=kind_name,
                        language=language,
                        scope=scope,
                        roles=roles,
                    )
                )

        return entries

    def _resolve_line_and_snippet(
        self, entry: TagEntry, include_context: bool
    ) -> tuple[int | None, str]:
        """Resolves a tag's line number (if pattern-based) and optional code snippet."""
        line_no = entry.line
        if line_no is not None and not include_context:
            return line_no, ""

        target = self.repo_root / entry.file_path
        if not target.is_file():
            return line_no, ""

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return line_no, ""

        if line_no is None and entry.pattern:
            for idx, file_line in enumerate(lines, start=1):
                if (
                    file_line.rstrip("\r\n") == entry.pattern
                    or entry.pattern in file_line
                ):
                    line_no = idx
                    break

        if not include_context or line_no is None or not lines:
            return line_no, ""

        total = len(lines)
        start = max(1, line_no - 1)
        default_end = line_no + MAX_CTAGS_CONTEXT_LINES - 2
        end = (
            min(total, entry.end_line, default_end)
            if entry.end_line is not None and entry.end_line >= line_no
            else min(total, default_end)
        )
        snippet_lines = [
            f"  {idx:5d} | {lines[idx - 1].rstrip()}" for idx in range(start, end + 1)
        ]
        return line_no, "\n".join(snippet_lines)

    def _matches_scope_path(self, file_path: str, scope_target: Path) -> bool:
        """Checks whether a relative POSIX tag path falls under `scope_target`."""
        if scope_target == self.repo_root:
            return True
        full = self.repo_root / file_path
        return full == scope_target or full.is_relative_to(scope_target)

    def search_symbols(
        self,
        query: str,
        path: str = ".",
        kind: str = "",
        language: str = "",
        case_sensitive: bool = False,
        include_context: bool = False,
        max_matches: int = MAX_SEARCH_MATCHES_DEFAULT,
    ) -> str:
        """Searches a Ctags index (`tags` / `.tags`) for symbol definitions.

        Prefer this tool before `search_code` when locating definitions of classes,
        functions, methods, types, variables, or other language declarations. If no
        Ctags file exists yet, call `generate_ctags` first or fall back to `outline_file`.

        Args:
            query: Regex or substring pattern to match against symbol names.
            path: Relative file or directory path to scope results. Defaults to '.'.
            kind: Optional symbol kind filter (e.g. 'class', 'function', 'method', 'variable').
            language: Optional language filter (e.g. 'Python', 'Go', 'Rust', 'TypeScript').
            case_sensitive: Whether regex matching on symbol names is case-sensitive.
            include_context: If True, includes surrounding source lines for each match.
            max_matches: Maximum number of symbol matches to return.

        Returns:
            Formatted list of matching symbol definitions (and optional code context).
        """
        try:
            scope_target = self._resolve_safe_path(path)
            tags_file = self._find_tags_file()
            if tags_file is None:
                return (
                    "No Ctags file ('tags', '.tags', or '.git/tags') found in "
                    "repository root. Run `generate_ctags` to create one, or fall back "
                    "to `outline_file` / `search_code`."
                )

            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(query, flags)
            entries = self._parse_tags_file(tags_file)

            results: list[str] = []
            for entry in entries:
                if not self._matches_scope_path(entry.file_path, scope_target):
                    continue
                if not entry.matches_kind(kind) or not entry.matches_language(language):
                    continue
                if not pattern.search(entry.name):
                    continue

                resolved_line, snippet = self._resolve_line_and_snippet(
                    entry, include_context=include_context
                )
                formatted = entry.format_summary(resolved_line)
                if snippet:
                    formatted = f"{formatted}\n{snippet}"
                results.append(formatted)

                if len(results) >= max_matches:
                    results.append(
                        f"\n[Truncated after reaching {max_matches} symbol matches. "
                        "Narrow your query, path, kind, or language filter.]"
                    )
                    break

            if not results:
                return f"No symbols matching query '{query}' found in ctags index."
            return "\n".join(results)
        except (re.error, OSError, CodebaseToolError) as err:
            return f"Error in search_symbols: {err}"

    def list_symbols(
        self,
        path: str = ".",
        kind: str = "",
        language: str = "",
        max_matches: int = MAX_FIND_FILES_DEFAULT,
    ) -> str:
        """Lists indexed symbols from the Ctags file under a file or directory.

        Use this to quickly map all declarations across a directory, package, module,
        or source file when a Ctags file is present. If no Ctags file exists yet,
        call `generate_ctags` first or fall back to `outline_file`.

        Args:
            path: Relative file or directory path to inspect. Defaults to '.'.
            kind: Optional symbol kind filter (e.g. 'class', 'function', 'variable').
            language: Optional language filter (e.g. 'Python', 'Go', 'Rust').
            max_matches: Maximum number of symbols to return before truncating.

        Returns:
            Formatted structural list of symbols sorted by file path and line number.
        """
        try:
            scope_target = self._resolve_safe_path(path)
            if not scope_target.exists():
                return f"Error: '{path}' does not exist."

            tags_file = self._find_tags_file()
            if tags_file is None:
                return (
                    "No Ctags file ('tags', '.tags', or '.git/tags') found in "
                    "repository root. Run `generate_ctags` to create one, or fall back "
                    "to `outline_file` / `find_files`."
                )

            entries = self._parse_tags_file(tags_file)
            matched: list[tuple[str, int, TagEntry]] = []

            for entry in entries:
                if not self._matches_scope_path(entry.file_path, scope_target):
                    continue
                if not entry.matches_kind(kind) or not entry.matches_language(language):
                    continue
                resolved_line, _ = self._resolve_line_and_snippet(
                    entry, include_context=False
                )
                matched.append((entry.file_path, resolved_line or 0, entry))

            if not matched:
                return f"No indexed symbols found under '{path}'."

            matched.sort(key=lambda item: (item[0], item[1], item[2].name))
            total = len(matched)
            trimmed = matched[:max_matches]

            lines = [f"### Indexed Symbols under `{path}` ({total} symbols):"]
            for _, resolved_line, entry in trimmed:
                lines.append(
                    entry.format_summary(resolved_line if resolved_line > 0 else None)
                )

            if total > len(trimmed):
                lines.append(f"\n[... and {total - len(trimmed)} more symbols]")

            return "\n".join(lines)
        except (OSError, CodebaseToolError) as err:
            return f"Error in list_symbols: {err}"
