"""Policy-guarded shell command executor."""

from __future__ import annotations

import subprocess

from rpi_agent.constants import SAFE_READ_COMMAND_PREFIXES
from rpi_agent.tools.base import RepoScopedTool


class ShellTool(RepoScopedTool):
    """Provides the `run_shell_command` tool."""

    def is_command_safe(self, command: str) -> bool:
        """Evaluates whether a command matches the safe read-only whitelist.

        Args:
            command: Shell command string to evaluate.

        Returns:
            True if command is whitelisted for safe execution without prompt.
        """
        trimmed = command.strip().lower()
        return any(trimmed.startswith(prefix) for prefix in SAFE_READ_COMMAND_PREFIXES)

    def run_shell_command(self, command: str) -> str:
        """Executes a shell command enforcing Option B safety policies.

        Args:
            command: Shell command string to execute in repository root.

        Returns:
            Combined stdout/stderr output from the executed command.
        """
        clean_cmd = command.strip()
        if not clean_cmd:
            return "Error: empty command requested."

        if not self.yolo and not self.is_command_safe(clean_cmd):
            print(f"\n[Command Request]: {clean_cmd}")
            choice = input("Execute this command? [y/N]: ").strip().lower()
            if choice not in {"y", "yes"}:
                return f"Command execution rejected by user: '{clean_cmd}'"

        try:
            result = subprocess.run(
                clean_cmd,
                cwd=self.repo_root,
                shell=True,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            output: list[str] = []
            if result.stdout:
                output.append(result.stdout)
            if result.stderr:
                output.append(f"[stderr]\n{result.stderr}")
            if result.returncode != 0:
                output.append(f"[Exit code: {result.returncode}]")

            return "\n".join(output) if output else "[Command finished with no output]"
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after 120s: '{clean_cmd}'"
        except (OSError, subprocess.SubprocessError) as err:
            return f"Error executing shell command: {err}"
