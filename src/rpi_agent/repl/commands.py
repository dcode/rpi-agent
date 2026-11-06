"""Expansion of RPI workflow slash commands into agent prompts."""

from __future__ import annotations

# command -> (prompt template with argument, prompt without argument)
WORKFLOW_COMMANDS: dict[str, tuple[str, str]] = {
    "/research": (
        "Please execute the 'rpi-research' skill for the following topic: {arg}",
        "Please trigger the 'rpi-research' skill.",
    ),
    "/plan": (
        "Please execute the 'rpi-plan' skill for: {arg}",
        "Please trigger the 'rpi-plan' skill.",
    ),
    "/iterate": (
        "Please execute the 'rpi-iterate' skill with this feedback: {arg}",
        "Please trigger the 'rpi-iterate' skill.",
    ),
    "/implement": (
        "Please execute the 'rpi-implement' skill for plan: {arg}",
        "Please trigger the 'rpi-implement' skill.",
    ),
}


def expand_command(raw: str) -> str:
    """Expands an RPI workflow slash command into a dispatch prompt.

    Args:
        raw: Stripped user input.

    Returns:
        Expanded prompt, or the input unchanged if it is not a workflow command.
    """
    for command, (with_arg, without_arg) in WORKFLOW_COMMANDS.items():
        if raw.startswith(command):
            arg = raw.removeprefix(command).strip()
            return with_arg.format(arg=arg) if arg else without_arg
    return raw
