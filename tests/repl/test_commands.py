from __future__ import annotations

import pytest

from rpi_agent.repl.commands import expand_command


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "/research auth flow",
            "Please execute the 'rpi-research' skill for the following topic: auth flow",
        ),
        ("/research", "Please trigger the 'rpi-research' skill."),
        ("/plan caching", "Please execute the 'rpi-plan' skill for: caching"),
        ("/plan   ", "Please trigger the 'rpi-plan' skill."),
        (
            "/iterate drop phase 2",
            "Please execute the 'rpi-iterate' skill with this feedback: drop phase 2",
        ),
        ("/iterate", "Please trigger the 'rpi-iterate' skill."),
        (
            "/implement thoughts/plans/x.md",
            "Please execute the 'rpi-implement' skill for plan: thoughts/plans/x.md",
        ),
        ("/implement", "Please trigger the 'rpi-implement' skill."),
    ],
)
def test_workflow_commands(raw: str, expected: str) -> None:
    assert expand_command(raw) == expected


@pytest.mark.parametrize("raw", ["hello", "/unknown thing", "what is /plan?"])
def test_other_input_passes_through(raw: str) -> None:
    assert expand_command(raw) == raw
