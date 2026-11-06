"""Context pruning engine applied to outgoing model requests.

Bulky tool outputs from earlier, completed turns are replaced with a short stub
before each model call. The stored session history is never modified, so a
resumed session still has every tool result on disk.

A turn only counts as completed if it ended with a text response from the
model. Tool results from a turn that died part-way (e.g. on quota exhaustion)
are left intact: no synthesized findings exist for them yet.
"""

from __future__ import annotations

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from rpi_agent.constants import MAX_TOOL_RESULT_PRESERVE_CHARS


def _is_user_prompt(content: types.Content) -> bool:
    return content.role == "user" and any(p.text for p in content.parts or [])


def _is_model_text(content: types.Content) -> bool:
    return content.role == "model" and any(p.text for p in content.parts or [])


def _prune_part(part: types.Part, max_chars: int) -> tuple[types.Part, int]:
    response = part.function_response
    if response is None or not isinstance(response.response, dict):
        return part, 0

    output_text = str(response.response.get("result", ""))
    if len(output_text) <= max_chars:
        return part, 0

    stub = (
        f"[Tool output pruned for token optimization: {len(output_text)} chars. "
        f"Synthesized findings recorded in subsequent response. "
        f"Re-read with read_file if verification is required.]"
    )
    pruned = types.Part(
        function_response=types.FunctionResponse(
            id=response.id, name=response.name, response={"result": stub}
        )
    )
    return pruned, len(output_text) - max_chars


def prune_contents(
    contents: list[types.Content],
    max_chars: int = MAX_TOOL_RESULT_PRESERVE_CHARS,
) -> tuple[list[types.Content], int]:
    """Returns contents with large tool outputs of completed past turns stubbed.

    Args:
        contents: Conversation contents of an outgoing model request.
        max_chars: Tool outputs longer than this are pruned.

    Returns:
        Tuple of (new contents list, approximate characters pruned). Inputs are
        not mutated.
    """
    turn_starts = [i for i, c in enumerate(contents) if _is_user_prompt(c)]
    if len(turn_starts) < 2:
        return list(contents), 0

    # Every turn but the current (last) one, as [start, end) index ranges.
    boundaries = [*turn_starts, len(contents)]
    prunable: set[int] = set()
    for start, end in zip(boundaries[:-2], boundaries[1:-1], strict=True):
        if _is_model_text(contents[end - 1]):
            prunable.update(range(start, end))

    result: list[types.Content] = []
    pruned_chars = 0
    for index, content in enumerate(contents):
        if index not in prunable or not content.parts:
            result.append(content)
            continue

        new_parts: list[types.Part] = []
        content_saved = 0
        for part in content.parts:
            new_part, saved = _prune_part(part, max_chars)
            new_parts.append(new_part)
            content_saved += saved
        pruned_chars += content_saved
        result.append(
            content.model_copy(update={"parts": new_parts})
            if content_saved
            else content
        )

    return result, pruned_chars


class ContextPruner:
    """`before_model_callback` that prunes each outgoing request.

    Attributes:
        last_pruned_chars: Characters omitted from the most recent request.
    """

    def __init__(self, max_chars: int = MAX_TOOL_RESULT_PRESERVE_CHARS) -> None:
        self.max_chars = max_chars
        self.last_pruned_chars = 0

    def __call__(
        self, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """Prunes the request in place and lets the model call proceed."""
        llm_request.contents, self.last_pruned_chars = prune_contents(
            llm_request.contents, self.max_chars
        )
        return None
