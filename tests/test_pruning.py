from __future__ import annotations

from typing import Any, cast

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.genai import types

from rpi_agent.constants import MAX_TOOL_RESULT_PRESERVE_CHARS
from rpi_agent.pruning import ContextPruner, prune_contents

BIG = "x" * (MAX_TOOL_RESULT_PRESERVE_CHARS + 400)


def user(text: str) -> types.Content:
    return types.Content(role="user", parts=[types.Part(text=text)])


def model_text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


def model_call(call_id: str) -> types.Content:
    return types.Content(
        role="model",
        parts=[
            types.Part(
                function_call=types.FunctionCall(id=call_id, name="read_file", args={})
            )
        ],
    )


def tool_result(call_id: str, response: Any) -> types.Content:
    return types.Content(
        role="user",
        parts=[
            types.Part(
                function_response=types.FunctionResponse(
                    id=call_id, name="read_file", response=response
                )
            )
        ],
    )


def result_of(content: types.Content) -> Any:
    assert content.parts
    response = content.parts[0].function_response
    assert response is not None
    return response.response


def completed_turn(call_id: str, output: str = BIG) -> list[types.Content]:
    return [
        user(f"question {call_id}"),
        model_call(call_id),
        tool_result(call_id, {"result": output}),
        model_text("findings"),
    ]


def test_single_turn_is_untouched() -> None:
    contents = completed_turn("a")
    pruned, chars = prune_contents(contents)
    assert chars == 0
    assert pruned == contents
    assert pruned is not contents


def test_prunes_completed_earlier_turn_only() -> None:
    current = [user("now"), model_call("c"), tool_result("c", {"result": BIG})]
    contents = [*completed_turn("a"), *current]

    pruned, chars = prune_contents(contents)

    assert chars == len(BIG) - MAX_TOOL_RESULT_PRESERVE_CHARS
    stub = result_of(pruned[2])["result"]
    assert stub.startswith(
        f"[Tool output pruned for token optimization: {len(BIG)} chars."
    )
    # Pairing with the function call is preserved.
    part = pruned[2].parts[0] if pruned[2].parts else None
    assert part is not None and part.function_response is not None
    assert (part.function_response.id, part.function_response.name) == (
        "a",
        "read_file",
    )
    # The current turn keeps its raw output.
    assert result_of(pruned[-1]) == {"result": BIG}


def test_does_not_mutate_input() -> None:
    contents = [*completed_turn("a"), user("now")]
    prune_contents(contents)
    assert result_of(contents[2]) == {"result": BIG}


def test_interrupted_turn_is_preserved() -> None:
    # A turn that ended without model text (e.g. died on quota) keeps its output.
    interrupted = [user("research"), model_call("a"), tool_result("a", {"result": BIG})]
    contents = [*interrupted, user("continue")]

    pruned, chars = prune_contents(contents)
    assert chars == 0
    assert result_of(pruned[2]) == {"result": BIG}


def test_small_and_non_dict_and_other_key_outputs_kept() -> None:
    turn = [
        user("q"),
        model_call("a"),
        tool_result("a", {"result": "small"}),
        tool_result("b", {"other": BIG}),
        types.Content(role="model", parts=[]),
        model_text("done"),
    ]
    pruned, chars = prune_contents([*turn, user("next")])
    assert chars == 0
    assert result_of(pruned[2]) == {"result": "small"}
    assert pruned[3] is turn[3]


def test_respects_custom_threshold() -> None:
    contents = [*completed_turn("a", output="y" * 50), user("now")]
    _, chars = prune_contents(contents, max_chars=10)
    assert chars == 40


def test_context_pruner_callback_rewrites_request() -> None:
    pruner = ContextPruner()
    request = LlmRequest(contents=[*completed_turn("a"), user("now")])

    assert pruner(cast(CallbackContext, None), request) is None
    assert pruner.last_pruned_chars == len(BIG) - MAX_TOOL_RESULT_PRESERVE_CHARS
    assert result_of(request.contents[2])["result"].startswith("[Tool output pruned")


def test_context_pruner_resets_count() -> None:
    pruner = ContextPruner()
    pruner(
        cast(CallbackContext, None),
        LlmRequest(contents=[*completed_turn("a"), user("n")]),
    )
    pruner(cast(CallbackContext, None), LlmRequest(contents=[user("fresh")]))
    assert pruner.last_pruned_chars == 0
