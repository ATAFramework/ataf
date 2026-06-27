"""Tests for the Claude adapter's pure translation helpers.

These exercise the neutral<->Anthropic conversion without constructing a
real client or hitting the API.
"""

from types import SimpleNamespace

from ataf.client.llm_adapters.base import (
    AssistantMessage,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)
from ataf.client.llm_adapters.claude import (
    _from_anthropic_response,
    _to_anthropic_messages,
    _to_anthropic_tools,
)


def test_tools_exposed_by_tool_id() -> None:
    """Each tool is exposed to the model under its unique tool_id."""

    tools = [
        {
            "tool_id": "circle_area_v1",
            "name": "circle_area",
            "description": "area",
            "input_schema": {"type": "object"},
        }
    ]
    converted = _to_anthropic_tools(tools)
    assert converted[0]["name"] == "circle_area_v1"
    assert "circle_area" in converted[0]["description"]


def test_messages_translate_roles_and_tool_use() -> None:
    """User/assistant/tool-result messages map to Anthropic shapes."""

    messages = [
        UserMessage("do the thing"),
        AssistantMessage(
            text="calling tool",
            tool_calls=[ToolCall(id="t1", name="circle_area_v1", input={"radius": 1})],
        ),
        ToolResultMessage(tool_call_id="t1", content="3.14", is_error=False),
    ]
    out = _to_anthropic_messages(messages)

    # User text turn.
    assert out[0] == {"role": "user", "content": "do the thing"}
    # Assistant turn has a text block and a tool_use block.
    assert out[1]["role"] == "assistant"
    types = [b["type"] for b in out[1]["content"]]
    assert types == ["text", "tool_use"]
    # Tool result is a user message with a tool_result block.
    assert out[2]["role"] == "user"
    assert out[2]["content"][0]["type"] == "tool_result"
    assert out[2]["content"][0]["tool_use_id"] == "t1"


def test_consecutive_tool_results_merge() -> None:
    """Two tool results in a row merge into one user message."""

    messages = [
        ToolResultMessage(tool_call_id="a", content="1"),
        ToolResultMessage(tool_call_id="b", content="2"),
    ]
    out = _to_anthropic_messages(messages)
    assert len(out) == 1
    assert len(out[0]["content"]) == 2


def test_from_response_splits_text_and_tool_use() -> None:
    """A mixed Anthropic response normalizes into text + tool_calls."""

    # Fake the Anthropic response: a content list of typed blocks.
    response = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="let me compute"),
            SimpleNamespace(
                type="tool_use", id="t9", name="circle_area_v1", input={"radius": 11}
            ),
        ]
    )
    normalized = _from_anthropic_response(response)
    assert normalized.text == "let me compute"
    assert len(normalized.tool_calls) == 1
    assert normalized.tool_calls[0].name == "circle_area_v1"
    assert normalized.tool_calls[0].input == {"radius": 11}
