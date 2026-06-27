"""Claude adapter — the v0.1 LLM adapter, backed by the Anthropic SDK.

Translates the neutral conversation (``base.py``) into Anthropic's Messages
API and normalizes the reply back into an :class:`LLMResponse`. This is the
only adapter in v0.1; OpenAI and Gemini adapters arrive in v0.2.

Tool exposure: each ATAF catalog entry is exposed to Claude under its
unique ``tool_id`` (function names can collide across versions, but
Anthropic tool names must be unique), so a returned tool call's ``name``
is exactly the ``tool_id`` the agent should invoke.
"""

from __future__ import annotations

from typing import Any

from .base import (
    AssistantMessage,
    LLMAdapter,
    LLMResponse,
    Message,
    ToolCall,
    ToolResultMessage,
    UserMessage,
)


# Default model for v0.1. The latest, most capable Claude model is a sound
# default for code generation; callers can override in the constructor.
_DEFAULT_MODEL = "claude-opus-4-8"


class ClaudeAdapter(LLMAdapter):
    """LLM adapter for Anthropic's Claude models."""

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        max_tokens: int = 2048,
        client: Any = None,
        api_key: str | None = None,
    ) -> None:
        """Construct the adapter.

        Args:
            model: Anthropic model id. Defaults to the latest Claude.
            max_tokens: Cap on the response length per turn.
            client: A pre-built ``anthropic.Anthropic`` client. Mostly for
                tests/dependency-injection; if omitted, one is created.
            api_key: API key passed to a freshly created client. If both
                ``client`` and ``api_key`` are omitted, the SDK picks up
                ``ANTHROPIC_API_KEY`` from the environment.
        """

        self._model = model
        self._max_tokens = max_tokens

        # Build a client lazily-ish: only import the SDK if we actually
        # need to construct one (so tests that inject a client or use a
        # different adapter don't require the package to be importable).
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Call Claude and normalize the reply. See :meth:`LLMAdapter.complete`."""

        # Translate the neutral conversation and tools into Anthropic shapes.
        anthropic_messages = _to_anthropic_messages(messages)
        anthropic_tools = _to_anthropic_tools(tools)

        # Anthropic rejects an empty `tools` list, so omit the kwarg when
        # there are no callable tools this turn.
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": system,
            "messages": anthropic_messages,
        }
        if anthropic_tools:
            request["tools"] = anthropic_tools

        response = self._client.messages.create(**request)

        # Split the response content into text and tool_use blocks.
        return _from_anthropic_response(response)


# ----------------------------------------------------------------------
# Translation helpers (module-level so they're unit-testable on their own)
# ----------------------------------------------------------------------

def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    """Convert ATAF catalog entries into Anthropic tool definitions.

    Each tool is exposed under its ``tool_id`` (unique) rather than its
    function ``name`` (which can collide), with the friendly name folded
    into the description.

    Args:
        tools: ATAF catalog entries (already filtered to the callable set).

    Returns:
        A list of Anthropic tool definition dicts.
    """

    definitions: list[dict] = []
    for tool in tools:
        definitions.append(
            {
                "name": tool["tool_id"],
                "description": f"{tool.get('name', '')}: {tool.get('description', '')}",
                "input_schema": tool.get("input_schema", {"type": "object"}),
            }
        )
    return definitions


def _to_anthropic_messages(messages: list[Message]) -> list[dict]:
    """Convert neutral messages into Anthropic's message list.

    Consecutive tool results are merged into a single ``user`` message, as
    Anthropic expects all tool_result blocks answering one assistant turn
    to arrive together.

    Args:
        messages: The neutral conversation.

    Returns:
        Anthropic-formatted message dicts.
    """

    out: list[dict] = []
    for message in messages:
        if isinstance(message, UserMessage):
            out.append({"role": "user", "content": message.content})

        elif isinstance(message, AssistantMessage):
            # An assistant turn is a text block (if any) followed by one
            # tool_use block per tool call.
            content: list[dict] = []
            if message.text:
                content.append({"type": "text", "text": message.text})
            for call in message.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.input,
                    }
                )
            out.append({"role": "assistant", "content": content})

        elif isinstance(message, ToolResultMessage):
            # A tool_result block lives in a user message. Merge it into the
            # previous user message if that was also tool results.
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
                "is_error": message.is_error,
            }
            if (
                out
                and out[-1]["role"] == "user"
                and isinstance(out[-1]["content"], list)
            ):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

    return out


def _from_anthropic_response(response: Any) -> LLMResponse:
    """Normalize an Anthropic Messages response into an :class:`LLMResponse`.

    Args:
        response: The object returned by ``client.messages.create``.

    Returns:
        The normalized response (text + tool calls).
    """

    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []

    # The response content is a list of blocks; collect text and tool_use.
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "tool_use":
            tool_calls.append(
                ToolCall(id=block.id, name=block.name, input=dict(block.input))
            )

    return LLMResponse(
        text="".join(text_parts),
        tool_calls=tool_calls,
        raw=response,
    )
