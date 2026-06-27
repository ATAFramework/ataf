"""LLM adapter interface — the provider-neutral seam for the agent loop.

The agent loop never talks to a specific vendor SDK. It talks to an
:class:`LLMAdapter`, which translates a neutral conversation (built from
the dataclasses here) into one provider's API and back. This keeps the
loop testable (a fake adapter returns scripted responses) and makes adding
OpenAI/Gemini in v0.2 a matter of writing one more adapter.

Neutral conversation model:
  * The agent builds a list of :class:`Message` (user / assistant / tool
    result) plus a system prompt and a list of callable tools.
  * The adapter returns an :class:`LLMResponse` — the assistant's text plus
    any native tool calls it requested.

Tool identity: the agent passes catalog entries as the callable ``tools``;
each adapter exposes a tool to the model under its unique ``tool_id`` (not
the function ``name``, which can collide across versions), so a returned
:class:`ToolCall` ``name`` IS the ``tool_id`` to invoke.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One native tool invocation the LLM requested.

    Attributes:
        id: Provider-assigned call id, echoed back in the tool result so
            the model can correlate request and response.
        name: The tool to call — this is the ATAF ``tool_id``.
        input: The keyword arguments the model chose, as a dict.
    """

    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """A normalized single response from the model.

    Attributes:
        text: The assistant's text content (may be empty if it only made
            tool calls).
        tool_calls: Native tool calls the model requested this turn. Empty
            when the model answered in text (prose / proposal / decline).
        raw: The untouched provider response object, for debugging. Not
            used by the agent loop.
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None


# ----------------------------------------------------------------------
# Neutral conversation messages. The agent constructs these; the adapter
# translates them into the provider's message format.
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class UserMessage:
    """A user (or system-injected) text turn.

    Attributes:
        content: The text shown to the model — the task, or an agent note
            such as a deployment outcome.
    """

    content: str


@dataclass(frozen=True)
class AssistantMessage:
    """A prior assistant turn, replayed back into the conversation.

    Attributes:
        text: The assistant's text content that turn.
        tool_calls: Any tool calls it made that turn (so the provider can
            reconstruct the tool_use blocks the tool results answer to).
    """

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResultMessage:
    """The result of executing one of the assistant's tool calls.

    Attributes:
        tool_call_id: The :class:`ToolCall` ``id`` this answers.
        content: The result rendered as text (JSON or an error string).
        is_error: True if the invocation failed; lets the model react.
    """

    tool_call_id: str
    content: str
    is_error: bool = False


# A conversation turn is one of the three message kinds above.
Message = UserMessage | AssistantMessage | ToolResultMessage


class LLMAdapter(ABC):
    """Abstract base every concrete LLM adapter implements."""

    @abstractmethod
    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict],
    ) -> LLMResponse:
        """Produce the next assistant response for the conversation.

        Args:
            system: The system prompt for this turn (policy-aware).
            messages: The neutral conversation so far, oldest first.
            tools: Callable tools for this turn — ATAF catalog entries
                (dicts with ``tool_id``, ``name``, ``description``,
                ``input_schema``). The adapter exposes each to the model
                under its ``tool_id``.

        Returns:
            The model's normalized :class:`LLMResponse`.
        """

        raise NotImplementedError
