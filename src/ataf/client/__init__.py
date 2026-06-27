"""ATAF client library — the agent loop and its building blocks.

Public API:

    from ataf.client import AtafAgent, ToolPolicy
    from ataf.client.llm_adapters.claude import ClaudeAdapter

    agent = AtafAgent(ClaudeAdapter(), server_url="http://127.0.0.1:9123",
                      tool_policy=ToolPolicy.PREFER_NEWTOOL)
    result = agent.run("What is the area of a circle with radius 11?")
"""

from .agent import (
    AgentResult,
    AtafAgent,
    AtafProtocolError,
    AtafServerError,
    Outcome,
)
from .policy import (
    DECLINE_NO_AUTHORIZED_TOOL,
    DECLINE_NOT_CAPABLE,
    DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED,
    ToolPolicy,
)

__all__ = [
    "AtafAgent",
    "AgentResult",
    "Outcome",
    "AtafProtocolError",
    "AtafServerError",
    "ToolPolicy",
    "DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED",
    "DECLINE_NOT_CAPABLE",
    "DECLINE_NO_AUTHORIZED_TOOL",
]
