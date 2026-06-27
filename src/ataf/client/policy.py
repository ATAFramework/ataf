"""Tool-use policy — the client-side knob for steering the LLM toward tools.

Governance (server-side, ``server/governance.py``) decides whether a tool
*may* be invoked. **Policy** is the complementary client-side concern:
how hard the agent steers the LLM toward using or proposing a tool in the
first place, and what it does when the LLM answers in prose instead.

Policy shapes two things, both inside the client:
  1. The RULES block injected into the prompt template (DESIGN.md §8.5).
  2. How the agent loop reacts to each LLM response (DESIGN.md §3.1).

The ATAF server is completely unaware of policy — it only ever sees
catalog fetches, proposals, and invocations.
"""

from __future__ import annotations

from enum import Enum


class ToolPolicy(str, Enum):
    """How aggressively the agent steers the LLM toward tools.

    Subclasses ``str`` so a policy is JSON-serializable and prints as its
    value, which is convenient for logging and config files.

    Members:
        MODEL_DECIDES_NEWTOOL: No steering. The LLM may use an authorized
            tool, propose a new one, or answer directly in prose — all
            equally allowed.
        PREFER_NEWTOOL: The prompt steers the LLM to use or propose a tool
            for any operation where a precise/repeatable result matters,
            but a direct prose answer is still accepted (and may be logged
            as a "no-tool answer"). This is the default.
        REQUIRE_NEWTOOL: The LLM must use an authorized tool, propose a new
            one, or decline with a reason code. A direct prose answer to a
            task request is a protocol violation (hard error).
        USE_ONLY_EXISTINGTOOL: Runtime tool creation is disabled. The LLM
            may only call an existing authorized tool or decline. It may
            never propose. A direct prose answer is a protocol violation.
    """

    MODEL_DECIDES_NEWTOOL = "MODEL_DECIDES_NEWTOOL"
    PREFER_NEWTOOL = "PREFER_NEWTOOL"
    REQUIRE_NEWTOOL = "REQUIRE_NEWTOOL"
    USE_ONLY_EXISTINGTOOL = "USE_ONLY_EXISTINGTOOL"

    # ------------------------------------------------------------------
    # Convenience predicates — keep the policy semantics in one place so
    # the prompt builder and the agent loop never branch on raw members.
    # ------------------------------------------------------------------

    def allows_proposing(self) -> bool:
        """Whether this policy permits the LLM to propose new tools.

        Returns:
            True for every policy except ``USE_ONLY_EXISTINGTOOL``.
        """

        return self is not ToolPolicy.USE_ONLY_EXISTINGTOOL

    def allows_prose_answer(self) -> bool:
        """Whether a direct prose answer to a task is a valid response.

        Returns:
            True for the two lenient policies (``MODEL_DECIDES_NEWTOOL``,
            ``PREFER_NEWTOOL``); False for the strict two, where prose is
            a protocol error.
        """

        return self in (
            ToolPolicy.MODEL_DECIDES_NEWTOOL,
            ToolPolicy.PREFER_NEWTOOL,
        )


# ----------------------------------------------------------------------
# Structured decline reason codes (DESIGN.md §3, Response 3).
# These are emitted by the LLM and recognized by the tag parser; the agent
# treats any of them as a terminal decline and halts without retrying.
# ----------------------------------------------------------------------

# A matching tool already exists but is PENDING_REVIEW or UNAUTHORIZED.
# The LLM declines rather than proposing a duplicate. This is also the
# correct terminal state after the LLM proposes a tool and, on the
# refreshed re-prompt, sees its own just-built tool is still pending.
DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED = "CANNOT_IMPLEMENT_TOOL_EXISTS_BUT_UNAUTHORIZED"

# No suitable tool exists, proposing is allowed, but the LLM cannot write
# the tool.
DECLINE_NOT_CAPABLE = "CANNOT_IMPLEMENT_NOT_CAPABLE"

# USE_ONLY_EXISTINGTOOL: proposing is disabled and no AUTHORIZED tool fits.
DECLINE_NO_AUTHORIZED_TOOL = "CANNOT_IMPLEMENT_NO_AUTHORIZED_TOOL"

# The full set, for membership checks by the tag parser.
DECLINE_CODES = frozenset(
    {
        DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED,
        DECLINE_NOT_CAPABLE,
        DECLINE_NO_AUTHORIZED_TOOL,
    }
)
