"""Prompt template — builds the system prompt sent to the LLM each turn.

This is the canonical ATAF prompt from DESIGN.md §8.5, made policy-aware:
the fixed RULES block is the same for everyone, and a per-policy paragraph
(rule 6) is appended based on the active :class:`ToolPolicy`.

The prompt carries the current tool catalog (with status flags) so the LLM
knows what already exists, the tag protocol for proposing new tools, and
the structured decline codes for giving up cleanly.
"""

from __future__ import annotations

from .policy import ToolPolicy


# The policy-specific "rule 6" appended to the RULES block. Keyed by
# ToolPolicy so the agent never branches on policy when building a prompt.
_POLICY_RULES: dict[ToolPolicy, str] = {
    ToolPolicy.MODEL_DECIDES_NEWTOOL: (
        "6. You MAY answer directly in prose if that is the best response."
    ),
    ToolPolicy.PREFER_NEWTOOL: (
        "6. For any calculation, data lookup, or operation where a precise "
        "or repeatable result matters, prefer using an AUTHORIZED tool or "
        "proposing a new one over answering from your own reasoning. You "
        "may still answer directly for purely conversational replies."
    ),
    ToolPolicy.REQUIRE_NEWTOOL: (
        "6. You MUST use an AUTHORIZED tool, propose a new tool, or decline "
        "with a reason code. Do NOT answer a task request directly in prose."
    ),
    ToolPolicy.USE_ONLY_EXISTINGTOOL: (
        "6. You MUST use an AUTHORIZED tool or decline. You may NOT propose "
        "new tools. If no AUTHORIZED tool fits, decline with "
        "CANNOT_IMPLEMENT_NO_AUTHORIZED_TOOL."
    ),
}


# The fixed body of the prompt. ``{catalog}`` and ``{policy_rule}`` are
# filled in per turn; everything else is constant across turns and policies.
_TEMPLATE = """\
You are an agent with access to a dynamic toolset managed by ATAF.

CURRENT TOOLS:
{catalog}

RULES:
1. To use an AUTHORIZED tool, call it in your native tool-use format.
2. Tools with status PENDING_REVIEW or UNAUTHORIZED are unavailable. Do
   NOT propose a duplicate of a non-AUTHORIZED tool that matches the
   user's request. Instead, decline with
   CANNOT_IMPLEMENT_TOOL_EXISTS_BUT_UNAUTHORIZED.
3. To propose a new tool (when allowed), emit a single function inside
   <ATAF param="newtool"> ... </ATAF> tags. Constraints:
   - Type-annotated parameters and return value.
   - Google-style or NumPy-style docstring.
   - Only Python standard-library imports (no external packages).
   - Pure function preferred; no global state.
4. If a tool invocation returns TOOL_NOT_AUTHORIZED, stop calling tools
   and decline with CANNOT_IMPLEMENT_TOOL_EXISTS_BUT_UNAUTHORIZED.
5. To decline, respond with exactly one reason code on its own line:
   - CANNOT_IMPLEMENT_TOOL_EXISTS_BUT_UNAUTHORIZED — a matching tool
     exists but is not AUTHORIZED (do not duplicate it).
   - CANNOT_IMPLEMENT_NOT_CAPABLE — no tool fits and you cannot write one.
   - CANNOT_IMPLEMENT_NO_AUTHORIZED_TOOL — proposing is disabled and no
     AUTHORIZED tool fits.
{policy_rule}\
"""


def render_catalog(tools: list[dict]) -> str:
    """Render the tool catalog into a compact, LLM-readable list.

    Args:
        tools: The ``tools`` array from a ``GET /tools`` response. Each item
            is a dict with at least ``name``, ``status``, ``description``,
            and ``input_schema``.

    Returns:
        A multi-line string, one line per tool, or a placeholder when the
        catalog is empty.
    """

    # An explicit "none" line is friendlier to the model than a blank.
    if not tools:
        return "  (none yet — the catalog is empty)"

    lines: list[str] = []
    for tool in tools:
        # Surface name, status, and description so the LLM can both pick an
        # AUTHORIZED tool and avoid duplicating a pending/rejected one.
        name = tool.get("name", "?")
        status = tool.get("status", "?")
        description = tool.get("description", "")
        lines.append(f"  - {name} [{status}]: {description}")

        # Include the parameter names so the LLM knows the call shape
        # without us having to dump the whole JSON schema.
        properties = tool.get("input_schema", {}).get("properties", {})
        if properties:
            params = ", ".join(properties.keys())
            lines.append(f"      params: {params}")

    return "\n".join(lines)


def build_system_prompt(tools: list[dict], policy: ToolPolicy) -> str:
    """Build the full system prompt for one turn.

    Args:
        tools: The current catalog (``tools`` array from ``GET /tools``).
        policy: The active tool-use policy; selects the rule-6 paragraph.

    Returns:
        The complete system prompt string.
    """

    # Compose the catalog and the policy-specific rule into the template.
    return _TEMPLATE.format(
        catalog=render_catalog(tools),
        policy_rule=_POLICY_RULES[policy],
    )
