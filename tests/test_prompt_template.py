"""Tests for the policy-aware prompt template."""

from ataf.client.policy import ToolPolicy
from ataf.client.prompt_template import build_system_prompt, render_catalog


SAMPLE_TOOLS = [
    {
        "tool_id": "circle_area_v1",
        "name": "circle_area",
        "status": "AUTHORIZED",
        "description": "Compute the area of a circle.",
        "input_schema": {"type": "object", "properties": {"radius": {"type": "number"}}},
    },
    {
        "tool_id": "square_area_v1",
        "name": "square_area",
        "status": "PENDING_REVIEW",
        "description": "Compute the area of a square.",
        "input_schema": {"type": "object", "properties": {"side": {"type": "number"}}},
    },
]


def test_render_catalog_lists_tools_with_status() -> None:
    """Each tool renders with its name, status, and params."""

    rendered = render_catalog(SAMPLE_TOOLS)
    assert "circle_area [AUTHORIZED]" in rendered
    assert "square_area [PENDING_REVIEW]" in rendered
    assert "params: radius" in rendered


def test_render_catalog_empty() -> None:
    """An empty catalog renders a friendly placeholder."""

    assert "none yet" in render_catalog([])


def test_prompt_includes_catalog_and_decline_codes() -> None:
    """The base prompt always carries the catalog and the decline codes."""

    prompt = build_system_prompt(SAMPLE_TOOLS, ToolPolicy.PREFER_NEWTOOL)
    assert "circle_area [AUTHORIZED]" in prompt
    assert "CANNOT_IMPLEMENT_TOOL_EXISTS_BUT_UNAUTHORIZED" in prompt
    assert "CANNOT_IMPLEMENT_NOT_CAPABLE" in prompt


def test_policy_rule_differs_per_policy() -> None:
    """Each policy injects a distinct rule-6 paragraph."""

    decides = build_system_prompt(SAMPLE_TOOLS, ToolPolicy.MODEL_DECIDES_NEWTOOL)
    prefer = build_system_prompt(SAMPLE_TOOLS, ToolPolicy.PREFER_NEWTOOL)
    require = build_system_prompt(SAMPLE_TOOLS, ToolPolicy.REQUIRE_NEWTOOL)
    use_only = build_system_prompt(SAMPLE_TOOLS, ToolPolicy.USE_ONLY_EXISTINGTOOL)

    assert "MAY answer directly in prose" in decides
    assert "prefer using an AUTHORIZED tool" in prefer
    assert "Do NOT answer a task request directly in prose" in require
    assert "may NOT propose" in use_only


def test_policy_predicates() -> None:
    """The policy predicates match the spec table."""

    assert ToolPolicy.PREFER_NEWTOOL.allows_proposing() is True
    assert ToolPolicy.USE_ONLY_EXISTINGTOOL.allows_proposing() is False
    assert ToolPolicy.MODEL_DECIDES_NEWTOOL.allows_prose_answer() is True
    assert ToolPolicy.REQUIRE_NEWTOOL.allows_prose_answer() is False
