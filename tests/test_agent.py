"""End-to-end agent-loop tests.

A scripted fake adapter feeds the loop deterministic LLM responses, while
the loop talks to a *real* in-process ATAF server through Starlette's
TestClient (which subclasses httpx.Client, so the agent accepts it as its
HTTP client). This exercises the whole loop — prompt, propose, deploy,
invoke, policy enforcement — without a network or a live LLM.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ataf.client.agent import (
    AtafAgent,
    AtafProtocolError,
    Outcome,
)
from ataf.client.llm_adapters.base import LLMAdapter, LLMResponse, ToolCall
from ataf.client.policy import (
    DECLINE_NOT_CAPABLE,
    DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED,
    ToolPolicy,
)
from ataf.server.main import create_app
from ataf.server.storage import StoragePaths


CIRCLE_AREA_CODE = '''def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius of the circle.
    """
    import math
    return math.pi * radius ** 2
'''

PROPOSAL_TEXT = f'<ATAF param="newtool">\n{CIRCLE_AREA_CODE}</ATAF>\n'


class ScriptedAdapter(LLMAdapter):
    """An adapter that replays a fixed list of responses, one per turn."""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, *, system, messages, tools) -> LLMResponse:
        # Record what the loop sent (so tests can assert on prompt/tools),
        # then return the next scripted response.
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        return self._responses.pop(0)


def _agent(tmp_path: Path, adapter: LLMAdapter, **kwargs) -> AtafAgent:
    """Wire an agent to a fresh in-process server via TestClient."""

    app = create_app(
        StoragePaths(root=tmp_path),
        allow_pending_invocation=kwargs.pop("server_allow_pending", False),
    )
    client = TestClient(app)
    return AtafAgent(adapter, http_client=client, **kwargs)


# ----------------------------------------------------------------------
# Prose answers + policy enforcement
# ----------------------------------------------------------------------

def test_prose_answer_under_model_decides(tmp_path: Path) -> None:
    """MODEL_DECIDES: a direct prose answer is accepted and returned."""

    adapter = ScriptedAdapter([LLMResponse(text="The area is about 380.13.")])
    agent = _agent(tmp_path, adapter, tool_policy=ToolPolicy.MODEL_DECIDES_NEWTOOL)

    result = agent.run("area of a circle radius 11")
    assert result.outcome == Outcome.ANSWERED
    assert "380.13" in result.answer


def test_prose_answer_under_require_is_protocol_error(tmp_path: Path) -> None:
    """REQUIRE: a prose answer with no tool used is a hard protocol error."""

    adapter = ScriptedAdapter([LLMResponse(text="The area is about 380.13.")])
    agent = _agent(tmp_path, adapter, tool_policy=ToolPolicy.REQUIRE_NEWTOOL)

    with pytest.raises(AtafProtocolError):
        agent.run("area of a circle radius 11")


def test_proposal_under_use_only_is_protocol_error(tmp_path: Path) -> None:
    """USE_ONLY_EXISTINGTOOL: proposing a tool is a hard protocol error."""

    adapter = ScriptedAdapter([LLMResponse(text=PROPOSAL_TEXT)])
    agent = _agent(tmp_path, adapter, tool_policy=ToolPolicy.USE_ONLY_EXISTINGTOOL)

    with pytest.raises(AtafProtocolError):
        agent.run("area of a circle radius 11")


# ----------------------------------------------------------------------
# Decline handling
# ----------------------------------------------------------------------

def test_decline_not_capable_terminates(tmp_path: Path) -> None:
    """A decline code ends the run immediately with DECLINED."""

    adapter = ScriptedAdapter(
        [LLMResponse(text=f"{DECLINE_NOT_CAPABLE}: can't build that.")]
    )
    agent = _agent(tmp_path, adapter, tool_policy=ToolPolicy.REQUIRE_NEWTOOL)

    result = agent.run("do something impossible")
    assert result.outcome == Outcome.DECLINED
    assert result.decline_code == DECLINE_NOT_CAPABLE


# ----------------------------------------------------------------------
# Propose -> pending -> decline (REQUIRE, approval required)
# ----------------------------------------------------------------------

def test_propose_then_pending_then_decline(tmp_path: Path) -> None:
    """REQUIRE with approval on: propose deploys PENDING, model then declines."""

    adapter = ScriptedAdapter(
        [
            # Turn 1: propose the tool.
            LLMResponse(text=PROPOSAL_TEXT),
            # Turn 2: sees its tool is pending review -> declines.
            LLMResponse(
                text=f"{DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED}: needs approval."
            ),
        ]
    )
    agent = _agent(tmp_path, adapter, tool_policy=ToolPolicy.REQUIRE_NEWTOOL)

    result = agent.run("area of a circle radius 11")
    assert result.outcome == Outcome.DECLINED
    assert result.decline_code == DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED
    # The tool was actually deployed along the way.
    assert result.tools_proposed == ["circle_area_v1"]


# ----------------------------------------------------------------------
# Full autonomous loop: propose -> use -> answer (allow_pending demo mode)
# ----------------------------------------------------------------------

def test_propose_then_invoke_then_answer(tmp_path: Path) -> None:
    """allow_pending demo: propose, invoke the new tool, synthesize an answer."""

    adapter = ScriptedAdapter(
        [
            # Turn 1: propose.
            LLMResponse(text=PROPOSAL_TEXT),
            # Turn 2: call the freshly-deployed (pending, but callable) tool.
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="circle_area_v1", input={"radius": 11})
                ],
            ),
            # Turn 3: synthesize the final answer from the tool result.
            LLMResponse(text="The area is 380.13."),
        ]
    )
    agent = _agent(
        tmp_path,
        adapter,
        tool_policy=ToolPolicy.REQUIRE_NEWTOOL,
        allow_pending=True,
        server_allow_pending=True,
    )

    result = agent.run("area of a circle radius 11")
    assert result.outcome == Outcome.ANSWERED
    assert result.tools_proposed == ["circle_area_v1"]
    assert result.tools_invoked == ["circle_area_v1"]
    assert "380.13" in result.answer


# ----------------------------------------------------------------------
# Using a pre-approved existing tool (USE_ONLY_EXISTINGTOOL)
# ----------------------------------------------------------------------

def test_use_existing_authorized_tool(tmp_path: Path) -> None:
    """USE_ONLY: call a pre-approved authorized tool and answer."""

    # Pre-deploy and approve circle_area_v1 directly on the server.
    app = create_app(StoragePaths(root=tmp_path))
    client = TestClient(app)
    client.post("/tools/propose", json={"intent": "seed", "code": CIRCLE_AREA_CODE})
    client.post("/admin/tools/circle_area_v1/approve")

    adapter = ScriptedAdapter(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(id="c1", name="circle_area_v1", input={"radius": 11})
                ],
            ),
            LLMResponse(text="The area is 380.13."),
        ]
    )
    agent = AtafAgent(
        adapter, http_client=client, tool_policy=ToolPolicy.USE_ONLY_EXISTINGTOOL
    )

    result = agent.run("area of a circle radius 11")
    assert result.outcome == Outcome.ANSWERED
    assert result.tools_invoked == ["circle_area_v1"]
    # Only the authorized tool was offered as callable.
    assert adapter.calls[0]["tools"][0]["tool_id"] == "circle_area_v1"


def test_invalid_proposal_is_reported_then_decline(tmp_path: Path) -> None:
    """A 400 validation failure is fed back; the model can then decline."""

    bad_code = "def f(x): return x\n"  # no types, no docstring
    adapter = ScriptedAdapter(
        [
            LLMResponse(text=f'<ATAF param="newtool">\n{bad_code}</ATAF>'),
            LLMResponse(text=f"{DECLINE_NOT_CAPABLE}: gave up."),
        ]
    )
    agent = _agent(tmp_path, adapter, tool_policy=ToolPolicy.PREFER_NEWTOOL)

    result = agent.run("do a thing")
    assert result.outcome == Outcome.DECLINED
    # Nothing was deployed (the proposal was rejected).
    assert result.tools_proposed == []
    # The validation error was surfaced to the model on turn 2.
    turn2_messages = adapter.calls[1]["messages"]
    assert any(
        "rejected" in m.content
        for m in turn2_messages
        if hasattr(m, "content") and isinstance(m.content, str)
    )
