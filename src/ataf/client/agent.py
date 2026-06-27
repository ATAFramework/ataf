"""AtafAgent — the runtime tool-acquisition loop.

This is the agent the user actually drives. Given a task, it:

  1. fetches the current tool catalog from the ATAF server,
  2. builds a policy-aware system prompt (DESIGN.md §8.5) and asks the LLM,
  3. interprets the response as one of four shapes (DESIGN.md §3):
       - a native tool call          -> invoke it on the server, feed the
                                        result back, loop;
       - a <ATAF> proposal           -> deploy it, refresh the catalog, loop;
       - a structured decline code   -> terminate (no retry);
       - a prose answer              -> terminate (or, under a strict
                                        policy with no tool used, a hard
                                        protocol error).

The ``tool_policy`` argument (DESIGN.md §3.1) governs how the loop steers
the model and how it treats a prose answer.

The agent talks to the server over HTTP via an injected ``httpx.Client``
(tests point one at the in-process app through an ASGI transport, so the
loop is exercised end-to-end without a network).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx

from .llm_adapters.base import (
    AssistantMessage,
    LLMAdapter,
    Message,
    ToolResultMessage,
    UserMessage,
)
from .policy import (
    DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED,
    ToolPolicy,
)
from .prompt_template import build_system_prompt
from .tag_parser import classify_text_response


class AtafProtocolError(Exception):
    """The LLM broke the response contract for the active policy.

    Raised when a strict policy (``REQUIRE_NEWTOOL`` /
    ``USE_ONLY_EXISTINGTOOL``) is in force and the model answers a task in
    prose without using a tool, or proposes a tool under
    ``USE_ONLY_EXISTINGTOOL``. Per the design, the agent surfaces this to
    the caller rather than silently accepting it or re-prompting.
    """


class AtafServerError(Exception):
    """The ATAF server returned an unexpected, non-recoverable error."""


class Outcome(str, Enum):
    """How an agent run ended.

    Members:
        ANSWERED: The agent produced a final answer (prose, possibly
            derived from tool results).
        DECLINED: The LLM declined with a structured reason code.
        MAX_TURNS: The loop hit its turn cap without terminating.
    """

    ANSWERED = "ANSWERED"
    DECLINED = "DECLINED"
    MAX_TURNS = "MAX_TURNS"


@dataclass
class AgentResult:
    """The result of one :meth:`AtafAgent.run`.

    Attributes:
        outcome: How the run ended (see :class:`Outcome`).
        answer: The final text answer, when ``outcome`` is ``ANSWERED``.
        decline_code: The reason code, when ``outcome`` is ``DECLINED``.
        tools_proposed: ``tool_id`` of every tool deployed during the run.
        tools_invoked: ``tool_id`` of every tool invoked during the run.
        turns: How many LLM turns the run took.
    """

    outcome: Outcome
    answer: str | None = None
    decline_code: str | None = None
    tools_proposed: list[str] = field(default_factory=list)
    tools_invoked: list[str] = field(default_factory=list)
    turns: int = 0


@dataclass(frozen=True)
class _ProposeOutcome:
    """Internal result of submitting a proposal to the server.

    Attributes:
        status: ``"DEPLOYED"``, ``"WAIT"``, or ``"INVALID"``.
        tool_id: The new tool's id (only when ``DEPLOYED``).
        tool_status: The deployed tool's status (only when ``DEPLOYED``).
        message: A human-readable note (validation error, etc).
    """

    status: str
    tool_id: str | None = None
    tool_status: str | None = None
    message: str = ""


class AtafAgent:
    """Drives the LLM <-> ATAF acquisition loop for a single server."""

    def __init__(
        self,
        llm: LLMAdapter,
        *,
        server_url: str = "http://127.0.0.1:9123",
        http_client: httpx.Client | None = None,
        tool_policy: ToolPolicy = ToolPolicy.PREFER_NEWTOOL,
        allow_pending: bool = False,
        max_turns: int = 10,
    ) -> None:
        """Construct the agent.

        Args:
            llm: The LLM adapter to reason with.
            server_url: Base URL of the ATAF server. Ignored if
                ``http_client`` is supplied.
            http_client: A pre-built ``httpx.Client`` (tests inject one
                bound to the in-process app). If omitted, one is created
                against ``server_url``.
            tool_policy: How hard to steer the model toward tools, and how
                to treat a prose answer (DESIGN.md §3.1).
            allow_pending: If True, PENDING_REVIEW tools are offered to the
                model as callable (mirrors the server's
                ``allow_pending_invocation`` — only useful for demos).
            max_turns: Safety cap on LLM turns per run.
        """

        self._llm = llm
        self._policy = tool_policy
        self._allow_pending = allow_pending
        self._max_turns = max_turns

        # Own the HTTP client only if we created it, so close() doesn't
        # shut down a client the caller passed in.
        if http_client is not None:
            self._http = http_client
            self._owns_http = False
        else:
            self._http = httpx.Client(base_url=server_url, timeout=35.0)
            self._owns_http = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> AgentResult:
        """Run the acquisition loop for one task to completion.

        Args:
            task: The user's natural-language task.

        Returns:
            An :class:`AgentResult` describing how the run ended.

        Raises:
            AtafProtocolError: The model violated the policy contract.
            AtafServerError: The server returned an unrecoverable error.
        """

        # The conversation starts with the user's task. The system prompt
        # is rebuilt every turn from the freshly-fetched catalog.
        messages: list[Message] = [UserMessage(task)]
        result = AgentResult(outcome=Outcome.MAX_TURNS)

        for turn in range(1, self._max_turns + 1):
            result.turns = turn

            # Fetch the catalog and decide which tools are callable this
            # turn. GET /tools long-polls if a build is in progress, so a
            # concurrent build resolves here before we prompt.
            catalog = self._fetch_catalog()
            callable_tools = self._callable_tools(catalog)
            system = build_system_prompt(catalog, self._policy)

            # Ask the model.
            response = self._llm.complete(
                system=system, messages=messages, tools=callable_tools
            )
            messages.append(
                AssistantMessage(text=response.text, tool_calls=response.tool_calls)
            )

            # --- Response shape 1: native tool calls ---
            if response.tool_calls:
                # Invoke each requested tool and feed every result back, then
                # loop so the model can synthesize a final answer.
                for call in response.tool_calls:
                    text, is_error = self._invoke_tool(call.name, call.input)
                    result.tools_invoked.append(call.name)
                    messages.append(
                        ToolResultMessage(
                            tool_call_id=call.id, content=text, is_error=is_error
                        )
                    )
                continue

            # No tool calls: classify the text into propose / decline / prose.
            parsed = classify_text_response(response.text)

            # --- Response shape 2: propose a new tool ---
            if parsed.kind == "propose":
                # A proposal under USE_ONLY_EXISTINGTOOL is a contract
                # violation — surface it rather than deploying.
                if not self._policy.allows_proposing():
                    raise AtafProtocolError(
                        f"policy {self._policy.value} forbids proposing tools, "
                        "but the model emitted an <ATAF> proposal"
                    )

                note = self._handle_proposal(parsed.code or "", intent=task, result=result)
                # Re-prompt with the deployment outcome; the catalog refresh
                # at the top of the next iteration surfaces the new tool.
                messages.append(UserMessage(note))
                continue

            # --- Response shape 3: structured decline ---
            if parsed.kind == "decline":
                result.outcome = Outcome.DECLINED
                result.decline_code = parsed.decline_code
                return result

            # --- Response shape 4: prose answer ---
            # A prose answer is the final answer when a tool has been used
            # (it's the synthesis step) or when the policy permits answering
            # directly. Otherwise it violates a strict policy.
            if result.tools_invoked or self._policy.allows_prose_answer():
                result.outcome = Outcome.ANSWERED
                result.answer = parsed.text
                return result

            raise AtafProtocolError(
                f"policy {self._policy.value} forbids a direct prose answer "
                "without using a tool; the model answered: "
                f"{parsed.text[:200]!r}"
            )

        # Hit the turn cap without a terminal response.
        return result

    def close(self) -> None:
        """Close the HTTP client if this agent created it."""

        if self._owns_http:
            self._http.close()

    def __enter__(self) -> "AtafAgent":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Server interactions
    # ------------------------------------------------------------------

    def _fetch_catalog(self) -> list[dict]:
        """GET /tools and return the ``tools`` array.

        Raises:
            AtafServerError: On a non-200 response.
        """

        resp = self._http.get("/tools")
        if resp.status_code != 200:
            raise AtafServerError(f"GET /tools failed: HTTP {resp.status_code}")
        return resp.json().get("tools", [])

    def _callable_tools(self, catalog: list[dict]) -> list[dict]:
        """Select which catalog entries are offered to the model as callable.

        AUTHORIZED tools are always callable; PENDING_REVIEW tools are added
        only when ``allow_pending`` is set (mirroring the server config).

        Args:
            catalog: The full catalog from ``_fetch_catalog``.

        Returns:
            The subset the model may call this turn.
        """

        callable_set: list[dict] = []
        for tool in catalog:
            status = tool.get("status")
            if status == "AUTHORIZED":
                callable_set.append(tool)
            elif status == "PENDING_REVIEW" and self._allow_pending:
                callable_set.append(tool)
        return callable_set

    def _invoke_tool(self, tool_id: str, args: dict) -> tuple[str, bool]:
        """POST to a tool's invoke endpoint and return (text, is_error).

        A governance refusal (403) or execution failure is returned as an
        error tool-result rather than raised, so the model can react (e.g.
        decline) per the prompt rules.

        Args:
            tool_id: The tool to invoke (the model's tool-call name).
            args: The arguments the model supplied.

        Returns:
            A ``(content, is_error)`` pair for the tool-result message.
        """

        resp = self._http.post(f"/tools/{tool_id}/invoke", json={"args": args})

        # Success: hand the JSON result back as text.
        if resp.status_code == 200:
            return json.dumps(resp.json().get("result")), False

        # Any error: surface the server's error envelope to the model.
        body = _safe_json(resp)
        message = body.get("message") or f"HTTP {resp.status_code}"
        error = body.get("error") or "ERROR"
        return f"{error}: {message}", True

    def _handle_proposal(
        self, code: str, *, intent: str, result: AgentResult
    ) -> str:
        """Submit a proposal and return a note to feed back to the model.

        Args:
            code: The proposed Python source.
            intent: The agent's stated intent (the original task).
            result: The running result, updated with the new tool_id.

        Returns:
            A short natural-language note describing the outcome, to append
            as the next user turn.
        """

        outcome = self._propose_tool(code, intent=intent)

        if outcome.status == "DEPLOYED" and outcome.tool_id is not None:
            result.tools_proposed.append(outcome.tool_id)
            # If the deployed tool is still PENDING_REVIEW and we can't call
            # pending tools, tell the model so it declines cleanly next turn.
            if outcome.tool_status != "AUTHORIZED" and not self._allow_pending:
                return (
                    f"Tool '{outcome.tool_id}' was deployed but is "
                    f"{outcome.tool_status} and cannot be invoked until a "
                    "human approves it. If it is the only tool that fits, "
                    f"decline with {DECLINE_TOOL_EXISTS_BUT_UNAUTHORIZED}."
                )
            return (
                f"Tool '{outcome.tool_id}' was deployed (status "
                f"{outcome.tool_status}). The catalog has been refreshed; "
                "use it to complete the task."
            )

        if outcome.status == "WAIT":
            # Another agent is mid-build. The next GET /tools long-polls
            # until it finishes; just nudge the model to re-evaluate.
            return (
                "Another build is in progress. The catalog will refresh "
                "once it completes; re-evaluate with the updated tools."
            )

        # INVALID: the proposed code failed validation. Report the error so
        # the model can correct it or decline.
        return (
            f"The proposed tool was rejected: {outcome.message}. "
            "Fix the code and re-propose, or decline with a reason code."
        )

    def _propose_tool(self, code: str, *, intent: str) -> _ProposeOutcome:
        """POST /tools/propose and normalize the result.

        Args:
            code: The proposed Python source.
            intent: The agent's stated intent.

        Returns:
            A :class:`_ProposeOutcome`.

        Raises:
            AtafServerError: On a server fault (5xx other than handled).
        """

        resp = self._http.post(
            "/tools/propose", json={"intent": intent, "code": code}
        )

        # 200 -> deployed.
        if resp.status_code == 200:
            body = resp.json()
            return _ProposeOutcome(
                status="DEPLOYED",
                tool_id=body.get("tool_id"),
                tool_status=body.get("tool_status"),
            )

        # 202 -> another build holds the lock.
        if resp.status_code == 202:
            return _ProposeOutcome(status="WAIT")

        # 400 -> validation failure (recoverable: model can fix/decline).
        if resp.status_code == 400:
            body = _safe_json(resp)
            return _ProposeOutcome(
                status="INVALID",
                message=body.get("message") or "validation failed",
            )

        # Anything else (e.g. 500 TOOL_NOT_DEPLOYED) is a server fault.
        body = _safe_json(resp)
        raise AtafServerError(
            f"propose failed: HTTP {resp.status_code} "
            f"{body.get('error', '')} {body.get('message', '')}".strip()
        )


def _safe_json(response: httpx.Response) -> dict:
    """Return the response JSON as a dict, or an empty dict on parse error."""

    try:
        data = response.json()
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}
