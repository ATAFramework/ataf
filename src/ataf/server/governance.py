"""Governance — the server-side half of the two-layer enforcement model.

ATAF governs tool use in two layers (DESIGN.md §6):

  * **Layer 1 — prompt-level.** The catalog sent to the LLM carries each
    tool's ``status``; the prompt tells the LLM not to use anything that
    isn't ``AUTHORIZED``. This is cooperation, not enforcement.
  * **Layer 2 — server-level.** Even if the LLM ignores Layer 1 and tries
    to invoke a non-authorized tool, the server refuses. **This module is
    Layer 2.**

That's the whole job: given a ``tool_id``, decide whether it may be
invoked right now, and if not, raise a clear, typed error the endpoint
can turn into a ``403``.
"""

from __future__ import annotations

from .registry import Registry, ToolRow


class NotAuthorizedError(Exception):
    """Raised when a tool exists but may not be invoked in its current state.

    The endpoint maps this to a ``403`` carrying the stable code
    ``TOOL_NOT_AUTHORIZED`` and ``message``. The agent forwards the
    message to the LLM, which is instructed to stop calling tools.

    Attributes:
        code: Always ``"TOOL_NOT_AUTHORIZED"``.
        tool_id: The tool that was refused.
        status: The tool's status at refusal time (for the message/audit).
        message: Human-readable explanation.
    """

    code = "TOOL_NOT_AUTHORIZED"

    def __init__(self, tool_id: str, status: str) -> None:
        """Build the error.

        Args:
            tool_id: The tool that was refused.
            status: The tool's current status.
        """

        self.tool_id = tool_id
        self.status = status

        # Phrase the message by status so the LLM gets an actionable
        # reason rather than a bare code.
        if status == "PENDING_REVIEW":
            reason = "is pending human review"
        elif status == "UNAUTHORIZED":
            reason = "has been rejected by a human reviewer"
        else:
            reason = f"is not authorized (status {status})"
        self.message = f"Tool {tool_id!r} {reason}."
        super().__init__(f"{self.code}: {self.message}")


class Governance:
    """Decides whether a tool may be invoked (DESIGN.md §6, Layer 2).

    One instance per ATAF server, sharing the registry. Stateless beyond
    its single config flag, so it is safe to call from many request
    threads concurrently.
    """

    def __init__(self, registry: Registry, allow_pending_invocation: bool = False) -> None:
        """Construct the governance gate.

        Args:
            registry: The shared tool registry, used to read tool status.
            allow_pending_invocation: If True, tools in ``PENDING_REVIEW``
                may be invoked without human approval. Defaults to False
                (the safe production default from DESIGN.md §6). Handy to
                flip on for local development and demos.
        """

        self._registry = registry
        self._allow_pending = allow_pending_invocation

    def ensure_invokable(self, tool_id: str) -> ToolRow:
        """Return the tool row if it may be invoked, else raise.

        Args:
            tool_id: The tool the agent is trying to call.

        Returns:
            The tool's ``ToolRow`` when invocation is permitted.

        Raises:
            KeyError: If the tool_id is not in the registry (endpoint
                maps this to a ``404``).
            NotAuthorizedError: If the tool exists but its status forbids
                invocation (endpoint maps this to a ``403``).
        """

        # Look the tool up; a missing tool is a 404, distinct from a 403.
        row = self._registry.get(tool_id)
        if row is None:
            raise KeyError(f"unknown tool_id: {tool_id!r}")

        # AUTHORIZED is always invokable — the normal happy path.
        if row.status == "AUTHORIZED":
            return row

        # PENDING_REVIEW is invokable only when the server is explicitly
        # configured to allow it (dev/demo convenience).
        if row.status == "PENDING_REVIEW" and self._allow_pending:
            return row

        # Everything else (UNAUTHORIZED, or PENDING with the flag off) is
        # refused. The caller logs an invoke_denied event from the error.
        raise NotAuthorizedError(tool_id, row.status)
