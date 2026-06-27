"""Wire protocol — Pydantic models for every request and response shape.

This module is the **lingua franca between the ATAF server and its
clients (agents)**. Every HTTP endpoint accepts and returns one of the
models defined here. The shapes mirror DESIGN.md §4 one-to-one.

Why a dedicated protocol module:
  * Single source of truth for request/response shapes. Both the server
    and the client library import from here.
  * Pydantic validates inbound and outbound payloads automatically,
    so the rest of the server can assume well-formed data.
  * FastAPI uses these models to auto-generate OpenAPI docs at
    /docs and /openapi.json.

Tool status values are defined in `governance.py` and re-exported here
as a simple `Literal` so callers don't need two imports.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Tool status — three-state lifecycle (see DESIGN.md §6).
# ----------------------------------------------------------------------
# PENDING_REVIEW: just deployed, awaiting human approval.
# AUTHORIZED:     human approved, invokable.
# UNAUTHORIZED:   human rejected; kept in registry for audit, never invokable.
ToolStatus = Literal["PENDING_REVIEW", "AUTHORIZED", "UNAUTHORIZED"]


# ----------------------------------------------------------------------
# Tool catalog entry — the public-facing description of one tool.
# Returned inside ToolCatalogResponse to both clients and the LLM.
# ----------------------------------------------------------------------
class ToolDescriptor(BaseModel):
    """A single tool's metadata as seen by an agent or LLM.

    Attributes:
        tool_id: Stable unique identifier for this tool. Used as the
            filename of the on-disk code (`tools/{tool_id}.py`) and as
            the URL path segment for invocation.
        name: The function name the LLM gave when it proposed the tool
            (e.g., ``circle_area``). Multiple tools may share a name
            across the registry; ``tool_id`` is what disambiguates them.
        description: First line of the tool's docstring. Short,
            human-readable, suitable for the LLM's tool catalog.
        input_schema: JSON schema describing the tool's input arguments.
            Generated from the function signature + docstring at
            deploy time. Shape mirrors OpenAPI parameter schemas and
            matches Anthropic's tool-use ``input_schema`` field.
        invoke_uri: Relative URI to POST args to in order to call the
            tool (e.g., ``/tools/circle_area_v1/invoke``).
        status: Current governance flag — see ``ToolStatus``.
        created_at: ISO-8601 UTC timestamp of original deployment.
    """

    tool_id: str
    name: str
    description: str
    input_schema: dict[str, Any]
    invoke_uri: str
    status: ToolStatus
    created_at: str


# ----------------------------------------------------------------------
# GET /tools response — the tool catalog.
# May be returned immediately, or after a long-poll wait if a build
# is in progress when the request arrives.
# ----------------------------------------------------------------------
class ToolCatalogResponse(BaseModel):
    """Response body for ``GET /tools`` and the long-poll endpoint.

    Attributes:
        tools: All tools currently in the registry, regardless of
            status. Filtering is the LLM's job, not the server's.
        build_in_progress: True only in the rare race where the
            response is being serialized at the exact moment a new
            build starts. Mostly informational.
        catalog_version: Monotonically increasing integer that bumps
            on every deploy / approve / reject. Clients may use this
            to detect a no-op refresh and skip an LLM round-trip.
    """

    tools: list[ToolDescriptor]
    build_in_progress: bool = False
    catalog_version: int


# ----------------------------------------------------------------------
# POST /tools/propose request — agent submits new tool code.
# ----------------------------------------------------------------------
class ProposeToolRequest(BaseModel):
    """Body for ``POST /tools/propose``.

    Attributes:
        intent: A short free-text description of what the agent was
            trying to do when the LLM proposed this tool. Stored on
            the tool row so human reviewers can understand the context.
            Example: "compute area of a circle for a geometry question".
        code: The complete Python source of the proposed tool. Must
            contain exactly one top-level function definition with
            type-annotated parameters, an annotated return type, and a
            Google-style or NumPy-style docstring. Only Python stdlib
            imports are allowed in v0.1.
    """

    intent: str = Field(min_length=1, max_length=2000)
    code: str = Field(min_length=1)


# ----------------------------------------------------------------------
# POST /tools/propose responses — three possible shapes.
# ----------------------------------------------------------------------
class ProposeDeployedResponse(BaseModel):
    """Returned when a proposal was built and deployed immediately.

    HTTP status: 200 OK.

    Attributes:
        status: Always the literal string ``"DEPLOYED"``.
        tool_id: The newly assigned identifier (e.g., ``circle_area_v1``).
        input_schema: The JSON schema of the new tool's input.
        invoke_uri: Where to POST args to call the new tool.
        tool_status: Will be ``"PENDING_REVIEW"`` for any newly deployed
            tool. Included explicitly so the client doesn't need to
            re-fetch the catalog to know whether it can immediately invoke.
        catalog_version: The new catalog version after this deploy.
    """

    status: Literal["DEPLOYED"] = "DEPLOYED"
    tool_id: str
    input_schema: dict[str, Any]
    invoke_uri: str
    tool_status: ToolStatus
    catalog_version: int


class ProposeWaitResponse(BaseModel):
    """Returned when another agent is already mid-build.

    HTTP status: 202 Accepted.

    The client should poll ``poll_url`` (long-poll) and treat the
    eventual response as a tool-list refresh — meaning it should
    re-prompt the LLM with the updated catalog and let the LLM decide
    whether the just-deployed tool covers its need, or whether it still
    wants to propose its own.

    Attributes:
        status: Always the literal string ``"WAIT"``.
        poll_url: The relative URL to long-poll.
        eta_seconds: Best-guess wait time. Advisory only.
    """

    status: Literal["WAIT"] = "WAIT"
    poll_url: str
    eta_seconds: int


# ----------------------------------------------------------------------
# POST /tools/{tool_id}/invoke request and responses.
# ----------------------------------------------------------------------
class InvokeToolRequest(BaseModel):
    """Body for ``POST /tools/{tool_id}/invoke``.

    Attributes:
        args: Keyword arguments to pass to the tool's function. Must
            match the tool's JSON schema. Validated by the server
            against the stored schema before invocation.
    """

    args: dict[str, Any] = Field(default_factory=dict)


class InvokeToolResponse(BaseModel):
    """Returned on a successful tool invocation.

    HTTP status: 200 OK.

    Attributes:
        result: Whatever the tool's function returned, JSON-serialized.
            Tools that return non-JSON-serializable values will fail at
            the executor layer with TOOL_EXECUTION_ERROR.
    """

    result: Any


class ErrorResponse(BaseModel):
    """Generic error envelope returned for non-2xx responses.

    Attributes:
        error: A machine-readable error code, e.g.
            ``"TOOL_NOT_AUTHORIZED"``, ``"TOOL_EXECUTION_ERROR"``,
            ``"VALIDATION_FAILED"``, ``"TOOL_NOT_DEPLOYED"`` (valid code
            that the server failed to deploy — retryable).
        message: Human-readable explanation. Safe to surface to an LLM
            as a tool-result so it can decide how to handle the failure.
    """

    error: str
    message: str
