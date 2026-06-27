"""ATAF server — the FastAPI application and its endpoints.

This module wires the inner machinery (registry, build lock, builder,
executor, governance, event log) into the HTTP surface described in
DESIGN.md §4:

    GET  /tools                       — fetch the catalog (long-polls
                                        during a build)
    POST /tools/propose               — submit new tool code
    GET  /tools/propose/wait/{token}  — long-poll a build in progress
    POST /tools/{tool_id}/invoke      — call a deployed tool
    POST /admin/tools/{tool_id}/approve | /reject  — human review

Because the app is built on FastAPI + Pydantic, the interactive docs at
``/docs`` and the machine-readable OpenAPI spec at ``/openapi.json`` are
generated automatically from the models in ``protocol.py``.

Invocation routing note: rather than hot-registering a separate route per
tool, we expose a single parametrized ``/tools/{tool_id}/invoke`` route
and look the tool up at call time. From the client's perspective each
tool still has its own ``invoke_uri``; this is purely a simpler, safer
implementation of the same contract.

Error contract: all non-2xx responses use the ``ErrorResponse`` envelope
(``{"error": <code>, "message": <text>}``) — the custom envelope ATAF
keeps for v0.1 (LLM-friendly: ``message`` goes straight back as a
tool-result).
"""

from __future__ import annotations

import os
import secrets

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .builder import Builder, CodeValidationError, DeploymentError
from .eventlog import DeploymentEventLog
from .executor import Executor, ToolExecutionError
from .governance import Governance, NotAuthorizedError
from .lock import BuildLock
from .protocol import (
    ErrorResponse,
    InvokeToolRequest,
    InvokeToolResponse,
    ProposeDeployedResponse,
    ProposeToolRequest,
    ProposeWaitResponse,
    ToolCatalogResponse,
    ToolDescriptor,
)
from .registry import IntegrityError, Registry
from .storage import StoragePaths, default_storage


# How long a long-poll waits for an in-flight build before giving up.
# The client is expected to retry on the 408 (DESIGN.md §5).
_LONG_POLL_TIMEOUT_SECONDS = 30.0


def create_app(
    paths: StoragePaths | None = None,
    *,
    allow_pending_invocation: bool = False,
    tool_timeout_seconds: float = 5.0,
) -> FastAPI:
    """Build and return a fully-wired ATAF FastAPI application.

    All shared components are created here and captured by the route
    handlers as closure state, so there are no module-level globals and
    tests can spin up an isolated app over a temp directory.

    Args:
        paths: Storage paths for this server's data dir. Defaults to
            ``./ataf_data`` resolved to an absolute path.
        allow_pending_invocation: If True, tools in ``PENDING_REVIEW`` may
            be invoked without approval. Convenient for demos; unsafe for
            real deployments (DESIGN.md §6).
        tool_timeout_seconds: Per-call wall-clock budget for tool
            execution.

    Returns:
        A configured ``FastAPI`` instance ready to serve.
    """

    # Resolve and create the data directory tree if the caller didn't
    # pass explicit paths.
    if paths is None:
        paths = default_storage(os.environ.get("ATAF_DATA_DIR", "ataf_data"))
    paths.ensure_exists()

    # --- Build the shared component graph (one of each, per server) ---
    registry = Registry(paths)
    registry.initialize()

    # On startup, re-verify every tool's code against its stored hash.
    # Any tampered/missing file is flipped to UNAUTHORIZED (never deleted).
    flipped = registry.verify_integrity()

    event_log = DeploymentEventLog(paths.deployment_log)
    build_lock = BuildLock()
    builder = Builder(registry, event_log)
    executor = Executor(registry, timeout_seconds=tool_timeout_seconds)
    governance = Governance(
        registry, allow_pending_invocation=allow_pending_invocation
    )

    # Record any startup integrity flips so the audit trail explains why a
    # previously-authorized tool came back UNAUTHORIZED after a restart.
    for tool_id, reason in flipped:
        event_log.record("integrity_flip", tool_id=tool_id, reason=reason)

    app = FastAPI(
        title="ATAF — Agent Tool Acquisition Framework",
        version="0.1.0.dev0",
        description=(
            "Agents acquire new tools at runtime. The LLM writes a tool, "
            "ATAF deploys it as a live endpoint, a human approves it, and "
            "every agent can then call it."
        ),
    )

    # ------------------------------------------------------------------
    # Helpers (closures over the component graph above)
    # ------------------------------------------------------------------

    def _catalog_response() -> ToolCatalogResponse:
        """Snapshot the registry into a wire catalog response."""

        # Convert each stored row into a public ToolDescriptor. The stored
        # schema is JSON text; parse it back into a dict for the response.
        descriptors: list[ToolDescriptor] = []
        for row in registry.list_all():
            descriptors.append(
                ToolDescriptor(
                    tool_id=row.tool_id,
                    name=row.name,
                    description=row.description,
                    input_schema=_loads_schema(row.schema_json),
                    invoke_uri=f"/tools/{row.tool_id}/invoke",
                    status=row.status,
                    created_at=row.created_at,
                )
            )

        return ToolCatalogResponse(
            tools=descriptors,
            build_in_progress=build_lock.is_building(),
            catalog_version=registry.catalog_version,
        )

    def _error(status_code: int, code: str, message: str) -> JSONResponse:
        """Build a JSONResponse wrapping the standard error envelope."""

        body = ErrorResponse(error=code, message=message)
        return JSONResponse(status_code=status_code, content=body.model_dump())

    # ------------------------------------------------------------------
    # GET /tools — fetch the catalog (long-polls during a build)
    # ------------------------------------------------------------------
    @app.get("/tools", response_model=ToolCatalogResponse)
    def get_tools() -> ToolCatalogResponse:
        """Return the current tool catalog.

        If a build is in progress, block (long-poll) until it finishes,
        then return the refreshed catalog (DESIGN.md §4).
        """

        # Only wait if a build is actually in flight; otherwise return now.
        if build_lock.is_building():
            build_lock.wait_for_release(timeout=_LONG_POLL_TIMEOUT_SECONDS)
        return _catalog_response()

    # ------------------------------------------------------------------
    # POST /tools/propose — submit new tool code
    # ------------------------------------------------------------------
    @app.post("/tools/propose")
    def propose_tool(request: ProposeToolRequest):
        """Validate, build, and deploy a proposed tool.

        Returns ``200 DEPLOYED`` on success, ``202 WAIT`` if another build
        holds the lock, ``400`` on validation failure, or ``500`` if valid
        code could not be deployed.
        """

        # Try to take the global build lock without blocking. If another
        # build holds it, tell the agent to long-poll instead of queueing
        # here (DESIGN.md §5).
        token = build_lock.acquire(no_wait=True)
        if token is None:
            wait = ProposeWaitResponse(
                poll_url=f"/tools/propose/wait/{secrets.token_hex(8)}",
                eta_seconds=5,
            )
            return JSONResponse(status_code=202, content=wait.model_dump())

        # We hold the lock. Build under it, and ALWAYS release — even on
        # error — before turning failures into HTTP responses.
        try:
            try:
                result = builder.build(
                    code=request.code, intent=request.intent
                )
            finally:
                token.release()
        except CodeValidationError as err:
            # Agent's fault: malformed code.
            return _error(400, err.code, err.message)
        except DeploymentError as err:
            # Server's fault: valid code we couldn't persist (retryable).
            return _error(500, err.code, err.message)

        # Success — report the deployed tool to the agent.
        return ProposeDeployedResponse(
            tool_id=result.tool_id,
            input_schema=result.input_schema,
            invoke_uri=result.invoke_uri,
            tool_status=result.status,
            catalog_version=result.catalog_version,
        )

    # ------------------------------------------------------------------
    # GET /tools/propose/wait/{token} — long-poll a build in progress
    # ------------------------------------------------------------------
    @app.get("/tools/propose/wait/{token}", response_model=ToolCatalogResponse)
    def propose_wait(token: str):
        """Block until the in-flight build finishes, then return the catalog.

        The ``token`` is opaque; the wait is global (one build at a time),
        so we simply wait for the lock to free. On timeout we return a
        ``408`` and the agent retries.
        """

        finished = build_lock.wait_for_release(
            timeout=_LONG_POLL_TIMEOUT_SECONDS
        )
        if not finished:
            return _error(
                408,
                "BUILD_WAIT_TIMEOUT",
                "Build still in progress; retry the wait.",
            )
        return _catalog_response()

    # ------------------------------------------------------------------
    # POST /tools/{tool_id}/invoke — call a deployed tool
    # ------------------------------------------------------------------
    @app.post("/tools/{tool_id}/invoke", response_model=InvokeToolResponse)
    def invoke_tool(tool_id: str, request: InvokeToolRequest):
        """Invoke a deployed, authorized tool and return its result.

        Governance (Layer 2) is enforced first: a non-authorized tool is
        refused with ``403`` and an ``invoke_denied`` audit event. Then
        the executor runs the code in a subprocess.
        """

        # Layer-2 enforcement: confirm the tool exists and may be invoked.
        try:
            governance.ensure_invokable(tool_id)
        except KeyError:
            return _error(
                404, "TOOL_NOT_FOUND", f"No tool with id {tool_id!r}."
            )
        except NotAuthorizedError as err:
            # Record the refusal for the audit trail, then 403.
            event_log.record(
                "invoke_denied", tool_id=tool_id, reason=err.code
            )
            return _error(403, err.code, err.message)

        # Run the tool. Execution failures (raise/timeout/crash) are 500s.
        try:
            result = executor.invoke(tool_id, request.args)
        except IntegrityError as err:
            return _error(500, "TOOL_INTEGRITY_ERROR", str(err))
        except ToolExecutionError as err:
            return _error(500, err.code, err.message)

        return InvokeToolResponse(result=result)

    # ------------------------------------------------------------------
    # Admin — approve / reject (human review)
    # ------------------------------------------------------------------
    @app.post("/admin/tools/{tool_id}/approve")
    def approve_tool(tool_id: str):
        """Mark a tool AUTHORIZED so it can be invoked."""

        try:
            registry.set_status(tool_id, "AUTHORIZED")
        except KeyError:
            return _error(
                404, "TOOL_NOT_FOUND", f"No tool with id {tool_id!r}."
            )
        event_log.record("approve", tool_id=tool_id, actor="admin")
        return {"tool_id": tool_id, "status": "AUTHORIZED"}

    @app.post("/admin/tools/{tool_id}/reject")
    def reject_tool(tool_id: str):
        """Mark a tool UNAUTHORIZED so it can never be invoked."""

        try:
            registry.set_status(tool_id, "UNAUTHORIZED")
        except KeyError:
            return _error(
                404, "TOOL_NOT_FOUND", f"No tool with id {tool_id!r}."
            )
        event_log.record("reject", tool_id=tool_id, actor="admin")
        return {"tool_id": tool_id, "status": "UNAUTHORIZED"}

    return app


def _loads_schema(schema_json: str) -> dict:
    """Parse stored schema JSON back into a dict, tolerating bad data.

    A malformed stored schema should never take down the whole catalog
    response, so we fall back to an empty object schema and let the
    integrity check / human review sort it out.

    Args:
        schema_json: The JSON text stored on the tool row.

    Returns:
        The parsed schema dict, or ``{"type": "object"}`` on parse error.
    """

    import json

    try:
        parsed = json.loads(schema_json)
        if isinstance(parsed, dict):
            return parsed
        return {"type": "object"}
    except (ValueError, TypeError):
        return {"type": "object"}


def run() -> None:
    """Console-script entry point (``ataf-server``).

    Reads configuration from the environment and starts uvicorn:

        ATAF_HOST              bind address (default 0.0.0.0)
        ATAF_PORT              port (default 9123)
        ATAF_DATA_DIR          data directory (default ./ataf_data)
        ATAF_ALLOW_PENDING     "1"/"true" to allow invoking PENDING tools
        ATAF_TOOL_TIMEOUT      per-call timeout seconds (default 5)
    """

    import uvicorn

    # Parse the dev-convenience flag from the environment.
    allow_pending = os.environ.get("ATAF_ALLOW_PENDING", "").lower() in (
        "1",
        "true",
        "yes",
    )
    tool_timeout = float(os.environ.get("ATAF_TOOL_TIMEOUT", "5"))

    app = create_app(
        allow_pending_invocation=allow_pending,
        tool_timeout_seconds=tool_timeout,
    )

    # Bind host/port from the environment, defaulting to 0.0.0.0:9123.
    host = os.environ.get("ATAF_HOST", "0.0.0.0")
    port = int(os.environ.get("ATAF_PORT", "9123"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
