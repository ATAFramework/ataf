"""ATAF chat — a tiny web app to test the agent and watch tools appear.

A single-page chat UI backed by a small FastAPI service. You type a task,
the :class:`AtafAgent` runs against an ATAF server (using Claude), and the
sidebar lists the live tool catalog — so as the agent proposes new tools
(auto-authorized on the server), you watch them show up.

This is an *example/demo*, not part of the framework. It holds the
Anthropic key and drives the public client library exactly as any user
would.

Run:

    export ANTHROPIC_API_KEY=sk-ant-...
    # optional overrides:
    export ATAF_SERVER_URL=http://192.168.1.156:9123   # default
    export ATAF_CHAT_MODEL=claude-opus-4-8             # default
    export ATAF_CHAT_PORT=8800                          # default

    python examples/chat/app.py
    # then open http://127.0.0.1:8800

The ATAF server it points at should run with ATAF_AUTO_AUTHORIZE=1 so newly
proposed tools become AUTHORIZED without a manual approve step (that's what
makes the "tools appear as you chat" experience work).
"""

from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ataf.client import AtafAgent, AtafProtocolError, AtafServerError, ToolPolicy
from ataf.client.llm_adapters.claude import ClaudeAdapter


# --- Configuration (all overridable via the environment) ---
ATAF_SERVER_URL = os.environ.get("ATAF_SERVER_URL", "http://192.168.1.156:9123")
CHAT_MODEL = os.environ.get("ATAF_CHAT_MODEL", "claude-opus-4-8")
STATIC_DIR = Path(__file__).parent / "static"


# Request body for the chat endpoint.
class ChatRequest(BaseModel):
    """One chat turn from the browser.

    Attributes:
        message: The user's task text.
        policy: Which tool-use policy to run under (one of the ToolPolicy
            values). Defaults to PREFER_NEWTOOL.
    """

    message: str
    policy: str = ToolPolicy.PREFER_NEWTOOL.value


app = FastAPI(title="ATAF Chat", version="0.1.0")

# A single shared HTTP client to the ATAF server, reused across requests
# (the agent borrows it; it does not own or close it).
_ataf_http = httpx.Client(base_url=ATAF_SERVER_URL, timeout=60.0)

# The Claude adapter is built lazily on first use so the app still starts
# (and serves the tools sidebar) when ANTHROPIC_API_KEY isn't set yet.
_adapter: ClaudeAdapter | None = None


def _get_adapter() -> ClaudeAdapter:
    """Return the shared Claude adapter, building it on first use.

    Raises:
        RuntimeError: If ANTHROPIC_API_KEY is not set.
    """

    global _adapter
    if _adapter is None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set — export it before chatting."
            )
        _adapter = ClaudeAdapter(model=CHAT_MODEL)
    return _adapter


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    """Serve the single-page chat UI."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/tools")
def list_tools() -> JSONResponse:
    """Proxy the ATAF server's catalog so the sidebar can poll it.

    Returns the ``tools`` array and ``catalog_version`` (or an ``error``
    field if the ATAF server is unreachable, so the UI can show a notice
    rather than breaking).
    """

    try:
        resp = _ataf_http.get("/tools")
        body = resp.json()
        return JSONResponse(
            {
                "tools": body.get("tools", []),
                "catalog_version": body.get("catalog_version", 0),
            }
        )
    except httpx.HTTPError as err:
        return JSONResponse(
            {"tools": [], "catalog_version": 0, "error": str(err)}, status_code=200
        )


@app.post("/api/chat")
def chat(request: ChatRequest) -> JSONResponse:
    """Run one agent turn for the user's message and return the result.

    The response describes the outcome and which tools were proposed and
    invoked during the run, so the UI can narrate the acquisition loop.
    """

    # Resolve the requested policy, defaulting to PREFER on anything odd.
    try:
        policy = ToolPolicy(request.policy)
    except ValueError:
        policy = ToolPolicy.PREFER_NEWTOOL

    # Build the agent for this turn, sharing the adapter and HTTP client.
    try:
        adapter = _get_adapter()
    except RuntimeError as err:
        return JSONResponse({"error": str(err)}, status_code=400)

    agent = AtafAgent(
        adapter,
        http_client=_ataf_http,
        tool_policy=policy,
        # The server auto-authorizes, so freshly built tools are AUTHORIZED
        # and callable without us relaxing the pending check.
        max_turns=12,
    )

    # Run the loop. Surface protocol/server errors as readable messages
    # rather than 500s so the chat keeps working.
    try:
        result = agent.run(request.message)
    except AtafProtocolError as err:
        return JSONResponse(
            {
                "outcome": "PROTOCOL_ERROR",
                "answer": f"[policy violation] {err}",
                "tools_proposed": [],
                "tools_invoked": [],
            }
        )
    except AtafServerError as err:
        return JSONResponse(
            {
                "outcome": "SERVER_ERROR",
                "answer": f"[ATAF server error] {err}",
                "tools_proposed": [],
                "tools_invoked": [],
            }
        )

    # Normalize the result into a flat JSON shape for the browser.
    return JSONResponse(
        {
            "outcome": result.outcome.value,
            "answer": result.answer,
            "decline_code": result.decline_code,
            "tools_proposed": result.tools_proposed,
            "tools_invoked": result.tools_invoked,
            "turns": result.turns,
        }
    )


def run() -> None:
    """Start the chat app with uvicorn."""

    import uvicorn

    port = int(os.environ.get("ATAF_CHAT_PORT", "8800"))
    print(f"ATAF chat → {ATAF_SERVER_URL}  (model: {CHAT_MODEL})")
    print(f"Open http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)


if __name__ == "__main__":
    run()
