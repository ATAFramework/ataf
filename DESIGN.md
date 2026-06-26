# ATAF — Agent Tool Acquisition Framework

**Design Document v1 — locked 2026-06-26**
**Status:** design review (no code yet)

---

## 1. Concept

### The problem
Every agent framework today (LangChain, CrewAI, Claude Agent SDK, MCP)
assumes the agent's toolset is **hand-curated at build time**. If the
agent needs a capability the developer didn't anticipate, the agent fails.

### The insight
The LLM itself is the best discovery + synthesis engine. Instead of
hunting external registries for missing capabilities, just ask the LLM
to **write the missing tool**, then deploy that tool live so the agent
can call it on the next turn.

### The loop, in one paragraph
The agent fetches its current tool catalog from an **ATAF server** and
asks an LLM how to perform a task. The LLM either (a) picks an existing
tool, (b) writes a new Python tool inside `<ATAF param="newtool">…</ATAF>`
tags, or (c) declines. On (b), the agent forwards the code to the ATAF
server, which materializes it into a live FastAPI endpoint and returns
the new tool's schema. The agent re-prompts the LLM with the refreshed
catalog, and the LLM calls the new tool. Tools persist indefinitely in
the ATAF registry and are reusable across all agents talking to that
server.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                          AGENT (client)                         │
│  - Fetches tool catalog from ATAF                               │
│  - Sends catalog + user task to LLM                             │
│  - Parses LLM response: invoke | propose | decline              │
│  - Invokes tools via ATAF                                       │
│  - Submits new tool code to ATAF when LLM proposes              │
└─────────┬──────────────────────────────────────────┬────────────┘
          │                                          │
          │  HTTP                                    │  HTTP
          ▼                                          ▼
┌────────────────────────┐              ┌──────────────────────────┐
│         LLM            │              │      ATAF SERVER         │
│  (Claude / GPT / etc)  │              │      (FastAPI)           │
│                        │              │                          │
│  Returns:              │              │  Endpoints:              │
│  - tool call, OR       │              │   GET  /tools            │
│  - new tool code       │              │   POST /tools/propose    │
│    in ATAF tags, OR    │              │   POST /tools/{id}/invoke│
│  - decline             │              │   POST /admin/.../approve│
└────────────────────────┘              │                          │
                                        │  Internals:              │
                                        │  - Tool Registry (SQLite)│
                                        │  - Build Lock (global)   │
                                        │  - Tool Executor         │
                                        │    (subprocess per call) │
                                        │  - Long-poll waiters     │
                                        └──────────────────────────┘
```

### Component responsibilities

| Component | Owns | Does not own |
|-----------|------|--------------|
| Agent | LLM conversation, parsing tag protocol, deciding when to propose | Tool storage, tool execution |
| ATAF Server | Tool storage, tool deployment, tool execution, concurrency control, governance flags | LLM calls, prompt engineering |
| LLM | Reasoning, tool selection, tool code generation | Knowledge of which tools are deployed (must be told each turn) |

---

## 3. The Three LLM Responses (the contract)

The agent's prompt to the LLM includes:
- The user's task
- The current tool catalog (with flags)
- The ATAF tag protocol instructions

The LLM MUST respond in one of three shapes:

### Response 1 — Use existing tool

Standard tool-use response in the LLM's native format
(`tool_use` block for Claude, `function_call` for OpenAI, etc).

```
Calling tool: circle_area
Args: { "radius": 10 }
```

### Response 2 — Propose new tool

The LLM emits a function inside ATAF tags. The function MUST have:
- Type-annotated parameters
- A Google-style or NumPy-style docstring
- A return type annotation
- No imports outside the Python stdlib (v0.1 restriction)

```
I don't see an area calculator. Here is one:

<ATAF param="newtool">
def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius of the circle, in any unit.

    Returns:
        The area in the squared unit of the radius.
    """
    import math
    return math.pi * radius ** 2
</ATAF>

Please deploy this and call it with radius=10.
```

### Response 3 — Decline

```
I cannot do this task with the available tools, and I cannot write
a tool that would do it either.
```

The agent halts the loop on Response 3.

---

## 4. Wire Protocol

All endpoints accept and return JSON. All timestamps are ISO-8601 UTC.

### `GET /tools`

Returns the current tool catalog.

**Behavior:** if a build is in progress, this call **blocks** (long-poll)
until the build completes, then returns the refreshed catalog.

**Response (200):**
```json
{
  "tools": [
    {
      "tool_id": "circle_area_v1",
      "name": "circle_area",
      "description": "Compute the area of a circle.",
      "schema": {
        "type": "object",
        "properties": { "radius": { "type": "number" } },
        "required": ["radius"]
      },
      "invoke_uri": "/tools/circle_area_v1/invoke",
      "status": "AUTHORIZED",
      "created_at": "2026-06-26T18:00:00Z"
    }
  ],
  "build_in_progress": false,
  "catalog_version": 42
}
```

`status` is one of `AUTHORIZED`, `PENDING_REVIEW`, `UNAUTHORIZED`.

### `POST /tools/propose`

Submit a new tool for deployment.

**Request:**
```json
{
  "intent": "compute area of a circle",
  "code": "def circle_area(radius: float) -> float:\n    \"\"\"...\"\"\"\n    import math\n    return math.pi * radius ** 2\n"
}
```

**Possible responses:**

`200 OK` — built and deployed:
```json
{
  "status": "DEPLOYED",
  "tool_id": "circle_area_v1",
  "schema": { ... },
  "invoke_uri": "/tools/circle_area_v1/invoke",
  "tool_status": "PENDING_REVIEW",
  "catalog_version": 43
}
```

`202 Accepted` — build in progress by another agent; wait:
```json
{
  "status": "WAIT",
  "poll_url": "/tools/propose/wait/abc123",
  "eta_seconds": 5
}
```

`400 Bad Request` — code failed to parse, missing docstring, or missing
type annotations.

### `GET /tools/propose/wait/{token}`

Long-poll until the in-flight build completes, then return the refreshed
catalog (same shape as `GET /tools`). The agent that called this should
treat the response as a tool-list refresh and re-prompt the LLM.

### `POST /tools/{tool_id}/invoke`

Call a deployed tool.

**Request:**
```json
{ "args": { "radius": 10 } }
```

**Responses:**

`200 OK`:
```json
{ "result": 314.159265 }
```

`403 Forbidden` — tool exists but is not AUTHORIZED and server is
configured to block unauthorized invocation:
```json
{
  "error": "TOOL_NOT_AUTHORIZED",
  "message": "Tool 'circle_area_v1' is pending human review."
}
```

The agent forwards this error to the LLM as a tool-result. The LLM is
instructed to halt tool calls on receiving this.

`500 Internal Server Error` — tool raised an exception:
```json
{ "error": "TOOL_EXECUTION_ERROR", "message": "ZeroDivisionError: ..." }
```

### `POST /admin/tools/{tool_id}/approve` and `/reject`

Human-review actions. Flip the tool's status. Out of v0.1 scope as a UI;
v0.1 ships a CLI script (`ataf-admin approve <tool_id>`).

---

## 5. Concurrency Model

### Global build lock (locked decision)

Only **one** tool can be built at a time across the entire ATAF server.
While a build is in progress:

- `POST /tools/propose` from any agent returns `202 WAIT`
- `GET /tools` from any agent **blocks** (long-poll) until build completes
- `POST /tools/{id}/invoke` is **not blocked** — execution of existing
  tools continues normally

### Why a global lock instead of a per-tool lock

Per-tool locking requires identifying "the same tool" across submissions,
which requires either fragile name normalization or unreliable semantic
similarity. A global lock sidesteps the question entirely. Tool builds
are infrequent and fast (sub-second to a few seconds), so the latency
cost is acceptable for v0.1.

### Long-poll mechanics

When an agent gets `202 WAIT`, it polls the returned `poll_url`. The
server holds the connection open until either:
- The build completes → returns the refreshed catalog (200)
- A timeout (default 30s) → returns `408`, agent retries

When the agent receives the refreshed catalog, it MUST re-prompt the LLM
with the new tool list before doing anything else. This is why the
refresh is hidden inside the ATAF client library — the user code just
sees "the tool list now includes the new tool."

### No semantic dedup (locked decision)

If two agents propose `circle_area` and `area_of_circle` an hour apart,
both get deployed. Duplicates accumulate. **Human review prunes them
later** via the approve/reject admin endpoints. We do not attempt to
auto-detect that two tools are functionally equivalent.

This is a deliberate tradeoff: simpler, predictable, no false-positive
merges (e.g., area-of-circle ≠ area-of-square even if their docstrings
embed similarly).

---

## 6. Governance Model

### Tool status states

| Status | Set by | LLM sees it as | Server allows invocation? |
|--------|--------|----------------|---------------------------|
| `PENDING_REVIEW` | Default after deploy | "Unavailable — pending review" | Config flag: `allow_pending_invocation` (default false) |
| `AUTHORIZED` | Human admin | Normal tool | Yes |
| `UNAUTHORIZED` | Human admin | "Unavailable — rejected" | No |

### Two-layer enforcement

**Layer 1 — Prompt-level (LLM cooperation):**
The catalog sent to the LLM includes the `status` flag. The prompt
template instructs:

> Tools with status other than AUTHORIZED are unavailable. If a
> non-AUTHORIZED tool matches the user's request, DO NOT propose
> a duplicate. Instead, decline with Response 3.

This prevents the duplication loop (LLM sees a PENDING tool, claims it
can't use it, generates the same tool again).

**Layer 2 — Server-level (hard enforcement):**
Even if the LLM hallucinates and invokes a non-AUTHORIZED tool, the
ATAF server returns `403 TOOL_NOT_AUTHORIZED`. The agent forwards this
error to the LLM, and the prompt template instructs the LLM:

> If a tool invocation returns TOOL_NOT_AUTHORIZED, stop calling tools
> and respond with the final user-facing message.

Both layers must exist. Layer 1 is the happy path; Layer 2 is the safety net.

---

## 7. Tool Lifecycle

```
                  [Agent submits code]
                          │
                          ▼
              ┌──────────────────────┐
              │  Parse + validate    │  (signature, docstring, types)
              └──────────┬───────────┘
                         │
                ┌────────┴────────┐
                │ valid?           │
                ├── no ──► 400 Bad Request
                ▼ yes
              ┌──────────────────────┐
              │  Acquire build lock  │  (or WAIT)
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Generate adapter:   │  (Jinja template:
              │  - FastAPI route     │   docstring → schema,
              │  - Pydantic model    │   signature → Pydantic model)
              │  - Persist to SQLite │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Hot-register route  │  (app.add_api_route)
              │  status = PENDING    │
              └──────────┬───────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Release lock        │
              │  Notify waiters      │
              └──────────┬───────────┘
                         │
                         ▼
                  [DEPLOYED, PENDING_REVIEW]
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      [admin approve]        [admin reject]
              │                     │
              ▼                     ▼
        AUTHORIZED            UNAUTHORIZED
              │                     │
              ▼                     │
       (invokable forever)          │
                                    ▼
                            (kept in registry,
                             never invokable)
```

Tools are never auto-deleted. The registry is append-only with status
mutations. This makes the system inspectable and undoable.

---

## 8. Component Design

### 8.1 ATAF Server (FastAPI)

```
server/
  main.py              # FastAPI app, route registration
  protocol.py          # Pydantic request/response models
  registry.py          # SQLite tool storage
  builder.py           # Code → adapter → live route
  executor.py          # Subprocess-based tool invocation
  lock.py              # Global build lock + waiter queue
  governance.py        # Status flag checks
  admin_cli.py         # `ataf-admin approve|reject|list`
```

### 8.2 Tool Registry (SQLite schema)

```sql
CREATE TABLE tools (
  tool_id         TEXT PRIMARY KEY,
  name            TEXT NOT NULL,
  description     TEXT NOT NULL,
  code            TEXT NOT NULL,
  schema_json     TEXT NOT NULL,
  status          TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
  intent          TEXT,
  created_at      TEXT NOT NULL,
  status_updated_at TEXT,
  call_count      INTEGER DEFAULT 0,
  last_called_at  TEXT
);

CREATE INDEX idx_status ON tools(status);
CREATE INDEX idx_name ON tools(name);
```

On server startup, every tool is re-imported into a fresh namespace and
its FastAPI route re-registered. The catalog survives restarts.

### 8.3 Tool Executor (v0.1)

Each invocation runs the tool's code in a `subprocess` with a 30-second
timeout, JSON args on stdin, JSON result on stdout. No sandboxing in
v0.1 — that is a deliberate v0.3 deferral (see §9).

```python
# Conceptual — actual implementation will be cleaner
def invoke(tool_id: str, args: dict) -> Any:
    code = registry.get_code(tool_id)
    func_name = registry.get_name(tool_id)
    runner = f"{code}\n\nimport sys, json\nargs = json.loads(sys.stdin.read())\nprint(json.dumps({func_name}(**args)))"
    result = subprocess.run(
        ["python", "-c", runner],
        input=json.dumps(args),
        capture_output=True,
        timeout=30,
        text=True,
    )
    return json.loads(result.stdout)
```

### 8.4 Agent Client (Python library)

```
client/
  __init__.py
  agent.py             # AtafAgent class
  llm_adapters/        # claude.py, openai.py, gemini.py
  tag_parser.py        # Extract <ATAF param="newtool">...</ATAF>
  prompt_template.py   # The canonical system prompt (see 8.5)
```

The user-facing API:

```python
from ataf.client import AtafAgent
from ataf.client.llm_adapters import ClaudeAdapter

agent = AtafAgent(
    ataf_server="http://localhost:8000",
    llm=ClaudeAdapter(model="claude-sonnet-4-6"),
)

answer = agent.run("Calculate the area of a circle with radius 10.")
# Internally:
# 1. GET /tools → catalog
# 2. Send catalog + task to Claude
# 3. Parse response: tool_use? new tool? decline?
# 4. If new tool: POST /tools/propose, handle WAIT, refresh catalog, re-prompt
# 5. Invoke tool, return result to Claude, loop until final answer
```

### 8.5 Canonical LLM prompt template (v1)

```
You are an agent with access to a dynamic toolset managed by ATAF.

CURRENT TOOLS:
{catalog_with_flags}

RULES:
1. To use an AUTHORIZED tool, call it in your native tool-use format.
2. Tools with status PENDING_REVIEW or UNAUTHORIZED are unavailable.
   Do NOT propose a duplicate of a non-AUTHORIZED tool that matches the
   user's request. Instead, respond with: "Cannot complete this task —
   required tool is pending review."
3. If no AUTHORIZED tool fits AND no PENDING_REVIEW tool matches, you
   may propose a new tool by emitting:

       <ATAF param="newtool">
       def my_tool(arg1: type, ...) -> return_type:
           """Description.

           Args:
               arg1: ...

           Returns:
               ...
           """
           # implementation
       </ATAF>

   Constraints on proposed tools:
   - Type-annotated parameters and return value
   - Google-style or NumPy-style docstring
   - Only Python stdlib imports (no external packages in v0.1)
   - Pure function preferred; no global state

4. If a tool invocation returns TOOL_NOT_AUTHORIZED, stop calling tools
   and give the user your best non-tool answer.

5. If no tool exists and you cannot write one, respond with:
   "I cannot complete this task."

USER TASK:
{user_task}
```

---

## 9. Out of Scope (v0.1)

Deliberately deferred to keep v0.1 shippable:

- **Sandboxing.** Subprocess only; no seccomp, no WASM, no container
  isolation. Trust assumption: the LLM is honest. Real sandboxing lands
  in v0.3.
- **Semantic dedup.** Duplicates accumulate; humans prune.
- **Authentication.** No API keys on the ATAF server. Localhost-only
  for v0.1.
- **Distributed ATAF servers.** Single-node. Multi-node coordination
  (with the same global-lock semantics) is a v0.4+ concern.
- **Auto-retire.** Tools live forever.
- **Non-stdlib imports.** Tools can only use the Python stdlib.
  Allowing `pip install` per-tool requires sandboxing first.
- **Non-Python tool languages.** Python only.

---

## 10. Phased Implementation

### v0.1 — MVP (target ~2 weeks of focused work)

- ATAF Server: registry + builder + executor + global lock + long-poll
- Tag parser
- Claude adapter (only)
- Canonical prompt template
- `ataf-admin` CLI for approve/reject
- One worked example: circle-area agent (writes the tool, gets approved,
  re-runs and uses it)
- README + this DESIGN.md
- Unit tests for: tag parser, code validator, build-lock concurrency,
  governance enforcement

**Exit criteria:** demo video of a fresh ATAF server, an agent asking
for circle area, the LLM proposing the tool, the admin approving it,
and a second agent re-using the deployed tool.

### v0.2 — Multi-LLM + observability (target ~2 weeks)

- OpenAI adapter, Gemini adapter
- Tool call/error metrics persisted to SQLite
- Web UI for admin review (simple FastAPI + HTMX page)
- Structured logs (JSONL)
- Second worked example: a research-helper agent that builds 3-4 tools
  across a session

### v0.3 — Sandboxing (target ~3 weeks)

- Pluggable executor interface
- WASM executor (Wasmtime + Python via Pyodide) OR Firecracker microVM
- Capability declarations (network y/n, filesystem y/n, env access)
- Policy engine: per-tool capability allowlist
- v0.1 trust assumption lifted

### v0.4+ — Future

- Multi-node ATAF cluster (distributed lock)
- Non-Python tools (Node, Go) via per-language executors
- Tool versioning + rollback
- Auto-retire based on call volume + age

---

## 11. Open Questions (deferred but tracked)

1. **GitHub org**: RESOLVED 2026-06-26 → `ATAFramework` (own org,
   matches HPRC pattern; single F since ATAF already ends in F for
   Framework).
2. **Admin auth model for v0.2 web UI**: shared secret env var, OAuth,
   or local-only?
3. **Schema generation library**: hand-roll from `inspect.signature` +
   `docstring-parser`, or pull in `griffe`?
4. **Catalog versioning header**: should `GET /tools` support
   `If-None-Match` so agents can skip the LLM round-trip if catalog
   hasn't changed?
5. **Tool naming collisions**: two LLM-proposed tools both named
   `circle_area`. Auto-suffix (`_v2`) or reject? (Current draft:
   auto-suffix on the `tool_id`; keep the public `name` field as the
   LLM-given name, since the LLM addresses tools by name in its
   prompt context.)

---

## Document control

- v1 — locked 2026-06-26 after design discussion with Rajesh
- Next review: before v0.1 code starts
- Owner: Rajesh Ramani
- License intent: Apache-2.0 (matches HPRC)
