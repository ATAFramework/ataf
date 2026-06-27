# ATAF Chat — a UI to test the agent and watch tools appear

A single-page chat app backed by a small FastAPI service. You type a task,
the `AtafAgent` runs it (via Claude) against an ATAF server, and the
**sidebar lists the live tool catalog**. When the agent proposes a new tool
— and the server is in auto-authorize mode — it deploys, authorizes, and
shows up in the sidebar (with a brief highlight) as you chat.

## Prerequisites

1. An **ATAF server** running with auto-authorize on, so new tools become
   usable without a manual approve step. The deployed instance at
   `192.168.1.156:9123` already runs this way (`ATAF_AUTO_AUTHORIZE=1`).
   To run one locally instead:
   ```bash
   ATAF_AUTO_AUTHORIZE=1 ATAF_PORT=9123 ataf-server
   ```
2. An **Anthropic API key**.

## Run

```bash
source .venv/bin/activate
export ANTHROPIC_API_KEY=sk-ant-...

# optional overrides (defaults shown):
export ATAF_SERVER_URL=http://192.168.1.156:9123
export ATAF_CHAT_MODEL=claude-opus-4-8
export ATAF_CHAT_PORT=8800

python examples/chat/app.py
# open http://127.0.0.1:8800
```

## Try it

Ask: **"What's the area of a circle with radius 11?"**

With the default `PREFER_NEWTOOL` policy and an empty catalog, the agent
writes a `circle_area` tool, deploys it (auto-authorized), invokes it, and
answers — and `circle_area_v1` appears in the sidebar. Ask a follow-up that
reuses it and you'll see the call without a rebuild.

## The policy selector

The dropdown switches the agent's `tool_policy` per message:

| Policy | Behavior |
|--------|----------|
| `PREFER_NEWTOOL` (default) | Steers toward building/using tools; still answers conversationally. |
| `MODEL_DECIDES_NEWTOOL` | No steering — may answer trivial things directly. |
| `REQUIRE_NEWTOOL` | Must use, build, or decline; a bare prose answer is a protocol error. |
| `USE_ONLY_EXISTINGTOOL` | Never builds; only uses already-authorized tools or declines. |

## How it works

- `app.py` holds the Anthropic key, builds a `ClaudeAdapter`, and runs
  `AtafAgent` against `ATAF_SERVER_URL`. Each chat message is one
  `agent.run(task)`; tools persist on the server across messages.
- `GET /api/tools` proxies the ATAF catalog; the page polls it every 3s and
  again right after each turn.
- `POST /api/chat` runs a turn and returns the outcome plus which tools were
  **proposed** and **invoked**, shown as chips under each reply.

This is an example/demo — it uses only the public `ataf.client` API.
