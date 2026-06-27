# ATAF — Agent Tool Acquisition Framework

**Runtime tool acquisition for AI agents.**

When an AI agent needs a capability it doesn't have, the LLM writes the
missing tool as Python code. The ATAF server hot-deploys that code as a
live FastAPI endpoint. The agent re-prompts the LLM with the refreshed
catalog and continues. Tools persist across restarts and are shared
across all agents talking to the same server.

> **Status:** v0.1 in progress. Server layer is implemented and runnable
> (registry, build lock, builder, executor, governance, FastAPI app +
> `ataf-admin` CLI; 71 tests). The LLM-driven agent client is the
> remaining v0.1 work.

📐 **[Interactive design doc →](https://ataframework.github.io/ataf/concept.html)**
(diagrams, wire protocol, lifecycle — rendered from `concept.html`)

---

## The problem this solves

Every agent framework today — LangChain, CrewAI, Claude Agent SDK, MCP —
assumes the agent's toolset is **hand-curated at build time**. If the
agent encounters a task the developer didn't anticipate, it fails.

MCP standardizes *how* tools are called. It does not address *how new
tools get discovered and onboarded*. ATAF closes that loop.

## The core idea

The LLM is the best discovery + synthesis engine you have. Instead of
hunting external registries for missing capabilities, just ask the LLM
to write the missing tool, then deploy it.

The agent's prompt to the LLM includes the current tool catalog plus
instructions to emit new tools inside `<ATAF param="newtool">…</ATAF>`
tags. The ATAF server materializes the proposed code into a live FastAPI
endpoint with auto-generated OpenAPI schema, persists it, and the agent
re-prompts the LLM with the refreshed catalog.

```
Agent  ── catalog request ─────▶  ATAF Server
Agent  ── task + catalog ──────▶  LLM
LLM    ── proposes new tool ──▶  Agent
Agent  ── tool code ───────────▶  ATAF Server (builds + persists)
ATAF   ── refreshed catalog ──▶  Agent
Agent  ── updated catalog ────▶  LLM
LLM    ── invokes new tool ───▶  Agent ──▶ ATAF ──▶ result
```

## What's interesting about the design

- **Global build lock + long-poll** — only one tool builds at a time
  across the server; concurrent agents wait and refresh. Sidesteps the
  semantic-dedup problem entirely.
- **Two-layer governance** — tools default to `PENDING_REVIEW`. The LLM
  prompt instructs against using non-AUTHORIZED tools; the server returns
  `403 TOOL_NOT_AUTHORIZED` as a hard safety net if it tries anyway.
- **Persistence by default** — tools live in SQLite, re-import on server
  restart, accumulate as a real shared registry.
- **Tag protocol, not a separate channel** — new tools come through the
  LLM's normal output stream wrapped in `<ATAF>` tags. Model-agnostic,
  no native tool-use extensions required.

Full architecture, wire protocol, concurrency model, and governance
model are in [DESIGN.md](DESIGN.md) — or browse the
[interactive design doc](https://ataframework.github.io/ataf/concept.html)
for the same material with rendered diagrams.

## Phased roadmap

| Version | Focus | Status |
|---------|-------|--------|
| v0.1 | MVP — server + Claude adapter + circle-area demo + admin CLI | Queued |
| v0.2 | OpenAI + Gemini adapters, metrics, HTMX admin page | Planned |
| v0.3 | Sandboxing (WASM/Pyodide or Firecracker), policy engine | Planned |
| v0.4+ | Multi-node cluster, non-Python tools, versioning | Future |

## What this is NOT (v0.1)

- **Not sandboxed.** v0.1 trusts the LLM. Real isolation lands in v0.3.
  Do not run v0.1 against untrusted models or in production.
- **Not auto-deduplicated.** Two similar tools both deploy; humans prune.
- **Not authenticated.** Localhost-only by default.
- **Not multi-tenant.** Single-node server.

These are deliberate v0.1 deferrals, not oversights. See
[DESIGN.md §9](DESIGN.md) for the full out-of-scope list.

## Related projects

- **[HPRC Framework](https://github.com/HPRCFramework/hprc-framework)** —
  Sibling open-source project. AI-native HTML templating engine.

## License

Apache-2.0. See [LICENSE](LICENSE).

## Author

Rajesh Ramani — Apple Alum and Agentic AI Engineering Consultant.
[github.com/rrvenkatrama](https://github.com/rrvenkatrama)
