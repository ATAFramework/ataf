"""ATAF server — FastAPI app + registry + builder + executor.

The server owns:
  * Persistent storage of generated tool code (filesystem)
  * Metadata index for tools (SQLite)
  * The global build lock for concurrent deploy requests
  * Code-to-FastAPI-route materialization (the builder)
  * Subprocess-based tool invocation (the executor)
  * Governance flag enforcement (PENDING_REVIEW / AUTHORIZED / UNAUTHORIZED)
  * Append-only deployment event log (JSONL)
"""
