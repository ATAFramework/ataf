"""Append-only JSONL event log for tool-deployment lifecycle events.

Every state-changing event in the tool-acquisition path is appended to
``ataf_data/logs/deployment.jsonl`` as one JSON object per line. This is
the audit trail — humans (and future analytics) can replay exactly what
happened, in what order, who did it, and how long it took.

Event types emitted (see DESIGN.md §8.2.1):

    propose         — agent submitted a new tool for deployment
    deploy          — server finished building and registered the tool
    deploy_failed   — server rejected the proposal (validation, build error)
    approve         — admin marked tool AUTHORIZED
    reject          — admin marked tool UNAUTHORIZED
    invoke_denied   — invocation refused because tool is not AUTHORIZED

Why JSONL (not a database table):
  * Trivially tail-able with standard unix tools.
  * Appends are atomic at the OS level (single write under PIPE_BUF).
  * No schema migration when we add new event types.
  * Easy to ship to log aggregators later (Loki, CloudWatch, etc).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class DeploymentEventLog:
    """Thread-safe writer for the deployment event log.

    A single instance is created at server startup and shared across
    every request handler. Each ``record()`` call writes one line.

    Concurrency: writes are serialized by an in-process ``Lock``.
    Since the FastAPI server is single-process in v0.1, this is enough
    to guarantee one-line-per-event without interleaving. Multi-process
    deployment in v0.4+ will need an OS-level append lock or a real
    log shipper.
    """

    def __init__(self, log_path: Path) -> None:
        """Create the writer.

        Args:
            log_path: Absolute path to the JSONL file. Parent directory
                must already exist (see ``StoragePaths.ensure_exists()``).
        """

        # Store the path; we open per-write to keep the file handle
        # short-lived. Small perf cost, big robustness win (no stale
        # handles if a log-rotation tool moves the file).
        self._log_path = log_path

        # Serialize writes across threads. FastAPI under uvicorn can
        # service requests concurrently from a thread pool, so without
        # this lock we'd get interleaved JSON on the same line.
        self._write_lock = Lock()

    def record(self, event: str, **fields: Any) -> None:
        """Append one event to the log.

        Args:
            event: Short identifier for what happened, e.g. ``"propose"``,
                ``"deploy"``, ``"approve"``. Conventionally snake_case.
            **fields: Arbitrary JSON-serializable kwargs. Common fields:
                ``tool_id``, ``intent``, ``actor``, ``reason``,
                ``build_duration_ms``. Pass whatever is relevant to
                the event type.

        The ``ts`` and ``event`` fields are always set by this method;
        callers should not pass them in ``fields``.
        """

        # Build the record. ISO-8601 UTC with explicit "Z" suffix is
        # unambiguous across timezones — better than .isoformat() alone,
        # which omits the offset for naive datetimes.
        record: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "event": event,
        }
        record.update(fields)

        # Serialize once outside the lock to minimize contention.
        # default=str is a safety net for things like Path or Decimal
        # that we don't want to crash on — better to log a stringified
        # value than to lose the event entirely.
        line = json.dumps(record, default=str) + "\n"

        # Serialize append across threads. Open in append-binary mode
        # to skip universal-newline translation on Windows.
        with self._write_lock:
            with self._log_path.open("ab") as handle:
                handle.write(line.encode("utf-8"))

    def read_all(self) -> list[dict[str, Any]]:
        """Read the entire log into memory as a list of dicts.

        Intended for tests and the admin CLI. Do NOT call this on hot
        request paths — the log can grow unboundedly. Production
        observability should tail the file instead.

        Returns:
            Events in insertion order, oldest first. Empty list if the
            log file doesn't exist yet.
        """

        # If we've never logged anything, the file may not exist yet.
        # Returning an empty list is friendlier than raising.
        if not self._log_path.exists():
            return []

        # Read line-by-line; skip blank lines defensively in case a
        # human or log-rotation tool has touched the file.
        events: list[dict[str, Any]] = []
        with self._log_path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                events.append(json.loads(line))

        return events
