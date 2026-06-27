"""Tool registry — SQLite metadata + filesystem-backed code.

This module is the persistence layer for the ATAF server. It owns:

  * **The SQLite metadata table** (one row per tool). See DESIGN.md
    §8.2 for the schema.
  * **The on-disk code files** under ``ataf_data/tools/{tool_id}.py``.
  * **The ``tool_id`` allocation policy** — when two tools share a
    function name, the second gets ``_v2``, the third ``_v3``, etc.
    (See DESIGN.md §11 open question 5 — auto-suffix policy.)
  * **The integrity check on startup** — every row's stored
    ``code_sha256`` must match the on-disk file. Mismatches mark the
    tool UNAUTHORIZED rather than silently auto-recovering.
  * **The catalog version counter** — bumps on every insert and every
    status mutation.

The registry does **not** know how to build tools or invoke them. It is
a pure storage layer. The builder calls ``insert()``; the executor
reads code via ``get_code()``; the governance module calls
``set_status()``.

Concurrency:
  * SQLite operations use a single ``Connection`` opened with
    ``check_same_thread=False`` and a process-wide lock. Volume is low
    enough (a deploy per minute is heavy traffic) that this is fine.
  * Filesystem writes go through ``StoragePaths`` and are not locked
    separately — the build lock in ``lock.py`` serializes all
    code-writing across the server.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .protocol import ToolStatus
from .storage import StoragePaths


# ----------------------------------------------------------------------
# Schema. Single CREATE statement, idempotent.
# ----------------------------------------------------------------------
_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS tools (
    tool_id           TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    description       TEXT NOT NULL,
    code_path         TEXT NOT NULL,
    code_sha256       TEXT NOT NULL,
    schema_json       TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'PENDING_REVIEW',
    intent            TEXT,
    created_at        TEXT NOT NULL,
    status_updated_at TEXT,
    call_count        INTEGER DEFAULT 0,
    last_called_at    TEXT
);

CREATE INDEX IF NOT EXISTS idx_status ON tools(status);
CREATE INDEX IF NOT EXISTS idx_name   ON tools(name);

CREATE TABLE IF NOT EXISTS server_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# Tool IDs must be safe to use as both a filename and a URL segment.
# We enforce a strict regex up front rather than escape later.
_TOOL_ID_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


@dataclass(frozen=True)
class ToolRow:
    """One row of the ``tools`` table, surfaced as a frozen dataclass.

    Public callers should not construct these directly — they come back
    from ``Registry.get()`` and ``Registry.list_all()``.

    Attributes:
        tool_id: Unique identifier, also the code filename stem.
        name: LLM-given function name.
        description: First line of the function's docstring.
        code_path: Relative path (under ``StoragePaths.tools_dir``) of
            the source file. We store it relative so the data
            directory can be moved.
        code_sha256: SHA-256 hex digest of the code file's bytes at
            insertion time. Verified at startup.
        schema_json: JSON-serialized invocation schema. Parsed back to
            a dict only when needed (kept as a string in SQLite for
            zero conversion at write time).
        status: One of PENDING_REVIEW / AUTHORIZED / UNAUTHORIZED.
        intent: The agent's stated intent at propose-time. Optional —
            older rows may have None if intent was not captured.
        created_at: ISO-8601 UTC timestamp of original insert.
        status_updated_at: ISO-8601 UTC timestamp of last status change.
            None if status has never been changed since insert.
        call_count: Total successful invocations. Bumped by the executor.
        last_called_at: ISO-8601 UTC timestamp of most recent invocation.
    """

    tool_id: str
    name: str
    description: str
    code_path: str
    code_sha256: str
    schema_json: str
    status: ToolStatus
    intent: str | None
    created_at: str
    status_updated_at: str | None
    call_count: int
    last_called_at: str | None


class IntegrityError(Exception):
    """Raised when a tool's on-disk code file is missing or has a
    different SHA-256 than the registry expects. The startup recovery
    routine catches this and marks the affected tool UNAUTHORIZED.
    """


class Registry:
    """SQLite + filesystem tool registry.

    One instance per ATAF server. Thread-safe via a single process-wide
    lock around the SQLite connection.

    Lifecycle:

        paths = default_storage("./ataf_data")
        paths.ensure_exists()
        registry = Registry(paths)
        registry.initialize()  # creates schema if needed
        # ... use registry.insert(), registry.list_all(), etc.
    """

    def __init__(self, paths: StoragePaths) -> None:
        """Construct, but do not yet touch the database.

        Args:
            paths: Resolved StoragePaths for this server's data dir.
        """

        self._paths = paths

        # Connection is created lazily in initialize() so that
        # __init__ stays side-effect-free (helps testability).
        self._conn: sqlite3.Connection | None = None

        # All SQL operations go through this lock. SQLite itself
        # serializes writes; the lock just keeps our Python-side
        # state (cursors, fetchall calls) safe across threads.
        self._db_lock = Lock()

        # The catalog_version counter, in-memory mirror of the
        # persisted value in server_meta. Bumped on every state change.
        self._catalog_version: int = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Open the SQLite connection, create schema, load catalog version.

        Idempotent: safe to call multiple times. Should be called once
        at server startup, after ``StoragePaths.ensure_exists()``.
        """

        # Open with check_same_thread=False because uvicorn's worker
        # thread pool may touch the connection from different threads;
        # we serialize ourselves via _db_lock.
        self._conn = sqlite3.connect(
            self._paths.sqlite_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit mode; we control transactions explicitly
        )

        # Enable foreign keys (not used yet, but good hygiene) and
        # write-ahead logging for slightly better concurrent reads.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

        # Apply schema. CREATE IF NOT EXISTS makes this safe to re-run.
        self._conn.executescript(_SCHEMA_SQL)

        # Load the persisted catalog version (or 0 if first boot).
        cursor = self._conn.execute(
            "SELECT value FROM server_meta WHERE key = ?",
            ("catalog_version",),
        )
        row = cursor.fetchone()
        if row is None:
            # First boot: insert the initial counter row.
            self._conn.execute(
                "INSERT INTO server_meta (key, value) VALUES (?, ?)",
                ("catalog_version", "0"),
            )
            self._catalog_version = 0
        else:
            self._catalog_version = int(row[0])

    def close(self) -> None:
        """Close the SQLite connection. Idempotent.

        Mostly useful in tests; the production server holds the
        connection open for its full lifetime.
        """

        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Tool ID allocation
    # ------------------------------------------------------------------

    def allocate_tool_id(self, function_name: str) -> str:
        """Pick a unique ``tool_id`` for a new tool with this function name.

        Policy: ``{function_name}_v1`` if free, otherwise ``_v2``, ``_v3``...

        Why version suffix instead of UUID: human-readable. ``circle_area_v1``
        in the URL and on disk is much friendlier than a UUID for debugging.

        Args:
            function_name: The Python identifier the LLM gave the function
                (e.g., ``circle_area``). Must already be validated as a
                safe Python identifier upstream.

        Returns:
            An unused ``tool_id`` of the form ``{name}_v{n}``.

        Raises:
            ValueError: If function_name contains anything other than
                identifier-safe characters.
        """

        # Defensive check — the builder validates this upstream, but
        # we re-check here to keep this module standalone-testable.
        if not _TOOL_ID_PATTERN.match(function_name):
            raise ValueError(
                f"function_name {function_name!r} is not a valid identifier"
            )

        # Find the highest existing version for this name and add one.
        # We don't bother with a transaction here because the build lock
        # ensures there's never a concurrent allocate_tool_id call.
        with self._db_lock:
            cursor = self._require_conn().execute(
                "SELECT tool_id FROM tools WHERE name = ? ORDER BY tool_id",
                (function_name,),
            )
            existing_ids = [row[0] for row in cursor.fetchall()]

        # Extract the version numbers from suffixes that match the pattern
        # ``{function_name}_v{N}``. Any rows that don't match (legacy data,
        # manually inserted rows) are ignored for the purposes of allocation.
        version_prefix = f"{function_name}_v"
        used_versions: set[int] = set()
        for tid in existing_ids:
            if tid.startswith(version_prefix):
                suffix = tid[len(version_prefix):]
                if suffix.isdigit():
                    used_versions.add(int(suffix))

        # Find the smallest positive integer not yet used.
        next_version = 1
        while next_version in used_versions:
            next_version += 1

        return f"{function_name}_v{next_version}"

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert(
        self,
        *,
        tool_id: str,
        name: str,
        description: str,
        code: str,
        schema_json: str,
        intent: str,
    ) -> ToolRow:
        """Persist a new tool: write the .py file, then the SQLite row.

        Atomicity model: the .py file is written first. If the SQLite
        insert then fails, we delete the file. This keeps the two stores
        consistent without a real distributed transaction.

        Args:
            tool_id: Pre-allocated unique ID. Use ``allocate_tool_id()``.
            name: The function name (must equal the def in ``code``).
            description: First line of the tool's docstring.
            code: Full Python source for the tool.
            schema_json: JSON-serialized invocation schema.
            intent: Agent's stated intent at propose-time.

        Returns:
            The freshly written ``ToolRow``.
        """

        # 1. Compute the integrity hash before any IO. We want the
        # bytes we hash to be exactly the bytes we write.
        code_bytes = code.encode("utf-8")
        code_sha256 = hashlib.sha256(code_bytes).hexdigest()

        # 2. Write the .py file. exclusive mode ("xb") fails if the
        # file already exists — that would indicate a tool_id collision,
        # which is a programming error (allocate_tool_id should prevent it).
        absolute_path = self._paths.tool_code_path(tool_id)
        relative_path = absolute_path.relative_to(self._paths.root)
        with absolute_path.open("xb") as handle:
            handle.write(code_bytes)

        # 3. Insert the metadata row. If this fails, roll back the file
        # write so we don't leave an orphan .py on disk.
        created_at = _now_iso()
        try:
            with self._db_lock:
                self._require_conn().execute(
                    """
                    INSERT INTO tools (
                        tool_id, name, description, code_path,
                        code_sha256, schema_json, status, intent,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        tool_id,
                        name,
                        description,
                        str(relative_path),
                        code_sha256,
                        schema_json,
                        "PENDING_REVIEW",
                        intent,
                        created_at,
                    ),
                )
                self._bump_catalog_version_locked()
        except Exception:
            # Roll back the file write so we don't leak orphan code.
            absolute_path.unlink(missing_ok=True)
            raise

        # 4. Return the row we just inserted. We could re-SELECT, but
        # constructing the dataclass directly is cheaper and we already
        # know every field.
        return ToolRow(
            tool_id=tool_id,
            name=name,
            description=description,
            code_path=str(relative_path),
            code_sha256=code_sha256,
            schema_json=schema_json,
            status="PENDING_REVIEW",
            intent=intent,
            created_at=created_at,
            status_updated_at=None,
            call_count=0,
            last_called_at=None,
        )

    def set_status(self, tool_id: str, new_status: ToolStatus) -> None:
        """Mutate the status flag for a tool (admin action).

        Args:
            tool_id: The tool to update.
            new_status: One of PENDING_REVIEW / AUTHORIZED / UNAUTHORIZED.

        Raises:
            KeyError: If the tool_id does not exist.
        """

        # Validate the target status. Pydantic's Literal type already
        # catches this at the API boundary, but the registry is also
        # callable from the admin CLI which doesn't go through Pydantic.
        if new_status not in ("PENDING_REVIEW", "AUTHORIZED", "UNAUTHORIZED"):
            raise ValueError(f"invalid status: {new_status!r}")

        # Do the update and ensure the row existed.
        with self._db_lock:
            cursor = self._require_conn().execute(
                """
                UPDATE tools
                SET status = ?, status_updated_at = ?
                WHERE tool_id = ?
                """,
                (new_status, _now_iso(), tool_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"unknown tool_id: {tool_id!r}")
            self._bump_catalog_version_locked()

    def record_invocation(self, tool_id: str) -> None:
        """Increment call_count and update last_called_at.

        Called by the executor after a successful invocation. Does NOT
        bump catalog_version — invocation stats don't change the catalog
        shape, so clients don't need to refresh on every call.

        Args:
            tool_id: The tool that was just invoked.
        """

        with self._db_lock:
            self._require_conn().execute(
                """
                UPDATE tools
                SET call_count = call_count + 1, last_called_at = ?
                WHERE tool_id = ?
                """,
                (_now_iso(), tool_id),
            )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get(self, tool_id: str) -> ToolRow | None:
        """Fetch one tool by ID, or None if it doesn't exist.

        Args:
            tool_id: The identifier to look up.

        Returns:
            A ``ToolRow`` or None.
        """

        with self._db_lock:
            cursor = self._require_conn().execute(
                "SELECT * FROM tools WHERE tool_id = ?",
                (tool_id,),
            )
            row = cursor.fetchone()
            column_names = [d[0] for d in cursor.description]

        if row is None:
            return None
        return self._row_to_toolrow(column_names, row)

    def list_all(self) -> list[ToolRow]:
        """Fetch every tool, ordered by ``created_at`` (oldest first).

        Returns:
            All tools in the registry regardless of status. Filtering
            by status is the LLM's job (per DESIGN.md §6 / §8.5).
        """

        with self._db_lock:
            cursor = self._require_conn().execute(
                "SELECT * FROM tools ORDER BY created_at"
            )
            rows = cursor.fetchall()
            column_names = [d[0] for d in cursor.description]

        return [self._row_to_toolrow(column_names, r) for r in rows]

    def get_code(self, tool_id: str) -> str:
        """Read the tool's source code from disk.

        Args:
            tool_id: The tool to load.

        Returns:
            The full Python source as a string.

        Raises:
            KeyError: If the tool_id is not in the registry.
            IntegrityError: If the on-disk file's hash doesn't match
                the stored ``code_sha256``.
            FileNotFoundError: If the .py file is missing.
        """

        row = self.get(tool_id)
        if row is None:
            raise KeyError(f"unknown tool_id: {tool_id!r}")

        absolute_path = self._paths.root / row.code_path
        code_bytes = absolute_path.read_bytes()

        # Verify integrity on every read. Cheap (SHA-256 on small files
        # is microseconds) and it catches manual edits or corruption.
        actual_sha = hashlib.sha256(code_bytes).hexdigest()
        if actual_sha != row.code_sha256:
            raise IntegrityError(
                f"code file {absolute_path} for tool {tool_id!r} "
                f"has hash {actual_sha} but registry expects {row.code_sha256}"
            )

        return code_bytes.decode("utf-8")

    @property
    def catalog_version(self) -> int:
        """Current catalog version. Bumps on every insert / status change.

        Cheap to call; reads in-memory mirror, not the database.
        """
        return self._catalog_version

    # ------------------------------------------------------------------
    # Startup recovery
    # ------------------------------------------------------------------

    def verify_integrity(self) -> list[tuple[str, str]]:
        """Re-check every tool's on-disk code against its stored hash.

        Called on server startup. For any mismatch or missing file,
        the tool is flipped to UNAUTHORIZED so it cannot be invoked
        until a human investigates.

        Returns:
            A list of ``(tool_id, reason)`` tuples for tools that
            were flipped. Empty list means everything is clean.
        """

        flipped: list[tuple[str, str]] = []
        for row in self.list_all():
            absolute_path = self._paths.root / row.code_path
            try:
                code_bytes = absolute_path.read_bytes()
            except FileNotFoundError:
                # File is gone. Mark UNAUTHORIZED.
                if row.status != "UNAUTHORIZED":
                    self.set_status(row.tool_id, "UNAUTHORIZED")
                    flipped.append((row.tool_id, "code file missing"))
                continue

            # File exists; check hash.
            actual_sha = hashlib.sha256(code_bytes).hexdigest()
            if actual_sha != row.code_sha256:
                if row.status != "UNAUTHORIZED":
                    self.set_status(row.tool_id, "UNAUTHORIZED")
                    flipped.append((row.tool_id, "code hash mismatch"))

        return flipped

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        """Return the SQLite connection or raise if initialize() wasn't called.

        Saves every method from having to assert against ``self._conn is None``.
        """
        if self._conn is None:
            raise RuntimeError("Registry not initialized — call initialize() first")
        return self._conn

    def _bump_catalog_version_locked(self) -> None:
        """Increment the catalog version. MUST be called with _db_lock held."""

        self._catalog_version += 1
        self._require_conn().execute(
            "UPDATE server_meta SET value = ? WHERE key = ?",
            (str(self._catalog_version), "catalog_version"),
        )

    @staticmethod
    def _row_to_toolrow(column_names: list[str], row: tuple) -> ToolRow:
        """Convert a raw sqlite row tuple into a frozen ``ToolRow``.

        Doing it by name (not position) means schema-add-column changes
        won't break this helper.
        """

        data = dict(zip(column_names, row))
        return ToolRow(
            tool_id=data["tool_id"],
            name=data["name"],
            description=data["description"],
            code_path=data["code_path"],
            code_sha256=data["code_sha256"],
            schema_json=data["schema_json"],
            status=data["status"],
            intent=data.get("intent"),
            created_at=data["created_at"],
            status_updated_at=data.get("status_updated_at"),
            call_count=data.get("call_count", 0) or 0,
            last_called_at=data.get("last_called_at"),
        )


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string with millisecond precision."""

    # Microsecond precision is overkill in logs; trim to milliseconds
    # for readability. Keeping the explicit "Z" suffix avoids any
    # ambiguity about timezone.
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
