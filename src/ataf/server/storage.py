"""Filesystem layout for the ATAF server.

This module owns one thing only: **where files live on disk**. It does
no IO and no business logic. The registry, event log, and builder all
import paths from here so that the layout is defined in exactly one place.

Layout (see DESIGN.md §8.2):

    ataf_data/                       # root, configurable
        ataf.sqlite                  # metadata index
        tools/
            circle_area_v1.py        # one file per deployed tool
            rectangle_area_v1.py
        logs/
            deployment.jsonl         # append-only event log

The root defaults to ``./ataf_data/`` (relative to the process working
directory), but can be overridden by passing a different path to
``StoragePaths(root=...)`` — useful for tests that need an isolated
temp directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoragePaths:
    """Resolved filesystem paths for one ATAF data directory.

    All paths are absolute after ``ensure_exists()`` is called.

    Attributes:
        root: The ATAF data root. Everything else hangs off this.
        sqlite_path: SQLite metadata database location.
        tools_dir: Directory holding generated tool ``.py`` files.
        logs_dir: Directory holding append-only event logs.
        deployment_log: The deployment event log file path.
    """

    root: Path

    @property
    def sqlite_path(self) -> Path:
        """Absolute path to the metadata SQLite database."""
        return self.root / "ataf.sqlite"

    @property
    def tools_dir(self) -> Path:
        """Absolute path to the directory of generated tool code files."""
        return self.root / "tools"

    @property
    def logs_dir(self) -> Path:
        """Absolute path to the logs directory."""
        return self.root / "logs"

    @property
    def deployment_log(self) -> Path:
        """Absolute path to the append-only deployment event log."""
        return self.logs_dir / "deployment.jsonl"

    def tool_code_path(self, tool_id: str) -> Path:
        """Resolve the on-disk code path for a given tool_id.

        Args:
            tool_id: The tool's unique identifier (e.g., ``circle_area_v1``).

        Returns:
            Absolute path where this tool's source code lives.
        """
        # Tool IDs are validated upstream to be safe filename strings
        # (alphanumeric + underscore). We append .py here; never let
        # tool_id carry the extension itself.
        return self.tools_dir / f"{tool_id}.py"

    def ensure_exists(self) -> None:
        """Create the data directory tree if it doesn't already exist.

        Idempotent. Safe to call on every server startup.
        """

        # Build the directory tree. parents=True walks up if root itself
        # doesn't exist; exist_ok=True makes the call idempotent.
        self.root.mkdir(parents=True, exist_ok=True)
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


def default_storage(root: str | Path = "ataf_data") -> StoragePaths:
    """Convenience constructor for a StoragePaths rooted at ``root``.

    Args:
        root: Either a string or a Path. Will be resolved to an
            absolute path so downstream code never has to worry about
            CWD changes.

    Returns:
        A frozen StoragePaths instance with an absolute root.
    """

    # Resolve to an absolute path immediately so a later os.chdir()
    # by user code can't break path references.
    resolved = Path(root).resolve()
    return StoragePaths(root=resolved)
