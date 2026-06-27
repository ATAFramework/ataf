"""Tests for the StoragePaths value object."""

from pathlib import Path

from ataf.server.storage import StoragePaths, default_storage


def test_default_storage_resolves_to_absolute(tmp_path: Path) -> None:
    """default_storage must return an absolute path even given a relative input."""

    # Pass a relative path; expect resolution to absolute.
    paths = default_storage(tmp_path / "ataf_data")
    assert paths.root.is_absolute()


def test_paths_derive_correctly(tmp_path: Path) -> None:
    """Each derived path should hang off root with the expected name."""

    paths = StoragePaths(root=tmp_path)

    # SQLite database
    assert paths.sqlite_path == tmp_path / "ataf.sqlite"

    # Tools directory
    assert paths.tools_dir == tmp_path / "tools"

    # Logs directory + deployment log file
    assert paths.logs_dir == tmp_path / "logs"
    assert paths.deployment_log == tmp_path / "logs" / "deployment.jsonl"

    # Tool code path uses the tool_id as filename stem
    assert paths.tool_code_path("circle_area_v1") == tmp_path / "tools" / "circle_area_v1.py"


def test_ensure_exists_creates_tree(tmp_path: Path) -> None:
    """ensure_exists must create root/tools/logs idempotently."""

    paths = StoragePaths(root=tmp_path / "new_data")

    # First call: directories should not exist yet
    assert not paths.root.exists()

    paths.ensure_exists()

    # All three directories now exist
    assert paths.root.is_dir()
    assert paths.tools_dir.is_dir()
    assert paths.logs_dir.is_dir()

    # Idempotency: a second call must not raise
    paths.ensure_exists()
