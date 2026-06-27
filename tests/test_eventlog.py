"""Tests for the JSONL deployment event log."""

from pathlib import Path

from ataf.server.eventlog import DeploymentEventLog
from ataf.server.storage import StoragePaths


def _make_log(tmp_path: Path) -> DeploymentEventLog:
    """Helper: create a StoragePaths tree and return a fresh log writer."""
    paths = StoragePaths(root=tmp_path)
    paths.ensure_exists()
    return DeploymentEventLog(paths.deployment_log)


def test_record_writes_one_line_per_event(tmp_path: Path) -> None:
    """Each record() call should append exactly one line to the file."""

    log = _make_log(tmp_path)

    log.record("propose", tool_id="circle_area_v1", intent="area of circle")
    log.record("deploy", tool_id="circle_area_v1", build_duration_ms=42)

    events = log.read_all()
    assert len(events) == 2

    # Order preserved
    assert events[0]["event"] == "propose"
    assert events[1]["event"] == "deploy"

    # Custom fields round-trip
    assert events[0]["intent"] == "area of circle"
    assert events[1]["build_duration_ms"] == 42


def test_record_adds_ts_field_automatically(tmp_path: Path) -> None:
    """Every event must carry a 'ts' field set by the writer."""

    log = _make_log(tmp_path)
    log.record("approve", tool_id="t1", actor="rajesh")

    events = log.read_all()
    assert "ts" in events[0]

    # Format sanity: ISO-8601 UTC ending in Z.
    assert events[0]["ts"].endswith("Z")


def test_read_all_empty_when_no_log_file(tmp_path: Path) -> None:
    """read_all() must return [] if nothing has been written yet."""

    log = _make_log(tmp_path)

    # No record() calls yet, file does not exist.
    assert log.read_all() == []
