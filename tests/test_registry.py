"""Tests for the SQLite + filesystem tool registry."""

import hashlib
from pathlib import Path

import pytest

from ataf.server.registry import IntegrityError, Registry
from ataf.server.storage import StoragePaths


CIRCLE_AREA_CODE = '''def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius of the circle.

    Returns:
        The area.
    """
    import math
    return math.pi * radius ** 2
'''


def _make_registry(tmp_path: Path) -> Registry:
    """Helper: build paths, init a fresh registry."""
    paths = StoragePaths(root=tmp_path)
    paths.ensure_exists()
    registry = Registry(paths)
    registry.initialize()
    return registry


def test_initialize_is_idempotent(tmp_path: Path) -> None:
    """Calling initialize() twice on the same path must not fail."""

    paths = StoragePaths(root=tmp_path)
    paths.ensure_exists()

    r1 = Registry(paths)
    r1.initialize()
    r1.close()

    # Second instance, same data dir: should re-open cleanly.
    r2 = Registry(paths)
    r2.initialize()
    r2.close()


def test_allocate_tool_id_starts_at_v1(tmp_path: Path) -> None:
    """First tool with a given name should get ``_v1``."""

    registry = _make_registry(tmp_path)
    assert registry.allocate_tool_id("circle_area") == "circle_area_v1"


def test_allocate_tool_id_increments_on_collision(tmp_path: Path) -> None:
    """Second tool with same name should get ``_v2``, etc."""

    registry = _make_registry(tmp_path)

    # Insert v1 to claim the slot
    registry.insert(
        tool_id="circle_area_v1",
        name="circle_area",
        description="circle area",
        code=CIRCLE_AREA_CODE,
        schema_json="{}",
        intent="test",
    )

    # Next allocation must skip v1
    assert registry.allocate_tool_id("circle_area") == "circle_area_v2"


def test_allocate_tool_id_rejects_invalid_names(tmp_path: Path) -> None:
    """Non-identifier function names must raise."""

    registry = _make_registry(tmp_path)

    with pytest.raises(ValueError):
        registry.allocate_tool_id("123_starts_with_digit")
    with pytest.raises(ValueError):
        registry.allocate_tool_id("has-dash")
    with pytest.raises(ValueError):
        registry.allocate_tool_id("../escape")


def test_insert_writes_code_to_disk(tmp_path: Path) -> None:
    """insert() must write the .py file with the exact code bytes."""

    registry = _make_registry(tmp_path)
    paths = StoragePaths(root=tmp_path)

    row = registry.insert(
        tool_id="circle_area_v1",
        name="circle_area",
        description="circle area",
        code=CIRCLE_AREA_CODE,
        schema_json='{"type": "object"}',
        intent="testing",
    )

    # File exists at expected path
    code_file = paths.tool_code_path("circle_area_v1")
    assert code_file.exists()

    # Contents match exactly
    assert code_file.read_text() == CIRCLE_AREA_CODE

    # Hash in the row matches what we wrote
    expected_sha = hashlib.sha256(CIRCLE_AREA_CODE.encode()).hexdigest()
    assert row.code_sha256 == expected_sha

    # New tools default to PENDING_REVIEW
    assert row.status == "PENDING_REVIEW"


def test_insert_bumps_catalog_version(tmp_path: Path) -> None:
    """catalog_version must increase by 1 on each successful insert."""

    registry = _make_registry(tmp_path)

    v0 = registry.catalog_version
    registry.insert(
        tool_id="circle_area_v1",
        name="circle_area",
        description="circle area",
        code=CIRCLE_AREA_CODE,
        schema_json="{}",
        intent="t",
    )
    assert registry.catalog_version == v0 + 1


def test_set_status_updates_and_bumps_version(tmp_path: Path) -> None:
    """set_status must change the flag and bump catalog_version."""

    registry = _make_registry(tmp_path)
    registry.insert(
        tool_id="circle_area_v1",
        name="circle_area",
        description="circle area",
        code=CIRCLE_AREA_CODE,
        schema_json="{}",
        intent="t",
    )

    pre_version = registry.catalog_version
    registry.set_status("circle_area_v1", "AUTHORIZED")

    row = registry.get("circle_area_v1")
    assert row is not None
    assert row.status == "AUTHORIZED"
    assert row.status_updated_at is not None
    assert registry.catalog_version == pre_version + 1


def test_set_status_rejects_unknown_tool(tmp_path: Path) -> None:
    """set_status on a missing tool_id must raise KeyError."""

    registry = _make_registry(tmp_path)
    with pytest.raises(KeyError):
        registry.set_status("does_not_exist_v1", "AUTHORIZED")


def test_get_code_verifies_hash(tmp_path: Path) -> None:
    """get_code must raise IntegrityError if the on-disk file is tampered with."""

    registry = _make_registry(tmp_path)
    paths = StoragePaths(root=tmp_path)

    registry.insert(
        tool_id="circle_area_v1",
        name="circle_area",
        description="circle area",
        code=CIRCLE_AREA_CODE,
        schema_json="{}",
        intent="t",
    )

    # Tamper with the on-disk file
    paths.tool_code_path("circle_area_v1").write_text("def circle_area(): return 99\n")

    with pytest.raises(IntegrityError):
        registry.get_code("circle_area_v1")


def test_verify_integrity_flips_missing_files(tmp_path: Path) -> None:
    """verify_integrity must mark UNAUTHORIZED any tool whose .py file is gone."""

    registry = _make_registry(tmp_path)
    paths = StoragePaths(root=tmp_path)

    registry.insert(
        tool_id="circle_area_v1",
        name="circle_area",
        description="circle area",
        code=CIRCLE_AREA_CODE,
        schema_json="{}",
        intent="t",
    )
    registry.set_status("circle_area_v1", "AUTHORIZED")

    # Delete the code file
    paths.tool_code_path("circle_area_v1").unlink()

    flipped = registry.verify_integrity()
    assert flipped == [("circle_area_v1", "code file missing")]

    row = registry.get("circle_area_v1")
    assert row is not None
    assert row.status == "UNAUTHORIZED"


def test_list_all_returns_tools_in_creation_order(tmp_path: Path) -> None:
    """list_all must order tools by created_at ascending."""

    registry = _make_registry(tmp_path)

    registry.insert(
        tool_id="circle_area_v1", name="circle_area",
        description="first", code=CIRCLE_AREA_CODE,
        schema_json="{}", intent="t",
    )
    # Use a slightly different code so the hash differs (and the file
    # write doesn't collide on the same path).
    second_code = CIRCLE_AREA_CODE + "\n# second\n"
    registry.insert(
        tool_id="square_area_v1", name="square_area",
        description="second", code=second_code,
        schema_json="{}", intent="t",
    )

    tools = registry.list_all()
    assert [t.tool_id for t in tools] == ["circle_area_v1", "square_area_v1"]


def test_record_invocation_increments_counter(tmp_path: Path) -> None:
    """record_invocation must bump call_count without bumping catalog_version."""

    registry = _make_registry(tmp_path)
    registry.insert(
        tool_id="circle_area_v1", name="circle_area",
        description="d", code=CIRCLE_AREA_CODE,
        schema_json="{}", intent="t",
    )

    version_before = registry.catalog_version
    registry.record_invocation("circle_area_v1")
    registry.record_invocation("circle_area_v1")

    row = registry.get("circle_area_v1")
    assert row is not None
    assert row.call_count == 2
    assert row.last_called_at is not None
    # Invocation must NOT bump catalog version — the catalog shape is unchanged
    assert registry.catalog_version == version_before


def test_catalog_version_persists_across_restart(tmp_path: Path) -> None:
    """The catalog version counter must survive a process restart."""

    r1 = _make_registry(tmp_path)
    r1.insert(
        tool_id="circle_area_v1", name="circle_area",
        description="d", code=CIRCLE_AREA_CODE,
        schema_json="{}", intent="t",
    )
    version_after_insert = r1.catalog_version
    r1.close()

    # Re-open same data dir
    r2 = Registry(StoragePaths(root=tmp_path))
    r2.initialize()
    assert r2.catalog_version == version_after_insert
    r2.close()
