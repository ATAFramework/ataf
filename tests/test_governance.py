"""Tests for the server-side governance gate (Layer 2)."""

from pathlib import Path

import pytest

from ataf.server.governance import Governance, NotAuthorizedError
from ataf.server.registry import Registry
from ataf.server.storage import StoragePaths


CODE = '''def f(x: int) -> int:
    """Echo.

    Args:
        x: a number.
    """
    return x
'''


def _registry_with_tool(tmp_path: Path) -> tuple[Registry, str]:
    """Helper: insert one PENDING_REVIEW tool, return (registry, tool_id)."""

    paths = StoragePaths(root=tmp_path)
    paths.ensure_exists()
    registry = Registry(paths)
    registry.initialize()
    registry.insert(
        tool_id="f_v1", name="f", description="echo",
        code=CODE, schema_json="{}", intent="t",
    )
    return registry, "f_v1"


def test_authorized_tool_is_invokable(tmp_path: Path) -> None:
    """An AUTHORIZED tool passes the gate."""

    registry, tool_id = _registry_with_tool(tmp_path)
    registry.set_status(tool_id, "AUTHORIZED")

    governance = Governance(registry)
    row = governance.ensure_invokable(tool_id)
    assert row.tool_id == tool_id


def test_pending_tool_refused_by_default(tmp_path: Path) -> None:
    """A PENDING_REVIEW tool is refused unless the flag is set."""

    registry, tool_id = _registry_with_tool(tmp_path)

    governance = Governance(registry)
    with pytest.raises(NotAuthorizedError) as exc:
        governance.ensure_invokable(tool_id)
    assert exc.value.code == "TOOL_NOT_AUTHORIZED"
    assert exc.value.status == "PENDING_REVIEW"


def test_pending_tool_allowed_when_flag_set(tmp_path: Path) -> None:
    """allow_pending_invocation lets PENDING tools through (dev/demo)."""

    registry, tool_id = _registry_with_tool(tmp_path)

    governance = Governance(registry, allow_pending_invocation=True)
    row = governance.ensure_invokable(tool_id)
    assert row.tool_id == tool_id


def test_rejected_tool_always_refused(tmp_path: Path) -> None:
    """An UNAUTHORIZED tool is refused even with the pending flag on."""

    registry, tool_id = _registry_with_tool(tmp_path)
    registry.set_status(tool_id, "UNAUTHORIZED")

    governance = Governance(registry, allow_pending_invocation=True)
    with pytest.raises(NotAuthorizedError) as exc:
        governance.ensure_invokable(tool_id)
    assert exc.value.status == "UNAUTHORIZED"


def test_unknown_tool_raises_keyerror(tmp_path: Path) -> None:
    """A missing tool is a KeyError (404), not a NotAuthorizedError (403)."""

    registry, _ = _registry_with_tool(tmp_path)

    governance = Governance(registry)
    with pytest.raises(KeyError):
        governance.ensure_invokable("missing_v1")
