"""Tests for the subprocess tool executor."""

from pathlib import Path

import pytest

from ataf.server.builder import Builder
from ataf.server.eventlog import DeploymentEventLog
from ataf.server.executor import Executor, ToolExecutionError
from ataf.server.registry import Registry
from ataf.server.storage import StoragePaths


CIRCLE_AREA_CODE = '''def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius of the circle.
    """
    import math
    return math.pi * radius ** 2
'''


def _deploy(tmp_path: Path, code: str) -> tuple[Registry, str]:
    """Helper: deploy a tool and return (registry, tool_id)."""

    paths = StoragePaths(root=tmp_path)
    paths.ensure_exists()
    registry = Registry(paths)
    registry.initialize()
    builder = Builder(registry, DeploymentEventLog(paths.deployment_log))
    result = builder.build(code=code, intent="t")
    return registry, result.tool_id


def test_invoke_returns_result(tmp_path: Path) -> None:
    """A normal tool call returns the computed value."""

    registry, tool_id = _deploy(tmp_path, CIRCLE_AREA_CODE)
    executor = Executor(registry)

    result = executor.invoke(tool_id, {"radius": 10})
    assert result == pytest.approx(314.159265, rel=1e-6)


def test_invoke_records_call_count(tmp_path: Path) -> None:
    """A successful invocation bumps the tool's call_count."""

    registry, tool_id = _deploy(tmp_path, CIRCLE_AREA_CODE)
    executor = Executor(registry)

    executor.invoke(tool_id, {"radius": 1})
    executor.invoke(tool_id, {"radius": 2})

    row = registry.get(tool_id)
    assert row is not None
    assert row.call_count == 2


def test_invoke_tool_that_raises_is_execution_error(tmp_path: Path) -> None:
    """A tool that raises surfaces as TOOL_EXECUTION_ERROR with the reason."""

    code = '''def divide(a: int, b: int) -> float:
    """Divide a by b.

    Args:
        a: numerator.
        b: denominator.
    """
    return a / b
'''
    registry, tool_id = _deploy(tmp_path, code)
    executor = Executor(registry)

    with pytest.raises(ToolExecutionError) as exc:
        executor.invoke(tool_id, {"a": 1, "b": 0})
    assert exc.value.code == "TOOL_EXECUTION_ERROR"
    assert "ZeroDivisionError" in exc.value.message


def test_invoke_timeout(tmp_path: Path) -> None:
    """A tool that runs too long is killed and reported as TOOL_TIMEOUT."""

    code = '''def spin(seconds: int) -> int:
    """Spin for a while.

    Args:
        seconds: how long to sleep.
    """
    import time
    time.sleep(seconds)
    return seconds
'''
    registry, tool_id = _deploy(tmp_path, code)
    # Tight timeout so the test stays fast.
    executor = Executor(registry, timeout_seconds=0.5)

    with pytest.raises(ToolExecutionError) as exc:
        executor.invoke(tool_id, {"seconds": 5})
    assert exc.value.code == "TOOL_TIMEOUT"


def test_invoke_unknown_tool_raises_keyerror(tmp_path: Path) -> None:
    """Invoking a non-existent tool raises KeyError."""

    registry, _ = _deploy(tmp_path, CIRCLE_AREA_CODE)
    executor = Executor(registry)

    with pytest.raises(KeyError):
        executor.invoke("does_not_exist_v1", {})


def test_invoke_non_serializable_result_is_error(tmp_path: Path) -> None:
    """A tool returning a non-JSON value fails cleanly, not silently."""

    code = '''def make_set(n: int) -> set:
    """Return a set (not JSON-serializable).

    Args:
        n: size.
    """
    return set(range(n))
'''
    registry, tool_id = _deploy(tmp_path, code)
    executor = Executor(registry)

    with pytest.raises(ToolExecutionError):
        executor.invoke(tool_id, {"n": 3})
