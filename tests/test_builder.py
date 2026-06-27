"""Tests for the tool builder — validation, schema generation, persistence.

These tests drive ``Builder`` directly (no FastAPI app, no build lock),
which is exactly why the builder was kept a pure unit: every rule in
DESIGN.md §3 and the schema-generation logic is testable in isolation.
"""

import json
from pathlib import Path

import pytest

from ataf.server.builder import (
    Builder,
    BuildResult,
    CodeValidationError,
    DeploymentError,
)
from ataf.server.eventlog import DeploymentEventLog
from ataf.server.registry import Registry
from ataf.server.storage import StoragePaths


# A canonical, fully-valid tool used across the happy-path tests.
CIRCLE_AREA_CODE = '''def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius of the circle, in any unit.

    Returns:
        The area in the squared unit of the radius.
    """
    import math
    return math.pi * radius ** 2
'''


def _make_builder(tmp_path: Path) -> tuple[Builder, Registry, DeploymentEventLog]:
    """Helper: wire a registry, event log, and builder over a temp dir."""

    paths = StoragePaths(root=tmp_path)
    paths.ensure_exists()

    registry = Registry(paths)
    registry.initialize()

    event_log = DeploymentEventLog(paths.deployment_log)
    builder = Builder(registry, event_log)
    return builder, registry, event_log


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------

def test_build_deploys_valid_tool(tmp_path: Path) -> None:
    """A valid tool is persisted, PENDING_REVIEW, with a usable result."""

    builder, registry, _ = _make_builder(tmp_path)

    result = builder.build(code=CIRCLE_AREA_CODE, intent="compute circle area")

    assert isinstance(result, BuildResult)
    assert result.tool_id == "circle_area_v1"
    assert result.name == "circle_area"
    assert result.description == "Compute the area of a circle."
    assert result.invoke_uri == "/tools/circle_area_v1/invoke"
    assert result.status == "PENDING_REVIEW"

    # The tool is actually in the registry and code round-trips.
    row = registry.get("circle_area_v1")
    assert row is not None
    assert row.status == "PENDING_REVIEW"
    assert registry.get_code("circle_area_v1") == CIRCLE_AREA_CODE


def test_build_generates_correct_schema(tmp_path: Path) -> None:
    """The generated input schema maps types and descriptions correctly."""

    builder, _, _ = _make_builder(tmp_path)
    result = builder.build(code=CIRCLE_AREA_CODE, intent="t")

    assert result.input_schema == {
        "type": "object",
        "properties": {
            "radius": {
                "type": "number",
                "description": "The radius of the circle, in any unit.",
            }
        },
        "required": ["radius"],
    }


def test_build_logs_propose_and_deploy_events(tmp_path: Path) -> None:
    """A successful build logs both a propose and a deploy event."""

    builder, _, event_log = _make_builder(tmp_path)
    builder.build(code=CIRCLE_AREA_CODE, intent="my intent", agent_id="agent-7")

    events = event_log.read_all()
    event_types = [e["event"] for e in events]
    assert event_types == ["propose", "deploy"]

    propose = events[0]
    assert propose["tool_id"] == "circle_area_v1"
    assert propose["intent"] == "my intent"
    assert propose["agent_id"] == "agent-7"

    deploy = events[1]
    assert deploy["tool_id"] == "circle_area_v1"
    assert deploy["status"] == "PENDING_REVIEW"
    assert "build_duration_ms" in deploy


def test_build_increments_tool_id_on_name_collision(tmp_path: Path) -> None:
    """Two tools with the same function name get _v1 then _v2."""

    builder, _, _ = _make_builder(tmp_path)

    first = builder.build(code=CIRCLE_AREA_CODE, intent="t")
    # Same name, different body so the code bytes differ.
    second_code = CIRCLE_AREA_CODE + "\n# variant\n"
    second = builder.build(code=second_code, intent="t")

    assert first.tool_id == "circle_area_v1"
    assert second.tool_id == "circle_area_v2"


# ----------------------------------------------------------------------
# Schema generation — types, optionals, containers
# ----------------------------------------------------------------------

def test_schema_maps_primitive_types(tmp_path: Path) -> None:
    """int/str/bool annotations map to the right JSON types."""

    code = '''def make_label(count: int, name: str, loud: bool) -> str:
    """Build a label.

    Args:
        count: How many.
        name: The name.
        loud: Whether to shout.
    """
    return name
'''
    builder, _, _ = _make_builder(tmp_path)
    schema = builder.build(code=code, intent="t").input_schema

    props = schema["properties"]
    assert props["count"]["type"] == "integer"
    assert props["name"]["type"] == "string"
    assert props["loud"]["type"] == "boolean"
    assert set(schema["required"]) == {"count", "name", "loud"}


def test_schema_marks_defaulted_params_optional(tmp_path: Path) -> None:
    """Parameters with defaults are not in the required list."""

    code = '''def greet(name: str, greeting: str = "hi") -> str:
    """Greet someone.

    Args:
        name: Who to greet.
        greeting: The greeting word.
    """
    return greeting + " " + name
'''
    builder, _, _ = _make_builder(tmp_path)
    schema = builder.build(code=code, intent="t").input_schema

    assert schema["required"] == ["name"]
    assert "greeting" in schema["properties"]


def test_schema_handles_optional_and_containers(tmp_path: Path) -> None:
    """Optional[T], T | None, and list[T] map sensibly."""

    code = '''def process(items: list[int], tag: str | None = None) -> int:
    """Process items.

    Args:
        items: Numbers to process.
        tag: Optional label.
    """
    return len(items)
'''
    builder, _, _ = _make_builder(tmp_path)
    schema = builder.build(code=code, intent="t").input_schema

    assert schema["properties"]["items"] == {
        "type": "array",
        "items": {"type": "integer"},
        "description": "Numbers to process.",
    }
    # `str | None` unwraps to string; defaulted, so not required.
    assert schema["properties"]["tag"]["type"] == "string"
    assert schema["required"] == ["items"]


def test_schema_unknown_type_is_unconstrained(tmp_path: Path) -> None:
    """An exotic annotation degrades to an unconstrained schema, not an error."""

    code = '''def identity(value: memoryview) -> memoryview:
    """Return the value unchanged.

    Args:
        value: Anything.
    """
    return value
'''
    builder, _, _ = _make_builder(tmp_path)
    schema = builder.build(code=code, intent="t").input_schema

    # No "type" key — accepts any JSON value.
    assert schema["properties"]["value"] == {"description": "Anything."}


def test_schema_no_required_key_when_all_optional(tmp_path: Path) -> None:
    """If every parameter has a default, the schema omits 'required'."""

    code = '''def ping(n: int = 1) -> int:
    """Ping n times.

    Args:
        n: Number of pings.
    """
    return n
'''
    builder, _, _ = _make_builder(tmp_path)
    schema = builder.build(code=code, intent="t").input_schema

    assert "required" not in schema


# ----------------------------------------------------------------------
# Validation failures — each maps to a stable error code
# ----------------------------------------------------------------------

def test_reject_syntax_error(tmp_path: Path) -> None:
    """Unparseable code raises SYNTAX_ERROR."""

    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code="def broken(:\n    pass\n", intent="t")
    assert exc.value.code == "SYNTAX_ERROR"


def test_reject_missing_docstring(tmp_path: Path) -> None:
    """A function with no docstring raises MISSING_DOCSTRING."""

    code = "def f(x: int) -> int:\n    return x\n"
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "MISSING_DOCSTRING"


def test_reject_missing_return_annotation(tmp_path: Path) -> None:
    """A function with no return annotation raises MISSING_RETURN_ANNOTATION."""

    code = 'def f(x: int):\n    """Doc."""\n    return x\n'
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "MISSING_RETURN_ANNOTATION"


def test_reject_missing_param_annotation(tmp_path: Path) -> None:
    """An unannotated parameter raises MISSING_TYPE_ANNOTATION."""

    code = 'def f(x) -> int:\n    """Doc."""\n    return x\n'
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "MISSING_TYPE_ANNOTATION"


def test_reject_non_stdlib_import(tmp_path: Path) -> None:
    """A non-stdlib import raises NON_STDLIB_IMPORT."""

    code = '''def f(x: int) -> int:
    """Doc.

    Args:
        x: A number.
    """
    import numpy
    return x
'''
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "NON_STDLIB_IMPORT"


def test_reject_from_non_stdlib_import(tmp_path: Path) -> None:
    """A `from <pkg> import` of a non-stdlib package is rejected too."""

    code = '''def f(x: int) -> int:
    """Doc.

    Args:
        x: A number.
    """
    from requests import get
    return x
'''
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "NON_STDLIB_IMPORT"


def test_reject_multiple_top_level_functions(tmp_path: Path) -> None:
    """More than one top-level definition raises NOT_SINGLE_FUNCTION."""

    code = '''def a(x: int) -> int:
    """Doc."""
    return x

def b(y: int) -> int:
    """Doc."""
    return y
'''
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "NOT_SINGLE_FUNCTION"


def test_reject_module_level_import(tmp_path: Path) -> None:
    """A module-level import (extra top-level stmt) raises NOT_SINGLE_FUNCTION."""

    code = '''import math

def f(x: float) -> float:
    """Doc.

    Args:
        x: A number.
    """
    return math.sqrt(x)
'''
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "NOT_SINGLE_FUNCTION"


def test_reject_varargs(tmp_path: Path) -> None:
    """*args / **kwargs raise VARIADIC_NOT_SUPPORTED."""

    code = '''def f(*args: int) -> int:
    """Doc."""
    return 0
'''
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "VARIADIC_NOT_SUPPORTED"


def test_reject_async_function(tmp_path: Path) -> None:
    """Async tools raise ASYNC_NOT_SUPPORTED."""

    code = '''async def f(x: int) -> int:
    """Doc.

    Args:
        x: A number.
    """
    return x
'''
    builder, _, _ = _make_builder(tmp_path)
    with pytest.raises(CodeValidationError) as exc:
        builder.build(code=code, intent="t")
    assert exc.value.code == "ASYNC_NOT_SUPPORTED"


def test_validation_failure_logs_deploy_failed(tmp_path: Path) -> None:
    """A rejected proposal logs a deploy_failed event with the reason."""

    builder, _, event_log = _make_builder(tmp_path)
    code = "def f(x: int) -> int:\n    return x\n"  # no docstring

    with pytest.raises(CodeValidationError):
        builder.build(code=code, intent="t", agent_id="agent-9")

    events = event_log.read_all()
    assert len(events) == 1
    assert events[0]["event"] == "deploy_failed"
    assert events[0]["reason"] == "MISSING_DOCSTRING"
    assert events[0]["agent_id"] == "agent-9"


def test_validation_failure_persists_nothing(tmp_path: Path) -> None:
    """A rejected proposal leaves the registry empty (no orphan rows/files)."""

    builder, registry, _ = _make_builder(tmp_path)
    code = "def f(x: int) -> int:\n    return x\n"  # no docstring

    with pytest.raises(CodeValidationError):
        builder.build(code=code, intent="t")

    assert registry.list_all() == []


# ----------------------------------------------------------------------
# Deployment failures — valid code, server-side fault → TOOL_NOT_DEPLOYED
# ----------------------------------------------------------------------

def test_deployment_failure_raises_tool_not_deployed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registry insert failure on valid code raises DeploymentError."""

    builder, registry, _ = _make_builder(tmp_path)

    # Simulate a server-side persistence failure (e.g. disk full, locked
    # DB) by making the registry insert blow up. The code itself is valid.
    def boom(**kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(registry, "insert", boom)

    with pytest.raises(DeploymentError) as exc:
        builder.build(code=CIRCLE_AREA_CODE, intent="t")

    # The stable wire code is fixed regardless of the underlying cause,
    # and the original message is preserved for the operator.
    assert exc.value.code == "TOOL_NOT_DEPLOYED"
    assert "disk full" in exc.value.message


def test_deployment_failure_logs_deploy_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment failure logs propose then a deploy_failed event."""

    builder, registry, event_log = _make_builder(tmp_path)

    def boom(**kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(registry, "insert", boom)

    with pytest.raises(DeploymentError):
        builder.build(code=CIRCLE_AREA_CODE, intent="t", agent_id="agent-3")

    events = event_log.read_all()
    # We got far enough to allocate a tool_id and log propose before the
    # insert blew up, so both events are present.
    assert [e["event"] for e in events] == ["propose", "deploy_failed"]
    failed = events[-1]
    assert failed["reason"] == "TOOL_NOT_DEPLOYED"
    assert "disk full" in failed["message"]
    assert failed["agent_id"] == "agent-3"


def test_deployment_error_is_distinct_from_validation_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DeploymentError must not be confused with CodeValidationError."""

    builder, registry, _ = _make_builder(tmp_path)
    monkeypatch.setattr(
        registry, "insert", lambda **k: (_ for _ in ()).throw(OSError("io"))
    )

    # A server-side failure is a DeploymentError, not a validation error —
    # the two map to different HTTP statuses (500 vs 400).
    with pytest.raises(DeploymentError):
        builder.build(code=CIRCLE_AREA_CODE, intent="t")
    assert not issubclass(DeploymentError, CodeValidationError)


def test_build_result_schema_is_json_serializable(tmp_path: Path) -> None:
    """The stored schema_json must equal the result.input_schema round-tripped."""

    builder, registry, _ = _make_builder(tmp_path)
    result = builder.build(code=CIRCLE_AREA_CODE, intent="t")

    row = registry.get(result.tool_id)
    assert row is not None
    assert json.loads(row.schema_json) == result.input_schema
