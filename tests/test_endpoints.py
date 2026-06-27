"""End-to-end endpoint tests using FastAPI's TestClient.

These drive the real app (built over a temp data dir) through the full
HTTP surface: catalog, propose, governance refusal, approve, invoke. They
are the closest thing to the v0.1 exit-criteria demo, in test form.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ataf.server.main import create_app
from ataf.server.storage import StoragePaths


CIRCLE_AREA_CODE = '''def circle_area(radius: float) -> float:
    """Compute the area of a circle.

    Args:
        radius: The radius of the circle.
    """
    import math
    return math.pi * radius ** 2
'''


def _client(tmp_path: Path, **kwargs) -> TestClient:
    """Build a TestClient over an app rooted at an isolated temp dir."""

    app = create_app(StoragePaths(root=tmp_path), **kwargs)
    return TestClient(app)


def test_catalog_starts_empty(tmp_path: Path) -> None:
    """A fresh server reports an empty catalog at version 0."""

    client = _client(tmp_path)
    body = client.get("/tools").json()
    assert body["tools"] == []
    assert body["catalog_version"] == 0


def test_propose_deploys_tool(tmp_path: Path) -> None:
    """POST /tools/propose deploys a valid tool as PENDING_REVIEW."""

    client = _client(tmp_path)
    resp = client.post(
        "/tools/propose",
        json={"intent": "area", "code": CIRCLE_AREA_CODE},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "DEPLOYED"
    assert body["tool_id"] == "circle_area_v1"
    assert body["tool_status"] == "PENDING_REVIEW"
    assert body["input_schema"]["properties"]["radius"]["type"] == "number"


def test_propose_invalid_code_returns_400(tmp_path: Path) -> None:
    """Malformed code returns 400 with the stable validation error code."""

    client = _client(tmp_path)
    resp = client.post(
        "/tools/propose",
        json={"intent": "x", "code": "def f(x): return x\n"},  # no docstring/types
    )
    assert resp.status_code == 400
    assert resp.json()["error"] in (
        "MISSING_RETURN_ANNOTATION",
        "MISSING_TYPE_ANNOTATION",
        "MISSING_DOCSTRING",
    )


def test_invoke_pending_tool_is_403(tmp_path: Path) -> None:
    """Invoking a still-pending tool is refused with 403 TOOL_NOT_AUTHORIZED."""

    client = _client(tmp_path)
    client.post("/tools/propose", json={"intent": "a", "code": CIRCLE_AREA_CODE})

    resp = client.post("/tools/circle_area_v1/invoke", json={"args": {"radius": 10}})
    assert resp.status_code == 403
    assert resp.json()["error"] == "TOOL_NOT_AUTHORIZED"


def test_full_loop_propose_approve_invoke(tmp_path: Path) -> None:
    """The full v0.1 loop: propose -> approve -> invoke returns the result."""

    client = _client(tmp_path)
    client.post("/tools/propose", json={"intent": "a", "code": CIRCLE_AREA_CODE})

    # Human approval flips it to AUTHORIZED.
    approve = client.post("/admin/tools/circle_area_v1/approve")
    assert approve.status_code == 200
    assert approve.json()["status"] == "AUTHORIZED"

    # Now invocation succeeds.
    resp = client.post("/tools/circle_area_v1/invoke", json={"args": {"radius": 10}})
    assert resp.status_code == 200
    assert resp.json()["result"] == pytest.approx(314.159265, rel=1e-6)


def test_allow_pending_flag_lets_invoke_through(tmp_path: Path) -> None:
    """With allow_pending_invocation, a pending tool is invokable."""

    client = _client(tmp_path, allow_pending_invocation=True)
    client.post("/tools/propose", json={"intent": "a", "code": CIRCLE_AREA_CODE})

    resp = client.post("/tools/circle_area_v1/invoke", json={"args": {"radius": 2}})
    assert resp.status_code == 200
    assert resp.json()["result"] == pytest.approx(12.566370, rel=1e-6)


def test_auto_authorize_deploys_authorized_and_invokable(tmp_path: Path) -> None:
    """With auto_authorize, a proposed tool is AUTHORIZED and immediately usable."""

    client = _client(tmp_path, auto_authorize=True)

    # Propose: response should already report AUTHORIZED, no human step.
    resp = client.post("/tools/propose", json={"intent": "a", "code": CIRCLE_AREA_CODE})
    assert resp.status_code == 200
    assert resp.json()["tool_status"] == "AUTHORIZED"

    # And it can be invoked right away.
    invoke = client.post("/tools/circle_area_v1/invoke", json={"args": {"radius": 10}})
    assert invoke.status_code == 200
    assert invoke.json()["result"] == pytest.approx(314.159265, rel=1e-6)

    # The catalog shows it AUTHORIZED.
    catalog = client.get("/tools").json()
    assert catalog["tools"][0]["status"] == "AUTHORIZED"


def test_invoke_unknown_tool_is_404(tmp_path: Path) -> None:
    """Invoking a non-existent tool returns 404 TOOL_NOT_FOUND."""

    client = _client(tmp_path)
    resp = client.post("/tools/nope_v1/invoke", json={"args": {}})
    assert resp.status_code == 404
    assert resp.json()["error"] == "TOOL_NOT_FOUND"


def test_reject_then_invoke_is_403(tmp_path: Path) -> None:
    """A rejected tool cannot be invoked even though it stays in the registry."""

    client = _client(tmp_path)
    client.post("/tools/propose", json={"intent": "a", "code": CIRCLE_AREA_CODE})
    client.post("/admin/tools/circle_area_v1/reject")

    resp = client.post("/tools/circle_area_v1/invoke", json={"args": {"radius": 1}})
    assert resp.status_code == 403

    # ...but it's still listed in the catalog (never auto-deleted).
    catalog = client.get("/tools").json()
    assert any(t["tool_id"] == "circle_area_v1" for t in catalog["tools"])


def test_openapi_and_docs_are_served(tmp_path: Path) -> None:
    """The OpenAPI spec and Swagger docs are auto-served by FastAPI."""

    client = _client(tmp_path)

    spec = client.get("/openapi.json")
    assert spec.status_code == 200
    assert "/tools/propose" in spec.json()["paths"]

    docs = client.get("/docs")
    assert docs.status_code == 200
