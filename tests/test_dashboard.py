"""Tests for dashboard.py endpoints."""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from prompt_interceptor.dashboard import app
from prompt_interceptor.interceptor import interceptor as _module_interceptor


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ollama_unavailable():
    """Mock httpx so Ollama target health checks return 503 instantly."""
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        yield


@pytest.fixture
async def client(mock_ollama_unavailable):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# Root / HTML
# ---------------------------------------------------------------------------

async def test_root_returns_html(client):
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "PromptInterceptor" in resp.text


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------

async def test_api_health(client):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


# ---------------------------------------------------------------------------
# /api/target-health
# ---------------------------------------------------------------------------

async def test_api_target_health_unreachable(client):
    # Ollama is mocked as unavailable (see mock_ollama_unavailable fixture)
    resp = await client.get("/api/target-health")
    assert resp.status_code in (200, 503, 504)


# ---------------------------------------------------------------------------
# /api/status
# ---------------------------------------------------------------------------

async def test_api_status_structure(client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await client.get("/api/status")

    assert resp.status_code == 200
    data = resp.json()
    assert "proxy" in data
    assert "rules" in data


# ---------------------------------------------------------------------------
# /api/stats
# ---------------------------------------------------------------------------

async def test_api_stats(client):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.status_code = 503
    mock_resp.text = "no ollama"
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await client.get("/api/stats")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/rules
# ---------------------------------------------------------------------------

async def test_api_rules_returns_list(client):
    resp = await client.get("/api/rules")
    assert resp.status_code == 200
    data = resp.json()
    assert "rules" in data
    assert isinstance(data["rules"], list)


# ---------------------------------------------------------------------------
# /api/enable-rule / /api/disable-rule
# ---------------------------------------------------------------------------

async def test_enable_rule_valid_index(client):
    resp = await client.get("/api/enable-rule/0")
    assert resp.status_code in (200, 404)   # 404 if no rules loaded


async def test_disable_rule_valid_index(client):
    resp = await client.get("/api/disable-rule/0")
    assert resp.status_code in (200, 404)


async def test_enable_rule_invalid_index(client):
    resp = await client.get("/api/enable-rule/9999")
    assert resp.status_code == 404
    assert resp.json()["status"] == "not_found"


async def test_disable_rule_invalid_index(client):
    resp = await client.get("/api/disable-rule/9999")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/logs
# ---------------------------------------------------------------------------

async def test_api_logs(client):
    resp = await client.get("/api/logs")
    assert resp.status_code == 200
    data = resp.json()
    assert "logs" in data
    assert isinstance(data["logs"], list)


async def test_api_logs_with_limit(client):
    resp = await client.get("/api/logs?limit=5")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# /api/raw-logs
# ---------------------------------------------------------------------------

async def test_api_raw_logs(client):
    resp = await client.get("/api/raw-logs")
    assert resp.status_code == 200
    assert "logs" in resp.json()


# ---------------------------------------------------------------------------
# /api/intercept/pending
# ---------------------------------------------------------------------------

async def test_intercept_pending_empty(client):
    resp = await client.get("/api/intercept/pending")
    assert resp.status_code == 200
    data = resp.json()
    assert "pending" in data
    assert "count" in data


# ---------------------------------------------------------------------------
# /api/intercept/{id}/forward|edit|drop — not found
# ---------------------------------------------------------------------------

async def test_intercept_forward_not_found(client):
    resp = await client.post("/api/intercept/nonexistent/forward")
    assert resp.status_code == 404


async def test_intercept_edit_not_found(client):
    resp = await client.post(
        "/api/intercept/nonexistent/edit",
        json={"model": "new"},
    )
    assert resp.status_code == 404


async def test_intercept_drop_not_found(client):
    resp = await client.post("/api/intercept/nonexistent/drop")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/intercept/{id}/forward — found (inject a pending request)
# ---------------------------------------------------------------------------

async def test_intercept_forward_found(client):
    from prompt_interceptor.interceptor import Interceptor, interceptor as _interceptor
    import prompt_interceptor.dashboard as dash_mod

    local_interceptor = Interceptor(intercept_timeout=5.0)
    original = dash_mod.interceptor
    dash_mod.interceptor = local_interceptor

    try:
        rid = "test-rid-fwd"
        # Start an intercept in the background
        task = asyncio.create_task(
            local_interceptor.intercept(rid, "POST", "/api/chat", {}, {"model": "x"})
        )
        await asyncio.sleep(0.05)

        resp = await client.post(f"/api/intercept/{rid}/forward")
        assert resp.status_code == 200
        assert resp.json()["status"] == "forwarded"

        action, _ = await task
        assert action == "forward"
    finally:
        dash_mod.interceptor = original


async def test_intercept_drop_found(client):
    from prompt_interceptor.interceptor import Interceptor
    import prompt_interceptor.dashboard as dash_mod

    local_interceptor = Interceptor(intercept_timeout=5.0)
    original = dash_mod.interceptor
    dash_mod.interceptor = local_interceptor

    try:
        rid = "test-rid-drop"
        task = asyncio.create_task(
            local_interceptor.intercept(rid, "POST", "/", {}, {})
        )
        await asyncio.sleep(0.05)

        resp = await client.post(f"/api/intercept/{rid}/drop")
        assert resp.status_code == 200

        action, _ = await task
        assert action == "drop"
    finally:
        dash_mod.interceptor = original


async def test_intercept_edit_found(client):
    from prompt_interceptor.interceptor import Interceptor
    import prompt_interceptor.dashboard as dash_mod

    local_interceptor = Interceptor(intercept_timeout=5.0)
    original = dash_mod.interceptor
    dash_mod.interceptor = local_interceptor

    try:
        rid = "test-rid-edit"
        task = asyncio.create_task(
            local_interceptor.intercept(rid, "POST", "/", {}, {"model": "old"})
        )
        await asyncio.sleep(0.05)

        resp = await client.post(
            f"/api/intercept/{rid}/edit",
            json={"model": "new"},
        )
        assert resp.status_code == 200

        action, body = await task
        assert action == "edit"
        assert body == {"model": "new"}
    finally:
        dash_mod.interceptor = original


# ---------------------------------------------------------------------------
# /api/mode  (new endpoint)
# ---------------------------------------------------------------------------

async def test_set_mode_passthrough(client):
    resp = await client.post("/api/mode", json={"mode": "passthrough"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "passthrough"


async def test_set_mode_intercept(client):
    resp = await client.post("/api/mode", json={"mode": "intercept"})
    assert resp.status_code == 200
    assert resp.json()["mode"] == "intercept"
    # Reset back to passthrough
    await client.post("/api/mode", json={"mode": "passthrough"})


async def test_set_mode_invalid(client):
    resp = await client.post("/api/mode", json={"mode": "invalid"})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# POST /api/rules  (new endpoint)
# ---------------------------------------------------------------------------

async def test_add_rule_valid(client):
    rule = {
        "match": {"path": "/api/chat", "jsonpath": "$.model"},
        "replace": {"jsonpath": "$.model", "value": "test-model"},
    }
    resp = await client.post("/api/rules", json=rule)
    assert resp.status_code == 201
    assert resp.json()["status"] == "created"


async def test_add_rule_invalid(client):
    resp = await client.post("/api/rules", json={"bad": "data"})
    assert resp.status_code == 400
    assert "error" in resp.json()


# ---------------------------------------------------------------------------
# DELETE /api/rules/{index}  (new endpoint)
# ---------------------------------------------------------------------------

async def test_delete_rule_not_found(client):
    resp = await client.delete("/api/rules/9999")
    assert resp.status_code == 404
    assert resp.json()["status"] == "not_found"


async def test_delete_rule_valid(client):
    # First add a rule, then delete it
    rule = {
        "match": {"path": "/api/generate", "jsonpath": "$.model"},
        "replace": {"jsonpath": "$.model", "value": "to-delete"},
    }
    add_resp = await client.post("/api/rules", json=rule)
    assert add_resp.status_code == 201

    # Get current rule count to find the new index
    list_resp = await client.get("/api/rules")
    rules = list_resp.json()["rules"]
    last_index = len(rules) - 1

    del_resp = await client.delete(f"/api/rules/{last_index}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"


# ---------------------------------------------------------------------------
# POST /api/reset
# ---------------------------------------------------------------------------

async def test_reset_session_clears_logs(client):
    """POST /api/reset calls clear_logs and returns status=reset."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod._logger, "clear_logs", return_value=5) as mock_clear:
        resp = await client.post("/api/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "reset"
    assert data["deleted"] == 5
    mock_clear.assert_called_once()


async def test_reset_session_no_logs(client):
    """POST /api/reset returns deleted=0 when there are no log files."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod._logger, "clear_logs", return_value=0):
        resp = await client.post("/api/reset")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == 0


# ---------------------------------------------------------------------------
# get_app
# ---------------------------------------------------------------------------

def test_get_app():
    from prompt_interceptor.dashboard import get_app
    from fastapi import FastAPI
    assert isinstance(get_app(), FastAPI)


# ---------------------------------------------------------------------------
# GET /favicon.ico — icon file exists  (lines 416-418)
# ---------------------------------------------------------------------------

@pytest.fixture
async def plain_client():
    """Dashboard client without Ollama mock (for favicon/static tests)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def test_favicon_returns_file_when_exists(plain_client):
    """When the icon file exists, /favicon.ico returns 200."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b'\x89PNG\r\n\x1a\n')
        tmp_path = tmp.name

    try:
        import prompt_interceptor.dashboard as dash_mod
        with patch.object(dash_mod.os.path, "exists", return_value=True), \
             patch("prompt_interceptor.dashboard._ICON_PATH", tmp_path):
            resp = await plain_client.get("/favicon.ico")
        assert resp.status_code == 200
    finally:
        os.unlink(tmp_path)


async def test_favicon_returns_404_when_missing(plain_client):
    """When the icon file is absent, /favicon.ico returns 404."""
    import prompt_interceptor.dashboard as dash_mod
    with patch.object(dash_mod.os.path, "exists", return_value=False):
        resp = await plain_client.get("/favicon.ico")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# GET / — static/index.html fallback  (lines 426-427)
# ---------------------------------------------------------------------------

async def test_root_serves_static_index_when_present(plain_client):
    """When static/index.html exists, it is served instead of inline HTML."""
    custom_html = "<html><body>Custom Dashboard</body></html>"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as tmp:
        tmp.write(custom_html)
        tmp_path = tmp.name

    try:
        import prompt_interceptor.dashboard as dash_mod
        original_exists = os.path.exists

        def _fake_exists(path):
            if "static" in str(path) and "index.html" in str(path):
                return True
            return original_exists(path)

        with patch.object(dash_mod.os.path, "exists", side_effect=_fake_exists), \
             patch("builtins.open", return_value=open(tmp_path)):
            resp = await plain_client.get("/")

        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Live Prompts — Guardar / Limpiar buttons in HTML
# ---------------------------------------------------------------------------

async def test_dashboard_html_has_clear_logs_button(client):
    """Dashboard HTML includes the Limpiar button that calls clearLogs()."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "clearLogs()" in resp.text
    assert "Limpiar" in resp.text


async def test_dashboard_html_has_save_logs_button(client):
    """Dashboard HTML includes the Guardar button that calls saveLogs()."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "saveLogs()" in resp.text
    assert "Guardar" in resp.text


async def test_dashboard_html_save_logs_js_function(client):
    """saveLogs() JS function creates a download link with the logs cache."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "URL.createObjectURL" in resp.text
    assert "prompt-interceptor-logs-" in resp.text


async def test_dashboard_html_clear_logs_calls_reset_api(client):
    """clearLogs() JS function calls /api/reset via fetch."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "/api/reset" in resp.text


async def test_dashboard_html_has_response_column(client):
    """Dashboard table has a Response column for LLM output."""
    resp = await client.get("/")
    assert resp.status_code == 200
    assert "<th>Response</th>" in resp.text
    assert "response_body" in resp.text
