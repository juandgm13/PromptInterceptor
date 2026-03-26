"""Tests for dashboard.py endpoints."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import AsyncClient, ASGITransport

from ollama_proxy.dashboard import app
from ollama_proxy.interceptor import interceptor as _module_interceptor


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
    with patch("ollama_proxy.health.httpx.AsyncClient", return_value=mock_client):
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
    assert "PyProxy" in resp.text


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

    with patch("ollama_proxy.health.httpx.AsyncClient", return_value=mock_client):
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

    with patch("ollama_proxy.health.httpx.AsyncClient", return_value=mock_client):
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
    from ollama_proxy.interceptor import Interceptor, interceptor as _interceptor
    import ollama_proxy.dashboard as dash_mod

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
    from ollama_proxy.interceptor import Interceptor
    import ollama_proxy.dashboard as dash_mod

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
    from ollama_proxy.interceptor import Interceptor
    import ollama_proxy.dashboard as dash_mod

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
# get_app
# ---------------------------------------------------------------------------

def test_get_app():
    from ollama_proxy.dashboard import get_app
    from fastapi import FastAPI
    assert isinstance(get_app(), FastAPI)
