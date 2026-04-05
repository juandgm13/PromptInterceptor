"""Tests for health.py."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prompt_interceptor.config import Config
from prompt_interceptor.health import (
    check_proxy_health,
    check_target_health,
    check_dashboard_health,
    get_status,
)


@pytest.fixture(autouse=True)
def patch_get_config(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="passthrough")
    import prompt_interceptor.health as health_mod
    monkeypatch.setattr(health_mod, "get_config", lambda: cfg)
    return cfg


# ---------------------------------------------------------------------------
# check_proxy_health
# ---------------------------------------------------------------------------

async def test_check_proxy_health_returns_200():
    resp = await check_proxy_health()
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["status"] == "healthy"
    assert "proxy_port" in data
    assert "timestamp" in data


# ---------------------------------------------------------------------------
# check_dashboard_health
# ---------------------------------------------------------------------------

async def test_check_dashboard_health_returns_200():
    resp = await check_dashboard_health()
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["status"] == "healthy"
    assert "dashboard_port" in data


# ---------------------------------------------------------------------------
# check_target_health
# ---------------------------------------------------------------------------

def _make_mock_client(status=200, json_data=None, raise_exc=None):
    """Build a mock httpx.AsyncClient context manager."""
    mock_response = MagicMock()
    mock_response.status_code = status
    mock_response.json.return_value = json_data or {"models": [{"name": "llama3"}]}
    mock_response.text = "error text"

    mock_client = AsyncMock()
    if raise_exc:
        mock_client.get = AsyncMock(side_effect=raise_exc)
    else:
        mock_client.get = AsyncMock(return_value=mock_response)

    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    return mock_client


async def test_check_target_health_success():
    mock_client = _make_mock_client(200, {"models": [{"name": "llama3"}]})
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await check_target_health()
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["target"] == "healthy"
    assert len(data["models"]) == 1


async def test_check_target_health_non_200():
    mock_client = _make_mock_client(503)
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await check_target_health()
    assert resp.status_code == 503
    data = json.loads(resp.body)
    assert data["target"] == "unhealthy"


async def test_check_target_health_timeout():
    mock_client = _make_mock_client(raise_exc=httpx.TimeoutException("t"))
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await check_target_health()
    assert resp.status_code == 504
    assert json.loads(resp.body)["target"] == "timeout"


async def test_check_target_health_connect_error():
    mock_client = _make_mock_client(raise_exc=httpx.ConnectError("refused"))
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await check_target_health()
    assert resp.status_code == 503
    assert "unhealthy" in json.loads(resp.body)["target"]


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

async def test_get_status_structure():
    mock_client = _make_mock_client(200, {"models": []})
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await get_status()
    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert "status" in data
    assert "proxy" in data
    assert "target" in data
    assert "dashboard" in data
    assert "rules" in data
    assert "logs" in data
    assert "timestamp" in data


async def test_get_status_includes_rule_count():
    mock_client = _make_mock_client(200, {"models": []})
    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await get_status()
    data = json.loads(resp.body)
    assert isinstance(data["rules"]["count"], int)
    assert isinstance(data["rules"]["enabled_count"], int)


# ---------------------------------------------------------------------------
# check_target_health — response.json() decode failure
# ---------------------------------------------------------------------------

async def test_check_target_health_json_decode_error_returns_healthy_empty_models():
    """When response.json() raises (malformed body), models defaults to [] but stays healthy."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.side_effect = Exception("invalid JSON")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("prompt_interceptor.health.httpx.AsyncClient", return_value=mock_client):
        resp = await check_target_health()

    assert resp.status_code == 200
    data = json.loads(resp.body)
    assert data["target"] == "healthy"
    assert data["models"] == []


# ---------------------------------------------------------------------------
# get_status — unexpected exception returns 503
# ---------------------------------------------------------------------------

async def test_get_status_exception_returns_503():
    """When get_status encounters an unexpected error it returns 503 instead of crashing."""
    with patch("prompt_interceptor.health.check_target_health",
               new=AsyncMock(side_effect=RuntimeError("internal failure"))):
        resp = await get_status()

    assert resp.status_code == 503
    data = json.loads(resp.body)
    assert data["status"] == "unhealthy"
    assert "internal failure" in data["error"]
