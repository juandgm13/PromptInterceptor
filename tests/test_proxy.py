"""Tests for proxy.py handler functions and helpers."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from prompt_interceptor.config import Config
from prompt_interceptor.logger import TrafficLogger
from prompt_interceptor.proxy import (
    _forward_headers,
    _HOP_BY_HOP,
    handle_chat_request,
    handle_generate_request,
    handle_stream_chat,
    handle_stream_generate,
)
from prompt_interceptor.rules_engine import RuleEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_req(body: dict, path: str = "/api/chat"):
    req = MagicMock()
    req.json = AsyncMock(return_value=body)
    req.url.path = path
    req.headers = {"content-type": "application/json"}
    return req


@pytest.fixture
def cfg(tmp_path):
    return Config(log_dir=str(tmp_path / "logs"), mode="passthrough", timeout=5)


@pytest.fixture
def engine_and_logger(cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return engine, logger


# ---------------------------------------------------------------------------
# _forward_headers
# ---------------------------------------------------------------------------

def test_forward_headers_strips_hop_by_hop():
    headers = {
        "content-type": "application/json",
        "transfer-encoding": "chunked",
        "content-encoding": "gzip",
        "x-custom": "keep",
    }
    result = _forward_headers(headers)
    assert "transfer-encoding" not in result
    assert "content-encoding" not in result
    assert result["content-type"] == "application/json"
    assert result["x-custom"] == "keep"


def test_forward_headers_case_insensitive():
    # All hop-by-hop names are lowercase in _HOP_BY_HOP
    assert "connection" in _HOP_BY_HOP
    result = _forward_headers({"Connection": "keep-alive", "Accept": "application/json"})
    assert "Connection" not in result
    assert result["Accept"] == "application/json"


def test_forward_headers_empty():
    assert _forward_headers({}) == {}


# ---------------------------------------------------------------------------
# handle_chat_request — success
# ---------------------------------------------------------------------------

async def test_handle_chat_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({"model": "llama3", "done": True}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req({"model": "llama3", "messages": [{"role": "user", "content": "hi"}]})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200
    assert json.loads(resp.body)["model"] == "llama3"


async def test_handle_chat_non_json_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain text")),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# handle_chat_request — error paths
# ---------------------------------------------------------------------------

async def test_handle_chat_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 408
    assert "timeout" in json.loads(resp.body)["error"].lower()


async def test_handle_chat_generic_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# handle_chat_request — intercept mode (forward / drop)
# ---------------------------------------------------------------------------

async def test_handle_chat_intercept_forward(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    response_bytes = json.dumps({"model": "llama3"}).encode()

    async def _fake_intercept(rid, method, path, headers, body):
        return "forward", body

    monkeypatch.setattr("prompt_interceptor.proxy.interceptor.intercept", _fake_intercept)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200


async def test_handle_chat_intercept_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    async def _fake_intercept(rid, method, path, headers, body):
        return "drop", body

    monkeypatch.setattr("prompt_interceptor.proxy.interceptor.intercept", _fake_intercept)

    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_chat_request(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_generate_request
# ---------------------------------------------------------------------------

async def test_handle_generate_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({"response": "hello", "done": True}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req({"model": "llama3", "prompt": "Say hello"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200


async def test_handle_generate_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("t")),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 408


async def test_handle_generate_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_generate_request(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_stream_chat
# ---------------------------------------------------------------------------

async def _mock_stream(*args, **kwargs):
    yield b'{"model":"llama3","done":false}\n'
    yield b'{"model":"llama3","done":true}\n'


async def test_handle_stream_chat_yields_chunks(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    # Consume body INSIDE the patch context: StreamingResponse is lazy
    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        req = make_req({"model": "llama3", "messages": []})
        resp = await handle_stream_chat(req, engine, logger)
        assert resp.media_type == "application/x-ndjson"
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"llama3" in body


async def test_handle_stream_chat_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_stream_generate
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_yields_chunks(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch("prompt_interceptor.proxy._stream_from_ollama", new=_mock_stream):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_stream_generate(req, engine, logger)
        assert resp.media_type == "application/x-ndjson"
        body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"llama3" in body


async def test_handle_stream_generate_drop(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("drop", None)),
    )
    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# Rule application in handlers
# ---------------------------------------------------------------------------

async def test_handle_chat_rule_switches_model(tmp_path, monkeypatch):
    """When a rule matches, the forwarded body uses the replaced model."""
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        rules=[{
            "match": {"path": "/api/chat", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek-coder"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    captured = {}

    async def _fake_fetch(target, method, path, headers, body, timeout):
        captured["body"] = json.loads(body)
        return 200, {"content-type": "application/json"}, b'{"done": true}'

    # Auto-forward in intercept mode
    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek-coder", "messages": []})),
    )

    with patch("prompt_interceptor.proxy._fetch_from_ollama", new=_fake_fetch):
        req = make_req({"model": "llama3", "messages": []})
        await handle_chat_request(req, engine, logger)

    assert captured["body"]["model"] == "deepseek-coder"
