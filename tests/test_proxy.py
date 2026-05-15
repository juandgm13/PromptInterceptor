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
    _inject_num_ctx,
    _parse_stream_response,
    _parse_sse_response,
    _parse_openai_sse_response,
    _detect_context_overflow,
    _resolve_context_size,
    _CONTEXT_OVERFLOW_MSG,
    handle_chat_request,
    handle_generate_request,
    handle_stream_chat,
    handle_stream_generate,
    handle_v1_messages,
    handle_v1_chat_completions,
    handle_passthrough,
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


@pytest.fixture(autouse=True)
def mock_resolve_context_size(monkeypatch):
    """Prevent real /api/ps calls in every test; individual tests may override."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))


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
# _inject_num_ctx
# ---------------------------------------------------------------------------

def test_inject_num_ctx_adds_option(monkeypatch):
    cfg = Config(context_size=16384)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    body = {"model": "llama3", "messages": []}
    result = _inject_num_ctx(body)
    assert result["options"]["num_ctx"] == 16384
    # original dict not mutated
    assert "options" not in body


def test_inject_num_ctx_overwrites_existing(monkeypatch):
    cfg = Config(context_size=16384)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    body = {"model": "llama3", "options": {"num_ctx": 4096, "temperature": 0.7}}
    result = _inject_num_ctx(body)
    # config value always wins so the launcher selection is respected
    assert result["options"]["num_ctx"] == 16384
    # other options are preserved
    assert result["options"]["temperature"] == 0.7


def test_inject_num_ctx_none_body(monkeypatch):
    cfg = Config(context_size=16384)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    assert _inject_num_ctx(None) is None


def test_inject_num_ctx_zero_context_size(monkeypatch):
    cfg = Config(context_size=0)
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    body = {"model": "llama3"}
    result = _inject_num_ctx(body)
    assert "options" not in result


# ---------------------------------------------------------------------------
# _parse_stream_response
# ---------------------------------------------------------------------------

def test_parse_stream_response_chat():
    chunks = [
        b'{"message":{"role":"assistant","content":"Hello"},"done":false}\n',
        b'{"message":{"role":"assistant","content":" world"},"done":true,"total_duration":100}\n',
    ]
    result = _parse_stream_response(chunks)
    assert result["message"]["content"] == "Hello world"
    assert result["total_duration"] == 100


def test_parse_stream_response_generate():
    chunks = [
        b'{"response":"foo","done":false}\n',
        b'{"response":"bar","done":true}\n',
    ]
    result = _parse_stream_response(chunks)
    assert result["response"] == "foobar"


def test_parse_stream_response_empty():
    assert _parse_stream_response([]) is None


def test_parse_stream_response_invalid_json():
    chunks = [b"not json\n", b'{"response":"ok","done":true}\n']
    result = _parse_stream_response(chunks)
    assert result["response"] == "ok"


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

async def test_handle_stream_chat_yields_chunks(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    chunks = [b'{"model":"llama3","done":false}\n', b'{"model":"llama3","done":true}\n']
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(chunks))
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

    chunks = [b'{"response":"llama3","done":true}\n']
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(chunks))
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


# ---------------------------------------------------------------------------
# _fetch_from_ollama direct tests  (lines 47-57)
# ---------------------------------------------------------------------------

async def test_fetch_from_ollama_success():
    """Exercise the real body of _fetch_from_ollama with a mocked httpx client."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"content-type": "application/json"}
    mock_response.aread = AsyncMock(return_value=b'{"done": true}')

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def request(self, *args, **kwargs):
            return mock_response

    from prompt_interceptor.proxy import _fetch_from_ollama
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=_FakeClient()):
        status, headers, body = await _fetch_from_ollama(
            "http://localhost:11434", "POST", "/api/chat",
            {"content-type": "application/json"}, b'{}', 5,
        )

    assert status == 200
    assert body == b'{"done": true}'
    assert "content-type" in headers


async def test_fetch_from_ollama_non_200():
    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.headers = {}
    mock_response.aread = AsyncMock(return_value=b'Service Unavailable')

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def request(self, *args, **kwargs):
            return mock_response

    from prompt_interceptor.proxy import _fetch_from_ollama
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=_FakeClient()):
        status, _, body = await _fetch_from_ollama(
            "http://localhost:11434", "GET", "/health", {}, None, 5,
        )

    assert status == 503
    assert b"Unavailable" in body


# ---------------------------------------------------------------------------
# _stream_from_ollama direct tests  (lines 77-89)
# ---------------------------------------------------------------------------

async def test_stream_from_ollama_yields_non_empty_chunks():
    """Exercise the real body of _stream_from_ollama with mocked httpx."""
    chunks_sent = [b'{"model":"llama3"}\n', b'', b'{"done":true}\n']

    async def _fake_aiter_bytes():
        for c in chunks_sent:
            yield c

    class _FakeStreamCtx:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def aiter_bytes(self):
            return _fake_aiter_bytes()

    class _FakeClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def stream(self, *args, **kwargs):
            return _FakeStreamCtx()

    from prompt_interceptor.proxy import _stream_from_ollama
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=_FakeClient()):
        result = [c async for c in _stream_from_ollama(
            "http://localhost:11434", "POST", "/api/chat", {}, b'{}', 5,
        )]

    assert b'' not in result
    assert b'{"model":"llama3"}\n' in result
    assert b'{"done":true}\n' in result


# ---------------------------------------------------------------------------
# handle_chat_request — HTTPStatusError  (line 167)
# ---------------------------------------------------------------------------

@pytest.fixture
def passthrough_setup(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="passthrough", timeout=5)
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return cfg, engine, logger


@pytest.fixture
def intercept_generate_setup(tmp_path, monkeypatch):
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        timeout=5,
        rules=[{
            "match": {"path": "/api/generate", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return cfg, engine, logger


@pytest.fixture
def intercept_chat_setup(tmp_path, monkeypatch):
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        timeout=5,
        rules=[{
            "match": {"path": "/api/chat", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)
    return cfg, engine, logger


async def test_handle_chat_http_status_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    mock_resp = MagicMock()
    mock_resp.status_code = 422

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.HTTPStatusError(
            "unprocessable", request=MagicMock(), response=mock_resp
        )),
    ):
        resp = await handle_chat_request(make_req({"model": "llama3"}), engine, logger)

    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# handle_generate_request — modified body (line 185)
# ---------------------------------------------------------------------------

async def test_handle_generate_modified_body(intercept_generate_setup, monkeypatch):
    """Rule modifies body → body_json is replaced before forwarding."""
    cfg, engine, logger = intercept_generate_setup
    captured = {}

    async def _fake_fetch(target, method, path, headers, body, timeout):
        captured["body"] = json.loads(body)
        return 200, {"content-type": "application/json"}, b'{"done":true}'

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek", "prompt": "hi"})),
    )
    with patch("prompt_interceptor.proxy._fetch_from_ollama", new=_fake_fetch):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200
    assert captured["body"]["model"] == "deepseek"


# ---------------------------------------------------------------------------
# handle_generate_request — non-JSON response (lines 204-205)
# ---------------------------------------------------------------------------

async def test_handle_generate_non_json_response(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain text")),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# handle_generate_request — HTTPStatusError and generic Exception
# ---------------------------------------------------------------------------

async def test_handle_generate_http_status_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup
    mock_resp = MagicMock()
    mock_resp.status_code = 500

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.HTTPStatusError(
            "server error", request=MagicMock(), response=mock_resp
        )),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 500


async def test_handle_generate_generic_error(passthrough_setup):
    cfg, engine, logger = passthrough_setup

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("unexpected")),
    ):
        req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# handle_stream_chat — modified body and exception paths
# ---------------------------------------------------------------------------

async def test_handle_stream_chat_modified_body(intercept_chat_setup, monkeypatch):
    """When rule modifies body, body_json is updated before streaming."""
    cfg, engine, logger = intercept_chat_setup
    import prompt_interceptor.proxy as proxy_mod

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek", "messages": []})),
    )
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b'{"done":true}\n']))
    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"done" in body


async def test_handle_stream_chat_timeout_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.TimeoutException("timeout")))

    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    assert resp.status_code == 408
    assert "timeout" in json.loads(resp.body)["error"].lower()


async def test_handle_stream_chat_connect_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.ConnectError("connection refused")))

    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    assert resp.status_code == 502
    data = json.loads(resp.body)
    assert "connect" in data["error"].lower() or "ollama" in data["error"].lower()


async def test_handle_stream_chat_generic_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=RuntimeError("boom")))

    req = make_req({"model": "llama3", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    assert resp.status_code == 500
    assert "boom" in json.loads(resp.body)["error"]


# ---------------------------------------------------------------------------
# handle_stream_generate — modified body and exception paths
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_modified_body(tmp_path, monkeypatch):
    """Rule modifies body in stream_generate path."""
    cfg = Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        timeout=5,
        rules=[{
            "match": {"path": "/api/generate", "jsonpath": "$.model", "value": ["llama3"]},
            "replace": {"jsonpath": "$.model", "value": "deepseek"},
        }],
    )
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", {"model": "deepseek", "prompt": "hi"})),
    )
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b'{"done":true}\n']))
    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"done" in body


async def test_handle_stream_generate_timeout_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.TimeoutException("timeout")))

    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    assert resp.status_code == 408
    assert "timeout" in json.loads(resp.body)["error"].lower()


async def test_handle_stream_generate_connect_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.ConnectError("refused")))

    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    assert resp.status_code == 502
    assert "error" in json.loads(resp.body)


async def test_handle_stream_generate_generic_error(passthrough_setup, monkeypatch):
    cfg, engine, logger = passthrough_setup
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=ValueError("bad value")))

    req = make_req({"model": "llama3", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    assert resp.status_code == 500
    assert "bad value" in json.loads(resp.body)["error"]


# ---------------------------------------------------------------------------
# Invalid JSON body — all handlers must return 400
# ---------------------------------------------------------------------------

def _make_bad_json_req(path: str = "/api/chat"):
    """Request whose .json() raises an exception (malformed body)."""
    req = MagicMock()
    req.json = AsyncMock(side_effect=Exception("JSON decode error"))
    req.url.path = path
    req.headers = {"content-type": "application/json"}
    return req


async def test_handle_chat_request_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_chat_request returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_chat_request(_make_bad_json_req("/api/chat"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_generate_request_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_generate_request returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_generate_request(_make_bad_json_req("/api/generate"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_stream_chat_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_stream_chat returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_stream_chat(_make_bad_json_req("/api/chat"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_stream_generate_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    """handle_stream_generate returns 400 when the request body is not valid JSON."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_stream_generate(_make_bad_json_req("/api/generate"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


# ---------------------------------------------------------------------------
# handle_passthrough
# ---------------------------------------------------------------------------

def _make_passthrough_req(body: dict | bytes | None, path: str = "/v1/messages",
                           method: str = "POST", query: str = ""):
    req = MagicMock()
    raw = json.dumps(body).encode() if isinstance(body, dict) else (body or b"")
    req.body = AsyncMock(return_value=raw)
    req.method = method
    req.url.path = path
    req.query_params = query
    req.headers = {"content-type": "application/json"}
    return req


async def test_passthrough_non_streaming_forwards_and_returns(cfg, monkeypatch):
    """Non-streaming passthrough returns Ollama's response body and status."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'{"ok": true}'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "llama3"}, "/v1/messages"))

    assert resp.status_code == 200
    assert b'"ok"' in resp.body


async def test_passthrough_streaming_returns_streaming_response(cfg, monkeypatch):
    """Streaming passthrough returns a StreamingResponse with SSE media type for /v1/ paths."""
    from fastapi.responses import StreamingResponse
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b"data: {}\n"]))

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req)

    assert isinstance(resp, StreamingResponse)
    assert "text/event-stream" in resp.media_type


async def test_passthrough_streaming_ndjson_for_api_path(cfg, monkeypatch):
    """Streaming passthrough uses application/x-ndjson for /api/ paths."""
    from fastapi.responses import StreamingResponse
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b'{"done":true}\n']))

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/api/generate")
    resp = await handle_passthrough(req)

    assert isinstance(resp, StreamingResponse)
    assert "ndjson" in resp.media_type


async def test_passthrough_timeout_returns_408(cfg, monkeypatch):
    """Passthrough returns 408 when Ollama times out."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timeout"))

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "llama3"}, "/v1/messages"))

    assert resp.status_code == 408


async def test_passthrough_connect_error_returns_502(cfg, monkeypatch):
    """Passthrough returns 502 when Ollama is unreachable."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req(None, "/api/tags", "GET"))

    assert resp.status_code == 502


async def test_passthrough_non_json_body_is_handled(cfg, monkeypatch):
    """Passthrough handles binary/non-JSON bodies without crashing."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b"pong"
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "text/plain"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req(b"not json", "/api/version", "GET"))

    assert resp.status_code == 200


async def test_passthrough_query_params_appended_to_url(cfg, monkeypatch):
    """Query params are appended to the forwarded URL."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    captured_urls = []

    fake_resp = MagicMock()
    fake_resp.content = b"{}"
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    async def capture_request(method, url, **kwargs):
        captured_urls.append(url)
        return fake_resp

    mock_client.request = capture_request

    req = _make_passthrough_req(None, "/api/tags", "GET", query="name=llama3")
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)

    assert resp.status_code == 200
    assert "name=llama3" in captured_urls[0]


async def test_passthrough_generic_exception_returns_500(cfg, monkeypatch):
    """Passthrough returns 500 on unexpected errors."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(side_effect=RuntimeError("unexpected"))

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "llama3"}, "/v1/messages"))

    assert resp.status_code == 500


async def test_passthrough_streaming_chunks_yielded(cfg, monkeypatch):
    """Streaming passthrough yields chunks from Ollama."""
    from fastapi.responses import StreamingResponse
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    chunks = [b'data: {"text": "hello"}', b'data: {"text": " world"}']
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(chunks))

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req)
    assert isinstance(resp, StreamingResponse)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"hello" in collected


async def test_passthrough_streaming_connect_error_returns_502(cfg, monkeypatch):
    """ConnectError from _start_streaming_request returns 502 JSONResponse."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.ConnectError("refused")))

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req)
    assert resp.status_code == 502
    assert "error" in json.loads(resp.body)


async def test_passthrough_streaming_generic_exception_returns_500(cfg, monkeypatch):
    """Generic exception from _start_streaming_request returns 500 JSONResponse."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=RuntimeError("boom")))

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req)
    assert resp.status_code == 500
    assert "boom" in json.loads(resp.body)["error"]


async def test_passthrough_streaming_logs_response_with_logger(cfg, monkeypatch, tmp_path):
    """Streaming passthrough calls logger.log_response after stream completes."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b'{"done":true}']))

    logger = TrafficLogger(cfg)
    logged = []
    original_log_response = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original_log_response(*a, **kw)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req, logger=logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) == 1


async def test_passthrough_body_json_attribute_error_handled(cfg, monkeypatch):
    """When body_json.get raises AttributeError, is_stream defaults to False."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'{"ok": true}'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.json = MagicMock(return_value={"ok": True})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    # Craft a request whose body is a non-JSON string so body_json ends up None
    # then override body_json inside handle_passthrough via a body that json.loads returns a string
    req = _make_passthrough_req(b'"just a string"', "/api/test", "POST")

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(req)

    assert resp.status_code == 200


async def test_passthrough_resp_json_generic_exception_handled(cfg, monkeypatch):
    """When resp.json() raises a non-JSONDecodeError exception, resp_json is None."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'data'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/octet-stream"}
    fake_resp.json = MagicMock(side_effect=RuntimeError("not json at all"))

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    logger = TrafficLogger(cfg)
    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(_make_passthrough_req({"model": "x"}, "/api/test"), logger=logger)

    assert resp.status_code == 200


async def test_passthrough_non_dict_json_response_handled(cfg, monkeypatch):
    """Non-dict resp.json() is treated as None when logging."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    fake_resp = MagicMock()
    fake_resp.content = b'"just a string"'
    fake_resp.status_code = 200
    fake_resp.headers = {"content-type": "application/json"}
    fake_resp.json.return_value = "just a string"  # str, not dict/list

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.request = AsyncMock(return_value=fake_resp)

    logger = TrafficLogger(cfg)

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        resp = await handle_passthrough(
            _make_passthrough_req({"model": "llama3"}, "/v1/test"),
            logger=logger,
        )

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _parse_sse_response
# ---------------------------------------------------------------------------

def test_parse_sse_response_assembles_content():
    chunks = [
        b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_01","model":"qwen3:9b","role":"assistant","content":[]}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
        b'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}\n\n',
        b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    ]
    result = _parse_sse_response(chunks)
    assert result is not None
    assert result["content"] == [{"type": "text", "text": "Hello world"}]


def test_parse_sse_response_preserves_message_start_metadata():
    chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_01","model":"qwen3:9b","role":"assistant"}}\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["id"] == "msg_01"
    assert result["model"] == "qwen3:9b"


def test_parse_sse_response_empty_chunks_returns_none():
    assert _parse_sse_response([]) is None


def test_parse_sse_response_ping_only_returns_none():
    chunks = [b'event: ping\ndata: {"type":"ping"}\n\n']
    assert _parse_sse_response(chunks) is None


def test_parse_sse_response_done_marker_skipped():
    """data: [DONE] line is skipped without crashing."""
    chunks = [
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["content"][0]["text"] == "hi"


def test_parse_sse_response_skips_invalid_json():
    chunks = [
        b'data: not-valid-json\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["content"] == [{"type": "text", "text": "ok"}]


def test_parse_sse_response_accepts_string_chunks():
    chunks = [
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result["content"][0]["text"] == "x"


def test_parse_sse_response_captures_thinking_delta():
    """thinking_delta blocks (Claude Code extended thinking) are captured as thinking content block."""
    chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_01","model":"claude","role":"assistant","content":[]}}\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"reason A"}}\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":" reason B"}}\n',
        b'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert result is not None
    thinking_blocks = [b for b in result["content"] if b["type"] == "thinking"]
    text_blocks = [b for b in result["content"] if b["type"] == "text"]
    assert thinking_blocks[0]["thinking"] == "reason A reason B"
    assert text_blocks[0]["text"] == "Answer"


def test_parse_sse_response_no_thinking_has_no_thinking_block():
    """When no thinking_delta present, content has only text blocks."""
    chunks = [
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n',
    ]
    result = _parse_sse_response(chunks)
    assert all(b["type"] != "thinking" for b in result.get("content", []))


# ---------------------------------------------------------------------------
# _parse_openai_sse_response
# ---------------------------------------------------------------------------

def test_parse_openai_sse_response_assembles_content():
    chunks = [
        b'data: {"id":"cmp-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n',
        b'data: {"id":"cmp-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "Hello world"


def test_parse_openai_sse_response_empty_returns_none():
    assert _parse_openai_sse_response([]) is None


def test_parse_openai_sse_response_done_only_returns_none():
    chunks = [b'data: [DONE]\n']
    assert _parse_openai_sse_response(chunks) is None


def test_parse_openai_sse_response_skips_invalid_json():
    chunks = [
        b'data: bad json\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"y"},"finish_reason":null}]}\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result["choices"][0]["message"]["content"] == "y"


def test_parse_openai_sse_response_ollama_usage_chunk_last():
    """Ollama sends a final chunk with choices:[] and usage stats. Content must still be captured."""
    chunks = [
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":"stop"}]}\n',
        b'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["usage"]["completion_tokens"] == 2


def test_parse_openai_sse_response_captures_thinking():
    """delta.thinking (Ollama 0.7+ via /v1/chat/completions) is captured separately."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"thinking":"step 1"},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"thinking":" step 2","content":""},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"The answer"},"finish_reason":"stop"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["thinking"] == "step 1 step 2"
    assert result["choices"][0]["message"]["content"] == "The answer"


def test_parse_openai_sse_response_captures_reasoning_field():
    """delta.reasoning (Open Code / qwen3 via Ollama) is captured as thinking."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"","reasoning":"think A"},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"","reasoning":" think B"},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Result"},"finish_reason":"stop"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["thinking"] == "think A think B"
    assert result["choices"][0]["message"]["content"] == "Result"


def test_parse_openai_sse_response_no_thinking_field_omitted():
    """When no thinking in stream, message has no thinking key."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert "thinking" not in result["choices"][0]["message"]


def test_parse_stream_response_chat_captures_thinking():
    """message.thinking (Ollama 0.7+ via /api/chat) is captured alongside content."""
    chunks = [
        b'{"message":{"role":"assistant","content":"","thinking":"reason 1"},"done":false}\n',
        b'{"message":{"role":"assistant","content":"","thinking":" reason 2"},"done":false}\n',
        b'{"message":{"role":"assistant","content":"Answer","thinking":""},"done":true}\n',
    ]
    result = _parse_stream_response(chunks)
    assert result["message"]["content"] == "Answer"
    assert result["message"]["thinking"] == "reason 1 reason 2"


def test_parse_stream_response_chat_no_thinking_field_omitted():
    """When no thinking in /api/chat stream, message has no thinking key."""
    chunks = [
        b'{"message":{"role":"assistant","content":"Hi"},"done":true}\n',
    ]
    result = _parse_stream_response(chunks)
    assert "thinking" not in result["message"]


def test_parse_openai_sse_response_captures_tool_calls():
    """tool_calls deltas (OpenAI streaming) are accumulated into message.tool_calls."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"city\\":"}}]},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"Paris\\"}"}}]},"finish_reason":"tool_calls"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert tool_calls[0]["id"] == "call_abc"
    assert '"city"' in tool_calls[0]["function"]["arguments"]


def test_parse_openai_sse_response_multiple_tool_calls():
    """Multiple tool_call indexes are each accumulated separately."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"id0","type":"function","function":{"name":"tool_a","arguments":""}}]},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"id1","type":"function","function":{"name":"tool_b","arguments":""}}]},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"x\\":1}"}}]},"finish_reason":null}]}\n',
        b'data: {"id":"x","choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\\"y\\":2}"}}]},"finish_reason":"tool_calls"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    tool_calls = result["choices"][0]["message"]["tool_calls"]
    assert len(tool_calls) == 2
    assert tool_calls[0]["function"]["name"] == "tool_a"
    assert tool_calls[1]["function"]["name"] == "tool_b"


def test_parse_openai_sse_response_no_tool_calls_field_omitted():
    """When no tool_calls in stream, message has no tool_calls key."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":"stop"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert "tool_calls" not in result["choices"][0]["message"]


def test_parse_openai_sse_response_tool_calls_only_no_content():
    """Response with only tool_calls (no text content) is still returned, not None."""
    chunks = [
        b'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"search","arguments":"{\\"q\\":\\"test\\"}"}}]},"finish_reason":"tool_calls"}]}\n',
        b'data: [DONE]\n',
    ]
    result = _parse_openai_sse_response(chunks)
    assert result is not None
    assert result["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "search"
    assert result["choices"][0]["message"]["content"] == ""


def test_parse_stream_response_chat_captures_tool_calls():
    """message.tool_calls (Ollama native /api/chat) is captured from the final chunk."""
    tool_calls = [{"function": {"name": "get_time", "arguments": {}}}]
    import json
    final_chunk = json.dumps({
        "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
        "done": True,
    }).encode()
    result = _parse_stream_response([final_chunk])
    assert result is not None
    assert result["message"]["tool_calls"] == tool_calls


def test_parse_stream_response_chat_no_tool_calls_field_omitted():
    """When no tool_calls in /api/chat stream, message has no tool_calls key."""
    chunks = [
        b'{"message":{"role":"assistant","content":"Hi"},"done":true}\n',
    ]
    result = _parse_stream_response(chunks)
    assert "tool_calls" not in result["message"]


def test_parse_stream_response_tool_calls_only_not_none():
    """Response with tool_calls but empty content is returned, not None."""
    import json
    chunk = json.dumps({
        "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "fn", "arguments": {}}}]},
        "done": True,
    }).encode()
    result = _parse_stream_response([chunk])
    assert result is not None
    assert len(result["message"]["tool_calls"]) == 1


# ---------------------------------------------------------------------------
# Helpers for v1 handler tests
# ---------------------------------------------------------------------------

def _make_v1_streaming_client(chunks: list, status_code: int = 200):
    """Build a mock httpx.AsyncClient that streams the given byte chunks."""
    async def fake_aiter_bytes():
        for c in chunks:
            yield c

    fake_stream_resp = MagicMock()
    fake_stream_resp.status_code = status_code
    fake_stream_resp.aiter_bytes = fake_aiter_bytes
    fake_stream_resp.__aenter__ = AsyncMock(return_value=fake_stream_resp)
    fake_stream_resp.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=fake_stream_resp)
    return mock_client


def _make_start_streaming_mock(chunks: list, status_code: int = 200):
    """Mock for _start_streaming_request returning (http_client, resp) with given chunks."""
    async def fake_aiter_bytes():
        for c in chunks:
            yield c

    fake_resp = MagicMock()
    fake_resp.status_code = status_code
    fake_resp.aiter_bytes = fake_aiter_bytes
    fake_resp.aclose = AsyncMock()
    fake_resp.aread = AsyncMock(return_value=b"".join(chunks))

    fake_client = MagicMock()
    fake_client.aclose = AsyncMock()

    return AsyncMock(return_value=(fake_client, fake_resp))


# ---------------------------------------------------------------------------
# handle_v1_messages — non-streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_non_streaming_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({
        "id": "msg_01", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "Hi"}],
        "model": "qwen3:9b", "stop_reason": "end_turn",
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req(
            {"model": "qwen3:9b", "max_tokens": 100, "messages": [{"role": "user", "content": "Hi"}]},
            path="/v1/messages",
        )
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200
    assert json.loads(resp.body)["id"] == "msg_01"


async def test_handle_v1_messages_non_streaming_non_json_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200


async def test_handle_v1_messages_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 408


async def test_handle_v1_messages_connect_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 502


async def test_handle_v1_messages_generic_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 500


async def test_handle_v1_messages_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_v1_messages(_make_bad_json_req("/v1/messages"), engine, logger)
    assert resp.status_code == 400
    assert "invalid json" in json.loads(resp.body)["error"].lower()


async def test_handle_v1_messages_intercept_drop(tmp_path, monkeypatch):
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
    req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    assert resp.status_code == 204


async def test_handle_v1_messages_intercept_forward(tmp_path, monkeypatch):
    cfg = Config(log_dir=str(tmp_path / "logs"), mode="intercept")
    import prompt_interceptor.rules_engine as re_mod, prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    logger = TrafficLogger(cfg)
    engine = RuleEngine(logger)

    body = {"model": "qwen3:9b", "messages": []}
    monkeypatch.setattr(
        "prompt_interceptor.proxy.interceptor.intercept",
        AsyncMock(return_value=("forward", body)),
    )
    response_bytes = json.dumps({"id": "msg_01", "type": "message"}).encode()
    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req(body, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# handle_v1_messages — streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_streaming_returns_event_stream(cfg, engine_and_logger, monkeypatch):
    from fastapi.responses import StreamingResponse
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(sse_chunks))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    assert isinstance(resp, StreamingResponse)
    assert "text/event-stream" in resp.media_type
    body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"Hi" in body


async def test_handle_v1_messages_streaming_logs_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_01","model":"qwen3:9b","role":"assistant"}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(sse_chunks))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    logged_body = logged[-1][3]
    assert logged_body is not None
    assert logged_body["content"][0]["text"] == "Hello"


async def test_handle_v1_messages_streaming_connect_error_returns_502(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.ConnectError("refused")))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    assert resp.status_code == 502
    assert "error" in json.loads(resp.body)


async def test_handle_v1_messages_streaming_timeout_returns_408(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.TimeoutException("timeout")))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    assert resp.status_code == 408
    assert "timeout" in json.loads(resp.body)["error"].lower()


# ---------------------------------------------------------------------------
# handle_v1_chat_completions — non-streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_non_streaming_success(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_bytes = json.dumps({
        "id": "chatcmpl-1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
        "model": "qwen3:9b",
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, response_bytes)),
    ):
        req = make_req(
            {"model": "qwen3:9b", "messages": [{"role": "user", "content": "Hi"}]},
            path="/v1/chat/completions",
        )
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 200
    assert json.loads(resp.body)["id"] == "chatcmpl-1"


async def test_handle_v1_chat_completions_timeout(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.TimeoutException("timeout")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 408


async def test_handle_v1_chat_completions_connect_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=httpx.ConnectError("refused")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 502


async def test_handle_v1_chat_completions_generic_error(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 500


async def test_handle_v1_chat_completions_invalid_json_returns_400(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    resp = await handle_v1_chat_completions(_make_bad_json_req("/v1/chat/completions"), engine, logger)
    assert resp.status_code == 400


async def test_handle_v1_chat_completions_intercept_drop(tmp_path, monkeypatch):
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
    req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    assert resp.status_code == 204


# ---------------------------------------------------------------------------
# handle_v1_chat_completions — streaming
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_streaming_returns_event_stream(cfg, engine_and_logger, monkeypatch):
    from fastapi.responses import StreamingResponse
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"id":"cmp-1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n',
        b'data: [DONE]\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(sse_chunks))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    assert isinstance(resp, StreamingResponse)
    assert "text/event-stream" in resp.media_type
    body = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"Hi" in body


async def test_handle_v1_chat_completions_streaming_logs_response(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"id":"cmp-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n',
        b'data: [DONE]\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(sse_chunks))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    logged_body = logged[-1][3]
    assert logged_body is not None
    assert logged_body["choices"][0]["message"]["content"] == "Hello"


async def test_handle_v1_messages_streaming_generic_error_returns_500(cfg, engine_and_logger, monkeypatch):
    """Generic exception from _start_streaming_request returns 500."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=RuntimeError("unexpected")))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    assert resp.status_code == 500
    assert "unexpected" in json.loads(resp.body)["error"]


async def test_handle_v1_chat_completions_streaming_timeout_returns_408(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.TimeoutException("timeout")))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    assert resp.status_code == 408
    assert "timeout" in json.loads(resp.body)["error"].lower()


async def test_handle_v1_chat_completions_streaming_connect_error_returns_502(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=httpx.ConnectError("refused")))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    assert resp.status_code == 502
    assert "error" in json.loads(resp.body)


async def test_handle_v1_chat_completions_streaming_generic_error_returns_500(cfg, engine_and_logger, monkeypatch):
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", AsyncMock(side_effect=RuntimeError("boom")))

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    assert resp.status_code == 500
    assert "boom" in json.loads(resp.body)["error"]


async def test_handle_v1_chat_completions_non_streaming_non_json_response(cfg, engine_and_logger, monkeypatch):
    """handle_v1_chat_completions handles non-JSON response body without crashing."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "text/plain"}, b"plain text")),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# JSON fallback when SSE/NDJSON parsing returns None (non-SSE Ollama responses)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_v1_messages_streaming_json_fallback(cfg, engine_and_logger, monkeypatch):
    """Non-2xx response body is read and logged by _handle_error_stream."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    error_body = b'{"error": "model not found"}'
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([error_body], status_code=404))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)

    assert len(logged) >= 1
    status_logged, logged_body = logged[-1][1], logged[-1][3]
    assert status_logged == 404
    assert logged_body == {"error": "model not found"}


@pytest.mark.asyncio
async def test_handle_v1_chat_completions_streaming_json_fallback(cfg, engine_and_logger, monkeypatch):
    """Non-2xx response body is read and logged by _handle_error_stream."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    error_body = b'{"error": "context length exceeded"}'
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([error_body], status_code=400))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)

    assert len(logged) >= 1
    status_logged, logged_body = logged[-1][1], logged[-1][3]
    assert status_logged == 400
    assert logged_body == {"error": "context length exceeded"}


@pytest.mark.asyncio
async def test_handle_stream_chat_json_fallback(cfg, engine_and_logger, monkeypatch):
    """Ollama error chunk (status 200) is parsed by _parse_stream_response and logged."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    error_body = b'{"error": "out of memory"}'
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([error_body]))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/api/chat")
    resp = await handle_stream_chat(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    logged_body = logged[-1][3]
    assert logged_body == {"error": "out of memory"}


@pytest.mark.asyncio
async def test_handle_stream_generate_json_fallback(cfg, engine_and_logger, monkeypatch):
    """Ollama error chunk (status 200) is parsed by _parse_stream_response and logged."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    error_body = b'{"error": "model load failed"}'
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([error_body]))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "prompt": "hello", "stream": True}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    logged_body = logged[-1][3]
    assert logged_body == {"error": "model load failed"}


@pytest.mark.asyncio
async def test_handle_v1_messages_streaming_json_fallback_invalid(cfg, engine_and_logger, monkeypatch):
    """Non-JSON non-2xx body is wrapped as {'error': raw_text} by _handle_error_stream."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b"not valid json"], status_code=500))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)

    assert len(logged) >= 1
    assert logged[-1][1] == 500
    assert logged[-1][3] == {"error": "not valid json"}


@pytest.mark.asyncio
async def test_handle_v1_chat_completions_streaming_json_fallback_invalid(cfg, engine_and_logger, monkeypatch):
    """Non-JSON non-2xx body is wrapped as {'error': raw_text} by _handle_error_stream."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b"not valid json"], status_code=500))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)

    assert len(logged) >= 1
    assert logged[-1][1] == 500
    assert logged[-1][3] == {"error": "not valid json"}


@pytest.mark.asyncio
async def test_handle_stream_chat_json_fallback_invalid(cfg, engine_and_logger, monkeypatch):
    """Unparseable status-200 stream is logged as None (no NDJSON or JSON fallback)."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b"not valid json"]))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/api/chat")
    resp = await handle_stream_chat(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    assert logged[-1][3] is None


@pytest.mark.asyncio
async def test_handle_stream_generate_json_fallback_invalid(cfg, engine_and_logger, monkeypatch):
    """Unparseable status-200 stream is logged as None (no NDJSON or JSON fallback)."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([b"not valid json"]))

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "prompt": "hello", "stream": True}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    assert logged[-1][3] is None


@pytest.mark.asyncio
async def test_passthrough_streaming_ndjson_path(cfg, monkeypatch, tmp_path):
    """Streaming passthrough uses _parse_stream_response for non-/v1/ NDJSON paths."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    ndjson_chunk = b'{"message":{"content":"hi"},"done":false}\n{"done":true}\n'
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([ndjson_chunk]))

    logger = TrafficLogger(cfg)
    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = _make_passthrough_req({"model": "qwen3:9b", "messages": [], "stream": True}, "/api/chat")
    resp = await handle_passthrough(req, logger=logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) == 1
    assert logged[0][1] == 200
    assert logged[0][3] is not None


@pytest.mark.asyncio
async def test_passthrough_streaming_non2xx_logged_with_error(cfg, monkeypatch, tmp_path):
    """Non-2xx streaming response is handled by _handle_error_stream and logged."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    garbage_chunk = b"not json at all"
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([garbage_chunk], status_code=500))

    logger = TrafficLogger(cfg)
    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = _make_passthrough_req({"model": "qwen3:9b", "messages": [], "stream": True}, "/api/chat")
    resp = await handle_passthrough(req, logger=logger)

    assert len(logged) == 1
    assert logged[0][1] == 500
    assert logged[0][3] == {"error": "not json at all"}


# ---------------------------------------------------------------------------
# Helper: streaming mock that raises mid-stream
# ---------------------------------------------------------------------------

def _make_failing_streaming_mock(error: Exception, chunks_before_error: list = None):
    """Mock where _start_streaming_request returns a resp whose aiter_bytes raises."""
    chunks_before_error = chunks_before_error or []

    async def fake_aiter_bytes():
        for c in chunks_before_error:
            yield c
        raise error

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.aiter_bytes = fake_aiter_bytes
    fake_resp.aclose = AsyncMock()

    fake_client = MagicMock()
    fake_client.aclose = AsyncMock()

    return AsyncMock(return_value=(fake_client, fake_resp))


# ---------------------------------------------------------------------------
# _start_streaming_request — exception closes client (lines 270-277)
# ---------------------------------------------------------------------------

async def test_start_streaming_request_returns_client_and_response():
    """Successful _start_streaming_request returns (http_client, resp) tuple."""
    from prompt_interceptor.proxy import _start_streaming_request

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(return_value=mock_resp)
    mock_client.aclose = AsyncMock()

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        client, resp = await _start_streaming_request("POST", "http://localhost", {}, b"{}", 5)

    assert client is mock_client
    assert resp is mock_resp


async def test_start_streaming_request_exception_closes_client():
    """When http_client.send raises, client is closed and exception re-raised."""
    from prompt_interceptor.proxy import _start_streaming_request

    mock_client = MagicMock()
    mock_client.build_request = MagicMock(return_value=MagicMock())
    mock_client.send = AsyncMock(side_effect=RuntimeError("send failed"))
    mock_client.aclose = AsyncMock()

    with patch("prompt_interceptor.proxy.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(RuntimeError, match="send failed"):
            await _start_streaming_request("POST", "http://localhost", {}, b"{}", 5)

    mock_client.aclose.assert_called_once()


# ---------------------------------------------------------------------------
# _handle_error_stream — aread() raises (lines 290-291)
# ---------------------------------------------------------------------------

async def test_handle_error_stream_aread_raises():
    """When resp.aread() raises, body defaults to b'' and a JSONResponse is returned."""
    from prompt_interceptor.proxy import _handle_error_stream

    resp = MagicMock()
    resp.aread = AsyncMock(side_effect=IOError("read error"))
    resp.aclose = AsyncMock()

    http_client = MagicMock()
    http_client.aclose = AsyncMock()

    result = await _handle_error_stream(resp, http_client, 502, None, None)
    assert result.status_code == 502


# ---------------------------------------------------------------------------
# handle_passthrough streaming — ConnectError WITH logger (line 342)
# ---------------------------------------------------------------------------

async def test_passthrough_streaming_connect_error_logs_with_logger(cfg, monkeypatch):
    """ConnectError from _start_streaming_request is logged when logger is provided."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        AsyncMock(side_effect=httpx.ConnectError("refused"))
    )

    logger = TrafficLogger(cfg)
    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req, logger=logger)

    assert resp.status_code == 502
    assert len(logged) == 1
    assert logged[0][1] == 502


# ---------------------------------------------------------------------------
# handle_passthrough streaming — generic error WITH logger (line 346)
# ---------------------------------------------------------------------------

async def test_passthrough_streaming_generic_error_logs_with_logger(cfg, monkeypatch):
    """Generic error from _start_streaming_request is logged when logger is provided."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        AsyncMock(side_effect=RuntimeError("oops"))
    )

    logger = TrafficLogger(cfg)
    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req, logger=logger)

    assert resp.status_code == 500
    assert len(logged) == 1
    assert logged[0][1] == 500


# ---------------------------------------------------------------------------
# handle_passthrough streaming — mid-stream error yields error chunk (lines 364-365)
# ---------------------------------------------------------------------------

async def test_passthrough_streaming_midstream_error_yields_error_chunk(cfg, monkeypatch):
    """When aiter_bytes raises mid-stream in _stream_gen, an error JSON chunk is yielded."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    async def _failing_aiter_bytes():
        yield b"data: first\n"
        raise IOError("mid-stream error")

    fake_resp = MagicMock()
    fake_resp.status_code = 200
    fake_resp.aiter_bytes = _failing_aiter_bytes
    fake_resp.aclose = AsyncMock()

    fake_client = MagicMock()
    fake_client.aclose = AsyncMock()

    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        AsyncMock(return_value=(fake_client, fake_resp))
    )

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/v1/messages")
    resp = await handle_passthrough(req)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"error" in collected


# ---------------------------------------------------------------------------
# handle_passthrough streaming — JSON fallback fails with logger (lines 374-375)
# ---------------------------------------------------------------------------

async def test_passthrough_streaming_json_fallback_invalid_with_logger(cfg, monkeypatch):
    """Garbage bytes that fail both NDJSON and JSON fallback are logged as None."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_start_streaming_mock([b"pure garbage text not json"])
    )

    logger = TrafficLogger(cfg)
    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = _make_passthrough_req({"model": "llama3", "stream": True}, "/api/chat")
    resp = await handle_passthrough(req, logger=logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) == 1
    assert logged[0][3] is None


# ---------------------------------------------------------------------------
# handle_v1_messages streaming — aiter_bytes raises (lines 464-465, 471-473)
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_streaming_aiter_bytes_raises(cfg, engine_and_logger, monkeypatch):
    """When aiter_bytes raises in generate(), error chunk is yielded and logged as 500."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_failing_streaming_mock(IOError("stream broken"))
    )

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected)
    assert data["error"] == "stream broken"
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["args"][1] == 500


# ---------------------------------------------------------------------------
# handle_v1_messages streaming — unparseable 200 body (lines 477-480)
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_streaming_unparseable_200_body(cfg, engine_and_logger, monkeypatch):
    """Status-200 body that fails SSE parsing and JSON fallback is logged as None."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_start_streaming_mock([b"raw garbage text not sse or json"])
    )

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    assert logged[-1][3] is None


# ---------------------------------------------------------------------------
# handle_v1_messages streaming — was_corrected=True (lines 486-487)
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """SSE with think-tags triggers normalization; emit_anthropic_sse is yielded."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"type":"message_start","message":{"id":"msg_01","role":"assistant","content":[]}}\n\n',
        b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"<think>deep reasoning</think>The answer"}}\n\n',
        b'data: {"type":"message_stop"}\n\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(sse_chunks))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/messages")
    resp = await handle_v1_messages(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"The answer" in collected
    assert b"deep reasoning" in collected
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_v1_messages non-streaming — was_corrected=True (lines 507-508)
# ---------------------------------------------------------------------------

async def test_handle_v1_messages_non_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """Non-streaming Anthropic response with think-tags is corrected and logged."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_data = {
        "id": "msg_01", "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "<think>my reasoning</think>The answer"}],
        "model": "qwen3:9b",
    }

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, json.dumps(response_data).encode())),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/messages")
        resp = await handle_v1_messages(req, engine, logger)

    assert resp.status_code == 200
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_v1_chat_completions streaming — aiter_bytes raises (lines 581-582, 588-590)
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_streaming_aiter_bytes_raises(cfg, engine_and_logger, monkeypatch):
    """When aiter_bytes raises in generate(), error chunk is yielded and logged as 500."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_failing_streaming_mock(IOError("network error"))
    )

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected)
    assert data["error"] == "network error"
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["args"][1] == 500


# ---------------------------------------------------------------------------
# handle_v1_chat_completions streaming — unparseable 200 body (lines 594-597)
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_streaming_unparseable_200_body(cfg, engine_and_logger, monkeypatch):
    """Status-200 body that fails SSE parsing and JSON fallback is logged as None."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_start_streaming_mock([b"raw garbage not sse or json"])
    )

    logged = []
    original = logger.log_response
    logger.log_response = lambda *a, **kw: logged.append(a) or original(*a, **kw)

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    b"".join([chunk async for chunk in resp.body_iterator])

    assert len(logged) >= 1
    assert logged[-1][3] is None


# ---------------------------------------------------------------------------
# handle_v1_chat_completions streaming — was_corrected=True (lines 603-604)
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """OpenAI SSE with think-tags triggers normalization; emit_openai_sse is yielded."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    sse_chunks = [
        b'data: {"id":"cmp-1","choices":[{"index":0,"delta":{"role":"assistant","content":"<think>reasoning</think>The answer"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(sse_chunks))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    assert b"The answer" in collected
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_v1_chat_completions non-streaming — was_corrected=True (lines 624-625)
# ---------------------------------------------------------------------------

async def test_handle_v1_chat_completions_non_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """Non-streaming OpenAI response with think-tags is corrected and logged."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_data = {
        "id": "chatcmpl-1", "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "<think>my thinking</think>The answer"}, "finish_reason": "stop"}],
        "model": "qwen3:9b",
    }

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, json.dumps(response_data).encode())),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 200
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_chat_request non-streaming — was_corrected=True (lines 715-716)
# ---------------------------------------------------------------------------

async def test_handle_chat_request_non_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """handle_chat_request corrects think-tags in message.content and logs correction_applied."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_data = {
        "model": "qwen3:9b",
        "message": {"role": "assistant", "content": "<think>chain of thought</think>The final answer"},
        "done": True,
    }

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, json.dumps(response_data).encode())),
    ):
        req = make_req({"model": "qwen3:9b", "messages": [{"role": "user", "content": "hi"}]})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 200
    body = json.loads(resp.body)
    assert "<think>" not in body["message"]["content"]
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_generate_request non-streaming — was_corrected=True (lines 780-781)
# ---------------------------------------------------------------------------

async def test_handle_generate_request_non_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """handle_generate_request corrects think-tags in message.content and logs correction_applied."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    response_data = {
        "model": "qwen3:9b",
        "message": {"role": "assistant", "content": "<think>step by step</think>Final answer"},
        "done": True,
    }

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, json.dumps(response_data).encode())),
    ):
        req = make_req({"model": "qwen3:9b", "prompt": "calculate 2+2"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 200
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_stream_chat — non-2xx response (line 848)
# ---------------------------------------------------------------------------

async def test_handle_stream_chat_non_2xx_returns_error_response(cfg, engine_and_logger, monkeypatch):
    """Non-2xx response from Ollama in handle_stream_chat is handled by _handle_error_stream."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_start_streaming_mock([b'{"error":"model not found"}'], status_code=404)
    )

    req = make_req({"model": "nonexistent", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# handle_stream_chat streaming — aiter_bytes raises (lines 857-858, 864-866)
# ---------------------------------------------------------------------------

async def test_handle_stream_chat_aiter_bytes_raises(cfg, engine_and_logger, monkeypatch):
    """When aiter_bytes raises in generate(), error chunk is yielded and logged as 500."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_failing_streaming_mock(IOError("connection dropped"))
    )

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected)
    assert data["error"] == "connection dropped"
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["args"][1] == 500


# ---------------------------------------------------------------------------
# handle_stream_chat streaming — was_corrected=True (lines 879-880)
# ---------------------------------------------------------------------------

async def test_handle_stream_chat_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """NDJSON response with think-tags triggers normalization; corrected chunk is yielded."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    ndjson_chunks = [
        b'{"message":{"role":"assistant","content":"<think>deep thought</think>The answer"},"done":true}\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(ndjson_chunks))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    body = json.loads(collected)
    assert "<think>" not in body["message"]["content"]
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# handle_stream_generate — non-2xx response (line 935)
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_non_2xx_returns_error_response(cfg, engine_and_logger, monkeypatch):
    """Non-2xx response from Ollama in handle_stream_generate is handled by _handle_error_stream."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_start_streaming_mock([b'{"error":"model not found"}'], status_code=404)
    )

    req = make_req({"model": "nonexistent", "prompt": "hello"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)

    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# handle_stream_generate streaming — aiter_bytes raises (lines 944-945, 951-953)
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_aiter_bytes_raises(cfg, engine_and_logger, monkeypatch):
    """When aiter_bytes raises in generate(), error chunk is yielded and logged as 500."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(
        proxy_mod, "_start_streaming_request",
        _make_failing_streaming_mock(IOError("network reset"))
    )

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "prompt": "hello"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected)
    assert data["error"] == "network reset"
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["args"][1] == 500


# ---------------------------------------------------------------------------
# handle_stream_generate streaming — was_corrected=True (lines 966-967)
# ---------------------------------------------------------------------------

async def test_handle_stream_generate_streaming_was_corrected(cfg, engine_and_logger, monkeypatch):
    """NDJSON generate response with think-tags triggers normalization."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)

    ndjson_chunks = [
        b'{"message":{"role":"assistant","content":"<think>thinking step</think>Final answer"},"done":true}\n',
    ]
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock(ndjson_chunks))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "prompt": "hello"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    body = json.loads(collected)
    assert "<think>" not in body["message"]["content"]
    assert len(logged_calls) >= 1
    assert logged_calls[-1]["kwargs"].get("correction_applied") is not None


# ---------------------------------------------------------------------------
# _detect_context_overflow — unit tests
# ---------------------------------------------------------------------------

def test_detect_context_overflow_ollama_done_reason_length_no_tokens():
    """done_reason='length' without token counts falls back to signal — treat as overflow."""
    assert _detect_context_overflow({"done_reason": "length", "done": True}) is True


def test_detect_context_overflow_ollama_done_reason_stop():
    assert _detect_context_overflow({"done_reason": "stop", "done": True}) is False


def test_detect_context_overflow_openai_finish_reason_length_no_tokens():
    """finish_reason='length' without token counts falls back to signal — treat as overflow."""
    parsed = {"choices": [{"finish_reason": "length", "message": {"content": "hi"}}]}
    assert _detect_context_overflow(parsed) is True


def test_detect_context_overflow_openai_finish_reason_stop():
    parsed = {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}
    assert _detect_context_overflow(parsed) is False


def test_detect_context_overflow_empty_choices():
    assert _detect_context_overflow({"choices": []}) is False


def test_detect_context_overflow_none():
    assert _detect_context_overflow(None) is False


def test_detect_context_overflow_non_dict():
    assert _detect_context_overflow("string") is False


def test_detect_context_overflow_empty_dict():
    assert _detect_context_overflow({}) is False


def test_detect_context_overflow_utilization_at_100pct():
    """prompt_eval_count == context_size triggers overflow (>= 100%)."""
    assert _detect_context_overflow({"prompt_eval_count": 8192, "done": True}, context_size=8192) is True


def test_detect_context_overflow_utilization_above_100pct():
    assert _detect_context_overflow({"prompt_eval_count": 9000, "done": True}, context_size=8192) is True


def test_detect_context_overflow_utilization_below_100pct():
    assert _detect_context_overflow({"prompt_eval_count": 4096, "done": True}, context_size=8192) is False


def test_detect_context_overflow_utilization_no_context_size():
    """Without context_size, token count alone cannot trigger overflow."""
    assert _detect_context_overflow({"prompt_eval_count": 9000, "done": True}, context_size=None) is False


def test_detect_context_overflow_openai_usage_prompt_tokens():
    """OpenAI usage.prompt_tokens >= context_size triggers overflow."""
    parsed = {
        "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 8192, "completion_tokens": 10},
    }
    assert _detect_context_overflow(parsed, context_size=8192) is True


def test_detect_context_overflow_openai_usage_below():
    parsed = {
        "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
    }
    assert _detect_context_overflow(parsed, context_size=8192) is False


def test_detect_context_overflow_done_reason_length_below_context():
    """done_reason='length' with prompt_tokens < context_size is a num_predict stop, not overflow."""
    assert _detect_context_overflow(
        {"done_reason": "length", "prompt_eval_count": 1000, "done": True},
        context_size=32768,
    ) is False


def test_detect_context_overflow_done_reason_length_at_context():
    """done_reason='length' with prompt_tokens == context_size is a real context overflow."""
    assert _detect_context_overflow(
        {"done_reason": "length", "prompt_eval_count": 32768, "done": True},
        context_size=32768,
    ) is True


def test_detect_context_overflow_finish_reason_length_below_context():
    """finish_reason='length' with usage.prompt_tokens < context_size is a num_predict stop."""
    parsed = {
        "choices": [{"finish_reason": "length", "message": {"content": "truncated"}}],
        "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
    }
    assert _detect_context_overflow(parsed, context_size=32768) is False


def test_detect_context_overflow_finish_reason_length_at_context():
    """finish_reason='length' with usage.prompt_tokens == context_size is real overflow."""
    parsed = {
        "choices": [{"finish_reason": "length", "message": {"content": "truncated"}}],
        "usage": {"prompt_tokens": 32768, "completion_tokens": 10},
    }
    assert _detect_context_overflow(parsed, context_size=32768) is True


# ---------------------------------------------------------------------------
# _resolve_context_size — unit tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_resolve_context_size_returns_ps_info(monkeypatch):
    """When /api/ps returns context_size for the model, use it."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(
        proxy_mod, "_get_ollama_ps_info",
        lambda model, target: {"context_size": 16384},
    )
    result = await _resolve_context_size("qwen3:9b", "http://localhost:11434", 8192)
    assert result == 16384


@pytest.mark.asyncio
async def test_resolve_context_size_fallback_to_config(monkeypatch):
    """When /api/ps returns no context_size, fall back to config value."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(
        proxy_mod, "_get_ollama_ps_info",
        lambda model, target: {},
    )
    result = await _resolve_context_size("qwen3:9b", "http://localhost:11434", 8192)
    assert result == 8192


@pytest.mark.asyncio
async def test_resolve_context_size_ps_raises_falls_back(monkeypatch):
    """When /api/ps raises, fall back to config value."""
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(
        proxy_mod, "_get_ollama_ps_info",
        lambda model, target: (_ for _ in ()).throw(RuntimeError("unreachable")),
    )
    result = await _resolve_context_size("qwen3:9b", "http://localhost:11434", 4096)
    assert result == 4096


@pytest.mark.asyncio
async def test_resolve_context_size_no_model_skips_ps(monkeypatch):
    """Without a model name, /api/ps is never queried; returns config value."""
    import prompt_interceptor.proxy as proxy_mod
    called = []
    monkeypatch.setattr(
        proxy_mod, "_get_ollama_ps_info",
        lambda model, target: called.append(1) or {},
    )
    result = await _resolve_context_size(None, "http://localhost:11434", 4096)
    assert result == 4096
    assert called == []


# ---------------------------------------------------------------------------
# handle_chat_request — context overflow (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_chat_context_overflow_returns_413(cfg, engine_and_logger, monkeypatch):
    """When Ollama signals done_reason=length, handle_chat_request returns 413."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))

    overflow_body = json.dumps({
        "model": "qwen3:9b",
        "message": {"role": "assistant", "content": "truncated"},
        "done": True,
        "done_reason": "length",
        "prompt_eval_count": 8192,
        "eval_count": 1,
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, overflow_body)),
    ):
        req = make_req({"model": "qwen3:9b", "messages": [{"role": "user", "content": "hi"}]})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 413
    assert json.loads(resp.body)["error"] == _CONTEXT_OVERFLOW_MSG


@pytest.mark.asyncio
async def test_handle_chat_context_overflow_by_utilization(cfg, engine_and_logger, monkeypatch):
    """When prompt tokens >= resolved context_size, handle_chat_request returns 413."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=8192))

    overflow_body = json.dumps({
        "model": "qwen3:9b",
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 8192,
        "eval_count": 5,
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, overflow_body)),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []})
        resp = await handle_chat_request(req, engine, logger)

    assert resp.status_code == 413
    assert json.loads(resp.body)["error"] == _CONTEXT_OVERFLOW_MSG


# ---------------------------------------------------------------------------
# handle_generate_request — context overflow (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_generate_context_overflow_returns_413(cfg, engine_and_logger, monkeypatch):
    """When Ollama signals done_reason=length, handle_generate_request returns 413."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))

    overflow_body = json.dumps({
        "model": "qwen3:9b",
        "response": "truncated",
        "done": True,
        "done_reason": "length",
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, overflow_body)),
    ):
        req = make_req({"model": "qwen3:9b", "prompt": "long prompt"}, path="/api/generate")
        resp = await handle_generate_request(req, engine, logger)

    assert resp.status_code == 413
    assert json.loads(resp.body)["error"] == _CONTEXT_OVERFLOW_MSG


# ---------------------------------------------------------------------------
# handle_stream_chat — context overflow (streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_stream_chat_context_overflow_yields_error(cfg, engine_and_logger, monkeypatch):
    """Streaming /api/chat with done_reason=length yields error chunk instead of response."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))

    ndjson = json.dumps({
        "message": {"role": "assistant", "content": "cut off"},
        "done": True,
        "done_reason": "length",
    }).encode() + b"\n"
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([ndjson]))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected.strip())
    assert data["error"] == _CONTEXT_OVERFLOW_MSG
    assert logged_calls[-1]["args"][1] == 413


@pytest.mark.asyncio
async def test_handle_stream_chat_context_overflow_by_utilization(cfg, engine_and_logger, monkeypatch):
    """Streaming /api/chat: prompt tokens >= resolved context triggers 413 even with done_reason=stop."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=4096))

    ndjson = json.dumps({
        "message": {"role": "assistant", "content": "ok"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 4096,
        "eval_count": 10,
    }).encode() + b"\n"
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([ndjson]))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": []})
    resp = await handle_stream_chat(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected.strip())
    assert data["error"] == _CONTEXT_OVERFLOW_MSG
    assert logged_calls[-1]["args"][1] == 413


# ---------------------------------------------------------------------------
# handle_stream_generate — context overflow (streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_stream_generate_context_overflow_yields_error(cfg, engine_and_logger, monkeypatch):
    """Streaming /api/generate with done_reason=length yields error chunk instead of response."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))

    ndjson = json.dumps({
        "response": "cut off",
        "done": True,
        "done_reason": "length",
    }).encode() + b"\n"
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([ndjson]))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "prompt": "hi"}, path="/api/generate")
    resp = await handle_stream_generate(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected.strip())
    assert data["error"] == _CONTEXT_OVERFLOW_MSG
    assert logged_calls[-1]["args"][1] == 413


# ---------------------------------------------------------------------------
# handle_v1_chat_completions — context overflow (streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_v1_chat_completions_streaming_context_overflow(cfg, engine_and_logger, monkeypatch):
    """SSE /v1/chat/completions with finish_reason=length yields error chunk instead of response."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))

    sse_chunk = (
        b'data: {"id":"c1","choices":[{"delta":{"content":"cut"},"finish_reason":"length"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    monkeypatch.setattr(proxy_mod, "_start_streaming_request", _make_start_streaming_mock([sse_chunk]))

    logged_calls = []
    original = logger.log_response

    def capture(*a, **kw):
        logged_calls.append({"args": a, "kwargs": kw})
        return original(*a, **kw)

    logger.log_response = capture

    req = make_req({"model": "qwen3:9b", "messages": [], "stream": True}, path="/v1/chat/completions")
    resp = await handle_v1_chat_completions(req, engine, logger)
    collected = b"".join([chunk async for chunk in resp.body_iterator])

    data = json.loads(collected.strip())
    assert data["error"] == _CONTEXT_OVERFLOW_MSG
    assert logged_calls[-1]["args"][1] == 413


# ---------------------------------------------------------------------------
# handle_v1_chat_completions — context overflow (non-streaming)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_handle_v1_chat_completions_non_streaming_context_overflow(cfg, engine_and_logger, monkeypatch):
    """Non-streaming /v1/chat/completions with finish_reason=length returns 413."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=None))

    overflow_body = json.dumps({
        "id": "c1",
        "choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "cut"}}],
        "usage": {"prompt_tokens": 8192, "completion_tokens": 1},
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, overflow_body)),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 413
    assert json.loads(resp.body)["error"] == _CONTEXT_OVERFLOW_MSG


@pytest.mark.asyncio
async def test_handle_v1_chat_completions_non_streaming_overflow_by_utilization(cfg, engine_and_logger, monkeypatch):
    """Non-streaming /v1/chat/completions: usage.prompt_tokens >= resolved ctx returns 413."""
    engine, logger = engine_and_logger
    import prompt_interceptor.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "get_config", lambda: cfg)
    monkeypatch.setattr(proxy_mod, "_resolve_context_size", AsyncMock(return_value=8192))

    overflow_body = json.dumps({
        "id": "c1",
        "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}],
        "usage": {"prompt_tokens": 8192, "completion_tokens": 5},
    }).encode()

    with patch(
        "prompt_interceptor.proxy._fetch_from_ollama",
        new=AsyncMock(return_value=(200, {"content-type": "application/json"}, overflow_body)),
    ):
        req = make_req({"model": "qwen3:9b", "messages": []}, path="/v1/chat/completions")
        resp = await handle_v1_chat_completions(req, engine, logger)

    assert resp.status_code == 413
    assert json.loads(resp.body)["error"] == _CONTEXT_OVERFLOW_MSG
