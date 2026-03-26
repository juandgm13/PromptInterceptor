"""Tests for logger.py."""

import json
import time
from pathlib import Path

import pytest

from ollama_proxy.config import Config
from ollama_proxy.logger import TrafficLogger


@pytest.fixture
def cfg(tmp_path):
    return Config(log_dir=str(tmp_path / "logs"), log_size_limit=512, max_log_files=3)


@pytest.fixture
def tl(cfg):
    return TrafficLogger(cfg)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_log_dir_created(cfg, tl):
    assert Path(cfg.log_dir).exists()


# ---------------------------------------------------------------------------
# log_request
# ---------------------------------------------------------------------------

def test_log_request_returns_str(tl):
    rid = tl.log_request("GET", "/api/chat", {}, None)
    assert isinstance(rid, str)
    assert len(rid) == 12


def test_log_request_writes_file(tl, cfg):
    rid = tl.log_request("POST", "/api/chat", {"x-foo": "bar"}, {"model": "llama3"})
    log_files = list(Path(cfg.log_dir).rglob("req_*.json"))
    assert len(log_files) == 1
    data = json.loads(log_files[0].read_text())
    assert data["type"] == "request"
    assert data["method"] == "POST"
    assert data["path"] == "/api/chat"
    assert data["request_id"] == rid


def test_log_request_filters_sensitive_headers(tl, cfg):
    tl.log_request("GET", "/", {"authorization": "Bearer secret", "x-safe": "ok"}, None)
    data = json.loads(list(Path(cfg.log_dir).rglob("req_*.json"))[0].read_text())
    assert "authorization" not in data["headers"]
    assert data["headers"].get("x-safe") == "ok"


# ---------------------------------------------------------------------------
# log_response
# ---------------------------------------------------------------------------

def test_log_response_writes_file(tl, cfg):
    rid = tl.log_request("POST", "/api/chat", {}, {})
    tl.log_response(rid, 200, {"content-type": "application/json"}, {"done": True})
    files = list(Path(cfg.log_dir).rglob("req_*.json"))
    # Two writes for same request_id → one file (overwritten)
    assert len(files) == 1


def test_log_response_no_body(tl):
    rid = tl.log_request("POST", "/api/chat", {}, None)
    tl.log_response(rid, 204, {}, None)  # should not raise


# ---------------------------------------------------------------------------
# _truncate_body
# ---------------------------------------------------------------------------

def test_truncate_body_small_passthrough(tl):
    data = {"key": "value"}
    assert tl._truncate_body(data) == data


def test_truncate_body_large_marked(tl, cfg):
    large = {"key": "x" * (cfg.log_size_limit + 1)}
    result = tl._truncate_body(large)
    assert result["_truncated"] is True
    assert "_original_size" in result


def test_truncate_body_none(tl):
    assert tl._truncate_body(None) is None


# ---------------------------------------------------------------------------
# log_raw_request
# ---------------------------------------------------------------------------

def test_log_raw_request_valid_json(tl):
    tl.log_raw_request("rid1", '{"model": "llama3"}', {"x-test": "1"})


def test_log_raw_request_invalid_json(tl):
    tl.log_raw_request("rid1", "not json", {})  # should not raise


def test_log_raw_request_empty(tl):
    tl.log_raw_request("rid1", "", {})  # should not raise


# ---------------------------------------------------------------------------
# log_intercepted
# ---------------------------------------------------------------------------

def test_log_intercepted_modified(tl, cfg):
    rid = tl.log_request("POST", "/", {}, {})
    tl.log_intercepted(rid, modified=True)


def test_log_intercepted_not_modified(tl):
    rid = tl.log_request("POST", "/", {}, {})
    tl.log_intercepted(rid, modified=False)


# ---------------------------------------------------------------------------
# get_logs
# ---------------------------------------------------------------------------

def test_get_logs_empty(tl):
    assert tl.get_logs() == []


def test_get_logs_returns_entries(tl):
    for i in range(3):
        tl.log_request("GET", f"/path/{i}", {}, None)
        time.sleep(0.01)
    logs = tl.get_logs(limit=10)
    assert len(logs) == 3


def test_get_logs_respects_limit(tl):
    for i in range(5):
        tl.log_request("GET", f"/path/{i}", {}, None)
        time.sleep(0.01)
    logs = tl.get_logs(limit=2)
    assert len(logs) == 2


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

def test_get_stats_empty(tl):
    stats = tl.get_stats()
    assert stats["total_requests"] == 0


def test_get_stats_with_entries(tl):
    tl.log_request("GET", "/", {}, None)
    stats = tl.get_stats()
    assert stats["total_requests"] >= 1


# ---------------------------------------------------------------------------
# get_all_logs
# ---------------------------------------------------------------------------

def test_get_all_logs_empty(tl):
    assert tl.get_all_logs() == []


def test_get_all_logs_returns_all(tl):
    tl.log_request("GET", "/a", {}, None)
    tl.log_request("GET", "/b", {}, None)
    all_logs = tl.get_all_logs()
    assert len(all_logs) >= 2
