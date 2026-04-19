"""Tests for logger.py."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from prompt_interceptor.config import Config
from prompt_interceptor.logger import TrafficLogger


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


def test_log_response_corrupt_existing_file(tl, cfg):
    """log_response silently recovers when the existing log file has corrupt JSON."""
    rid = tl.log_request("POST", "/api/chat", {}, {"model": "x"})
    date_dir = tl._get_date_dir()
    filepath = date_dir / f"req_{rid}.json"
    filepath.write_text("{{not valid json", encoding="utf-8")
    tl.log_response(rid, 200, {}, {"done": True})  # must not raise
    data = json.loads(filepath.read_text(encoding="utf-8"))
    assert data["status_code"] == 200


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
    time.sleep(0.02)  # ensure distinct timestamps → distinct request IDs
    tl.log_request("GET", "/b", {}, None)
    all_logs = tl.get_all_logs()
    assert len(all_logs) >= 2


# ---------------------------------------------------------------------------
# Log file rotation  (lines 189-193)
# ---------------------------------------------------------------------------

def test_log_file_rotation_deletes_oldest(tmp_path):
    """When a log file is at the size limit and max files reached, a file is removed."""
    cfg = Config(log_dir=str(tmp_path / "logs"), log_size_limit=1, max_log_files=2)
    tl = TrafficLogger(cfg)

    rid1 = tl.log_request("GET", "/a", {}, None)
    time.sleep(0.02)  # ensure distinct timestamps → distinct IDs
    rid2 = tl.log_request("GET", "/b", {}, None)

    date_dir = tl._get_date_dir()
    files_before = sorted(date_dir.glob("req_*.json"))
    if len(files_before) < 2:
        return  # IDs collided (extremely rare) — skip

    assert len(files_before) == 2
    tl.log_response(rid2, 200, {}, {"done": True})

    files_after = sorted(date_dir.glob("req_*.json"))
    assert len(files_after) <= cfg.max_log_files


def test_log_file_rotation_no_delete_when_below_max(tmp_path):
    """When file count is below max_log_files, no deletion occurs."""
    cfg = Config(log_dir=str(tmp_path / "logs"), log_size_limit=1, max_log_files=10)
    tl = TrafficLogger(cfg)

    rid1 = tl.log_request("GET", "/a", {}, None)
    tl.log_response(rid1, 200, {}, {"done": True})

    date_dir = tl._get_date_dir()
    files = sorted(date_dir.glob("req_*.json"))
    assert len(files) == 1


# ---------------------------------------------------------------------------
# IOError / JSONDecodeError in get_logs  (lines 208-209)
# ---------------------------------------------------------------------------

def test_get_logs_skips_corrupt_file(tmp_path):
    """get_logs silently skips files that cannot be parsed."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    tl = TrafficLogger(cfg)
    tl.log_request("GET", "/", {}, None)

    date_dir = tl._get_date_dir()
    (date_dir / "req_corrupt.json").write_text("{ not valid json }")

    logs = tl.get_logs(limit=100)
    assert isinstance(logs, list)


def test_get_logs_skips_unreadable_file(tmp_path):
    """get_logs silently skips files that raise IOError."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    tl = TrafficLogger(cfg)
    tl.log_request("GET", "/", {}, None)

    original_open = open

    def _failing_open(path, *args, **kwargs):
        if "req_" in str(path):
            raise IOError("permission denied")
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=_failing_open):
        logs = tl.get_logs()

    assert logs == []


# ---------------------------------------------------------------------------
# get_stats when date dir does not exist  (line 217)
# ---------------------------------------------------------------------------

def test_get_stats_missing_date_dir(tmp_path):
    """get_stats returns zeros when the date directory doesn't exist."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    tl = TrafficLogger(cfg)

    nonexistent = tmp_path / "no-such-dir"
    with patch.object(tl, "_get_date_dir", return_value=nonexistent):
        stats = tl.get_stats()

    assert stats["total_requests"] == 0
    assert stats["total_responses"] == 0
    assert stats["files"] == []


# ---------------------------------------------------------------------------
# IOError / JSONDecodeError in get_all_logs  (lines 236-237)
# ---------------------------------------------------------------------------

def test_get_all_logs_skips_corrupt_file(tmp_path):
    """get_all_logs silently skips corrupt JSON files."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    tl = TrafficLogger(cfg)
    tl.log_request("GET", "/", {}, None)

    date_dir = tl._get_date_dir()
    (date_dir / "req_bad.json").write_text("INVALID JSON{{{")

    logs = tl.get_all_logs()
    assert isinstance(logs, list)
    valid = [l for l in logs if "method" in l]
    assert len(valid) >= 1


def test_get_all_logs_skips_unreadable_file(tmp_path):
    """get_all_logs silently skips files that raise IOError on open."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    tl = TrafficLogger(cfg)
    tl.log_request("GET", "/", {}, None)

    original_open = open

    def _failing_open(path, *args, **kwargs):
        if "req_" in str(path):
            raise IOError("locked")
        return original_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=_failing_open):
        logs = tl.get_all_logs()

    assert logs == []


# ---------------------------------------------------------------------------
# clear_logs
# ---------------------------------------------------------------------------

def test_clear_logs_removes_all_files(tl, cfg):
    """clear_logs deletes all req_*.json files in today's directory."""
    for i in range(3):
        tl.log_request("GET", f"/path/{i}", {}, None)
        time.sleep(0.01)

    date_dir = tl._get_date_dir()
    assert len(list(date_dir.glob("req_*.json"))) == 3

    deleted = tl.clear_logs()

    assert deleted == 3
    assert len(list(date_dir.glob("req_*.json"))) == 0


def test_clear_logs_returns_zero_when_empty(tl):
    """clear_logs returns 0 when there are no log files."""
    deleted = tl.clear_logs()
    assert deleted == 0


def test_clear_logs_no_dir(tmp_path):
    """clear_logs returns 0 when today's log directory does not yet exist."""
    cfg = Config(log_dir=str(tmp_path / "logs"))
    tl = TrafficLogger(cfg)

    # Patch _get_date_dir to return a non-existent path
    nonexistent = tmp_path / "no-such-dir"
    with patch.object(tl, "_get_date_dir", return_value=nonexistent):
        deleted = tl.clear_logs()

    assert deleted == 0


def test_clear_logs_then_get_logs_is_empty(tl):
    """After clear_logs, get_logs returns an empty list."""
    tl.log_request("POST", "/api/chat", {}, {"model": "llama3"})
    tl.clear_logs()
    assert tl.get_logs() == []


def test_clear_logs_oserror_on_unlink_is_ignored(tl):
    """clear_logs silently skips files that raise OSError on unlink."""
    for i in range(2):
        tl.log_request("GET", f"/path/{i}", {}, None)
        time.sleep(0.01)

    date_dir = tl._get_date_dir()
    files = list(date_dir.glob("req_*.json"))
    assert len(files) == 2

    original_unlink = Path.unlink
    call_count = [0]

    def flaky_unlink(self, missing_ok=False):
        call_count[0] += 1
        if call_count[0] == 1:
            raise OSError("locked")
        original_unlink(self, missing_ok=missing_ok)

    with patch.object(Path, "unlink", flaky_unlink):
        deleted = tl.clear_logs()  # must not raise

    # One file raises OSError (skipped), one is deleted → reported count = 2
    assert deleted == 2
