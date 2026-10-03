"""Tests for logger.py."""

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from prompt_interceptor.config import Config
from prompt_interceptor.logger import TrafficLogger, _get_ollama_ps_info


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


def test_log_response_correction_applied_stored(tl, cfg):
    """correction_applied field is saved in log when provided."""
    rid = tl.log_request("POST", "/api/chat", {}, {"model": "qwen3"})
    tl.log_response(rid, 200, {}, {"done": True}, correction_applied="think→thinking")
    files = list(Path(cfg.log_dir).rglob("req_*.json"))
    data = json.loads(files[0].read_text())
    assert data.get("_correction_applied") == "think→thinking"


def test_log_response_no_correction_field_omitted(tl, cfg):
    """When correction_applied is None, _correction_applied is not saved."""
    rid = tl.log_request("POST", "/api/chat", {}, {})
    tl.log_response(rid, 200, {}, {"done": True})
    files = list(Path(cfg.log_dir).rglob("req_*.json"))
    data = json.loads(files[0].read_text())
    assert "_correction_applied" not in data


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
# start_timer / _duration_ms
# ---------------------------------------------------------------------------

def test_start_timer_records_duration(tl, cfg):
    """log_response stores _duration_ms when the request was timed."""
    rid = tl.log_request("POST", "/api/chat", {}, None)
    tl.start_timer(rid)
    time.sleep(0.01)
    tl.log_response(rid, 200, {}, {"done": True})

    data = json.loads((tl._get_date_dir() / f"req_{rid}.json").read_text())
    assert isinstance(data["_duration_ms"], int)
    assert data["_duration_ms"] >= 0


def test_no_duration_field_without_start_timer(tl, cfg):
    """_duration_ms is omitted entirely when the request was never timed."""
    rid = tl.log_request("POST", "/api/chat", {}, None)
    tl.log_response(rid, 200, {}, {"done": True})

    data = json.loads((tl._get_date_dir() / f"req_{rid}.json").read_text())
    assert "_duration_ms" not in data


def test_start_timer_ignores_empty_request_id(tl):
    """start_timer is a no-op for a missing request id (passthrough w/o logger id)."""
    tl.start_timer(None)
    tl.start_timer("")
    assert tl._forward_start == {}


def test_start_timer_dict_is_bounded(tl):
    """Requests that never reach log_response must not grow the dict forever."""
    from prompt_interceptor.logger import _MAX_TIMERS

    for i in range(_MAX_TIMERS + 50):
        tl.start_timer(f"{i:012x}")

    assert len(tl._forward_start) <= _MAX_TIMERS


def test_start_timer_evicts_oldest_first(tl):
    """The bound drops the oldest mark, keeping the most recent ones."""
    from prompt_interceptor.logger import _MAX_TIMERS

    for i in range(_MAX_TIMERS + 1):
        tl.start_timer(f"{i:012x}")

    assert f"{0:012x}" not in tl._forward_start
    assert f"{_MAX_TIMERS:012x}" in tl._forward_start


# ---------------------------------------------------------------------------
# delete_log
# ---------------------------------------------------------------------------

def test_delete_log_removes_only_that_entry(tl):
    """delete_log removes the requested file and leaves the others alone."""
    rids = [tl.log_request("GET", f"/path/{i}", {}, None) for i in range(3)]
    date_dir = tl._get_date_dir()

    assert tl.delete_log(rids[1]) is True

    remaining = {f.name for f in date_dir.glob("req_*.json")}
    assert remaining == {f"req_{rids[0]}.json", f"req_{rids[2]}.json"}


def test_delete_log_missing_file_returns_false(tl):
    """A well-formed but unknown request id is a no-op."""
    assert tl.delete_log("abcdef123456") is False


@pytest.mark.parametrize("bad_id", [
    "../../etc/passwd",
    "..",
    "",
    None,
    "req_abcdef123456",
    "ABCDEF123456",      # uppercase is not what _generate_request_id emits
    "abcdef12345",       # too short
    "abcdef1234567",     # too long
    "abcdef12345/",
])
def test_delete_log_rejects_malformed_ids(tl, bad_id):
    """delete_log refuses anything that is not a 12-char hex id, untouched disk."""
    rid = tl.log_request("GET", "/api/chat", {}, None)
    date_dir = tl._get_date_dir()
    before = {f.name for f in date_dir.glob("*")}

    assert tl.delete_log(bad_id) is False
    assert {f.name for f in date_dir.glob("*")} == before
    assert (date_dir / f"req_{rid}.json").exists()


def test_delete_log_does_not_escape_log_dir(tl, tmp_path):
    """A traversal attempt cannot reach a file outside the log directory."""
    victim = tmp_path / "victim.txt"
    victim.write_text("keep me")

    assert tl.delete_log(f"../../{victim.name}") is False
    assert victim.exists()


def test_delete_log_then_get_logs_excludes_it(tl):
    """The deleted entry no longer shows up in get_logs."""
    rids = [tl.log_request("GET", f"/path/{i}", {}, None) for i in range(3)]
    tl.delete_log(rids[0])

    got = {entry["request_id"] for entry in tl.get_logs(limit=10)}
    assert got == {rids[1], rids[2]}


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


# ---------------------------------------------------------------------------
# _get_ollama_ps_info
# ---------------------------------------------------------------------------

def _mock_ps_client(status_code: int, models: list):
    """Build a patched httpx.Client that returns a fake /api/ps response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = {"models": models}
    mock_client = MagicMock()
    mock_client.__enter__ = lambda s: mock_client
    mock_client.__exit__ = MagicMock(return_value=False)
    mock_client.get.return_value = mock_resp
    return mock_client


def test_get_ollama_ps_info_returns_context_and_gpu():
    model = {"name": "llama3:latest", "num_ctx": 16384, "size": 8_000_000_000, "size_vram": 8_000_000_000}
    with patch("prompt_interceptor.logger.httpx.Client", return_value=_mock_ps_client(200, [model])):
        info = _get_ollama_ps_info("llama3", "http://localhost:11434")
    assert info["context_size"] == 16384
    assert info["offload"] == "gpu"
    assert info["size_vram"] == 8_000_000_000


def test_get_ollama_ps_info_partial_gpu():
    model = {"name": "llama3:latest", "num_ctx": 8192, "size": 8_000_000_000, "size_vram": 4_000_000_000}
    with patch("prompt_interceptor.logger.httpx.Client", return_value=_mock_ps_client(200, [model])):
        info = _get_ollama_ps_info("llama3", "http://localhost:11434")
    assert info["offload"] == "partial"


def test_get_ollama_ps_info_cpu_only():
    model = {"name": "llama3:latest", "num_ctx": 8192, "size": 8_000_000_000, "size_vram": 0}
    with patch("prompt_interceptor.logger.httpx.Client", return_value=_mock_ps_client(200, [model])):
        info = _get_ollama_ps_info("llama3", "http://localhost:11434")
    assert info["offload"] == "cpu"


def test_get_ollama_ps_info_model_not_loaded():
    """Returns {} when the model isn't in the ps list."""
    model = {"name": "other:latest", "num_ctx": 4096, "size": 1000, "size_vram": 1000}
    with patch("prompt_interceptor.logger.httpx.Client", return_value=_mock_ps_client(200, [model])):
        info = _get_ollama_ps_info("llama3", "http://localhost:11434")
    assert info == {}


def test_get_ollama_ps_info_non_200_returns_empty():
    with patch("prompt_interceptor.logger.httpx.Client", return_value=_mock_ps_client(503, [])):
        assert _get_ollama_ps_info("llama3", "http://localhost:11434") == {}


def test_get_ollama_ps_info_exception_returns_empty():
    with patch("prompt_interceptor.logger.httpx.Client", side_effect=Exception("refused")):
        assert _get_ollama_ps_info("llama3", "http://localhost:11434") == {}


def test_get_ollama_ps_info_no_context_field():
    """Returns {} context_size when ps has no known num_ctx field."""
    model = {"name": "llama3:latest", "size": 1000, "size_vram": 1000}
    with patch("prompt_interceptor.logger.httpx.Client", return_value=_mock_ps_client(200, [model])):
        info = _get_ollama_ps_info("llama3", "http://localhost:11434")
    assert "context_size" not in info
    assert info["offload"] == "gpu"


# ---------------------------------------------------------------------------
# _extract_token_usage
# ---------------------------------------------------------------------------

def test_extract_token_usage_none_body(tl):
    assert tl._extract_token_usage(None) is None


def test_extract_token_usage_no_token_fields(tl):
    assert tl._extract_token_usage({"done": True, "model": "x"}) is None


def test_extract_token_usage_ollama_native(tl, cfg):
    body = {"done": True, "prompt_eval_count": 100, "eval_count": 50}
    result = tl._extract_token_usage(body)
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["total_tokens"] == 150
    assert result["context_size"] == cfg.context_size
    # Percentage uses prompt_tokens only (context fill before generation)
    assert result["context_utilization_pct"] == round(100 / cfg.context_size * 100, 1)


def test_extract_token_usage_openai_compat(tl):
    body = {"usage": {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275}}
    result = tl._extract_token_usage(body)
    assert result["prompt_tokens"] == 200
    assert result["completion_tokens"] == 75
    assert result["total_tokens"] == 275


def test_extract_token_usage_anthropic_compat(tl):
    body = {"usage": {"input_tokens": 300, "output_tokens": 60}}
    result = tl._extract_token_usage(body)
    assert result["prompt_tokens"] == 300
    assert result["completion_tokens"] == 60
    assert result["total_tokens"] == 360


def test_extract_token_usage_uses_ollama_ps_ctx_when_available(tl):
    body = {"prompt_eval_count": 100, "eval_count": 50}
    ps = {"context_size": 16384, "offload": "gpu", "size_vram": 8_000_000_000, "size_total": 8_000_000_000}
    with patch("prompt_interceptor.logger._get_ollama_ps_info", return_value=ps):
        result = tl._extract_token_usage(body, model_name="llama3")
    assert result["context_size"] == 16384
    assert result["context_utilization_pct"] == round(100 / 16384 * 100, 1)
    assert result["offload"] == "gpu"
    assert result["size_vram"] == 8_000_000_000


def test_extract_token_usage_falls_back_to_config_when_ps_empty(tl, cfg):
    body = {"prompt_eval_count": 100, "eval_count": 50}
    with patch("prompt_interceptor.logger._get_ollama_ps_info", return_value={}):
        result = tl._extract_token_usage(body, model_name="llama3")
    assert result["context_size"] == cfg.context_size
    assert "offload" not in result


# ---------------------------------------------------------------------------
# _tokens_usage stored in log_response
# ---------------------------------------------------------------------------

def test_log_response_stores_tokens_usage_ollama_native(tl, cfg):
    """_tokens_usage is written to the log file for Ollama native responses."""
    rid = tl.log_request("POST", "/api/chat", {}, {"model": "llama3"})
    with patch("prompt_interceptor.logger._get_ollama_ps_info", return_value={}):
        tl.log_response(rid, 200, {}, {"model": "llama3", "done": True, "prompt_eval_count": 500, "eval_count": 100})
    data = json.loads(list(Path(cfg.log_dir).rglob("req_*.json"))[0].read_text())
    assert "_tokens_usage" in data
    assert data["_tokens_usage"]["prompt_tokens"] == 500
    assert data["_tokens_usage"]["total_tokens"] == 600


def test_log_response_no_tokens_usage_when_absent(tl, cfg):
    """_tokens_usage is omitted when response has no token fields."""
    rid = tl.log_request("POST", "/api/chat", {}, {})
    tl.log_response(rid, 200, {}, {"done": True})
    data = json.loads(list(Path(cfg.log_dir).rglob("req_*.json"))[0].read_text())
    assert "_tokens_usage" not in data


def test_log_response_model_from_request_body_used_for_ctx(tl, cfg):
    """When response body has no model, model from request body is used for Ollama query."""
    rid = tl.log_request("POST", "/api/chat", {}, {"model": "qwen3"})
    ps = {"context_size": 16384, "offload": "gpu", "size_vram": 4_000_000_000, "size_total": 4_000_000_000}
    with patch("prompt_interceptor.logger._get_ollama_ps_info", return_value=ps) as mock_fn:
        tl.log_response(rid, 200, {}, {"done": True, "prompt_eval_count": 50, "eval_count": 20})
    mock_fn.assert_called_once_with("qwen3", tl.config.target)
    data = json.loads(list(Path(cfg.log_dir).rglob("req_*.json"))[0].read_text())
    assert data["_tokens_usage"]["context_size"] == 16384
    assert data["_tokens_usage"]["offload"] == "gpu"


# ---------------------------------------------------------------------------
# update_request_body — additional coverage paths  (lines 307-308, 316)
# ---------------------------------------------------------------------------

def test_update_request_body_corrupt_existing_file(tl, cfg):
    """Lines 307-308: update_request_body silently recovers when the existing log has corrupt JSON."""
    rid = tl.log_request("POST", "/api/chat", {}, {"model": "llama3"})
    date_dir = tl._get_date_dir()
    filepath = date_dir / f"req_{rid}.json"
    # Overwrite the log file with invalid JSON to trigger the JSONDecodeError branch
    filepath.write_text("{ this is not valid json {{{{", encoding="utf-8")

    # Must not raise; starts from empty dict when corrupt
    tl.update_request_body(rid, {"model": "qwen3"})

    data = json.loads(filepath.read_text(encoding="utf-8"))
    assert data["body"]["model"] == "qwen3"


def test_update_request_body_intercepted_modified_flag(tl, cfg):
    """Line 316: update_request_body with intercepted_modified=True sets _intercepted_modified in the log."""
    rid = tl.log_request("POST", "/api/chat", {}, {"model": "llama3"})

    tl.update_request_body(rid, {"model": "qwen3"}, intercepted_modified=True)

    date_dir = tl._get_date_dir()
    filepath = date_dir / f"req_{rid}.json"
    data = json.loads(filepath.read_text(encoding="utf-8"))
    assert data.get("_intercepted_modified") is True


# ---------------------------------------------------------------------------
# Image helpers (vision requests)
# ---------------------------------------------------------------------------

from prompt_interceptor.logger import extract_images, strip_images  # noqa: E402

_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGD4DwABBAEAwS2OUAAAAABJRU5ErkJggg=="
_JPEG = "/9j/4AAQSkZJRgABAQAAAQABAAD"


def test_extract_images_ollama_chat_and_generate():
    chat = {"messages": [{"role": "system", "content": "s"},
                         {"role": "user", "content": "q", "images": [_PNG, _JPEG]}]}
    imgs = extract_images(chat)
    assert [i["mime"] for i in imgs] == ["image/png", "image/jpeg"]
    assert imgs[0]["data"] == _PNG
    assert imgs[0]["location"] == "messages[1]"

    gen = extract_images({"prompt": "q", "images": ["R0lGODlhAQABAAAAACw="]})
    assert gen == [{"mime": "image/gif", "data": "R0lGODlhAQABAAAAACw=", "location": "prompt"}]


def test_extract_images_openai_formats():
    body = {"messages": [{"role": "user", "content": [
        {"type": "text", "text": "q"},
        {"type": "image_url", "image_url": {"url": f"data:image/webp;base64,{_PNG}"}},
        {"type": "image_url", "image_url": "https://example.com/a.png"},
    ]}]}
    imgs = extract_images(body)
    assert imgs[0] == {"mime": "image/webp", "data": _PNG, "location": "messages[0]"}
    assert imgs[1] == {"mime": None, "url": "https://example.com/a.png", "location": "messages[0]"}


def test_extract_images_anthropic_formats():
    body = {"messages": [{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _JPEG}},
        {"type": "image", "source": {"type": "url", "url": "https://example.com/b.jpg"}},
    ]}]}
    imgs = extract_images(body)
    assert imgs[0]["mime"] == "image/jpeg" and imgs[0]["data"] == _JPEG
    assert imgs[1]["url"] == "https://example.com/b.jpg"


def test_extract_images_none_and_text_only():
    assert extract_images(None) == []
    assert extract_images({"messages": [{"role": "user", "content": "hi"}]}) == []


def test_strip_images_replaces_payloads_without_mutating():
    body = {"messages": [
        {"role": "user", "content": "q", "images": [_PNG]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}},
            {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": _JPEG}},
        ]},
    ]}
    original = json.dumps(body)
    stripped, count = strip_images(body)

    assert count == 4
    assert json.dumps(body) == original
    assert stripped["messages"][0]["images"][0].startswith("<image #1,")
    blocks = stripped["messages"][1]["content"]
    assert blocks[0]["image_url"]["url"].startswith("<image #2,")
    assert blocks[1]["image_url"]["url"] == "https://example.com/a.png"
    assert blocks[2]["source"]["data"].startswith("<image #4,")
    assert _PNG not in json.dumps(stripped)


def test_strip_images_no_images_returns_same_object():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    stripped, count = strip_images(body)
    assert stripped is body and count == 0


def test_update_request_body_stores_num_ctx_override(tl):
    rid = tl.log_request("POST", "/api/chat", {}, {"options": {"num_ctx": 8192}})
    tl.update_request_body(rid, {"options": {"num_ctx": 32768}},
                           num_ctx_override={"client": 8192, "sent": 32768})
    tl.log_response(rid, 500, {}, {"error": "boom"})
    entry = tl.get_log(rid)
    assert entry["_num_ctx_override"] == {"client": 8192, "sent": 32768}
    assert entry["body"]["options"]["num_ctx"] == 32768


def test_get_log_rejects_bad_ids(tl):
    assert tl.get_log("../../etc/passwd") is None
    assert tl.get_log("abcdef123456") is None


def test_request_ids_unique_when_clock_does_not_advance(tl):
    """Requests logged in the same clock tick (common on Windows) get distinct ids."""
    import re
    from datetime import datetime as real_datetime

    frozen = real_datetime(2026, 10, 3, 12, 0, 0)

    class FrozenDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    with patch("prompt_interceptor.logger.datetime", FrozenDatetime):
        ids = [tl.log_request("GET", f"/p/{i}", {}, None) for i in range(200)]

    assert len(set(ids)) == 200
    assert all(re.fullmatch(r"[0-9a-f]{12}", i) for i in ids)
    assert len(list(tl._get_date_dir().glob("req_*.json"))) == 200


def test_image_helpers_skip_malformed_entries():
    """Non-dict messages/blocks and non-string payloads are ignored, not crashed on."""
    body = {"messages": [
        "not a message",
        {"role": "user", "images": [None, "AAAAunknownsignature"], "content": [
            "not a block",
            {"type": "image_url", "image_url": {"url": 123}},
            {"type": "image", "source": {"type": "url", "url": "https://example.com/c.png"}},
        ]},
    ]}
    imgs = extract_images(body)
    # unknown base64 signature falls back to PNG
    assert imgs[0] == {"mime": "image/png", "data": "AAAAunknownsignature", "location": "messages[1]"}
    assert imgs[1] == {"mime": None, "url": "https://example.com/c.png", "location": "messages[1]"}
    assert len(imgs) == 2

    stripped, count = strip_images(body)
    assert count == 2
    msg = stripped["messages"][1]
    assert msg["images"][0] is None
    assert msg["images"][1].startswith("<image #1,")
    assert msg["content"][1]["image_url"]["url"] == 123
    # anthropic URL images are counted but kept as-is
    assert msg["content"][2]["source"]["url"] == "https://example.com/c.png"
