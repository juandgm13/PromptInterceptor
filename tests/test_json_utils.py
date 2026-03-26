"""Tests for utils/json_utils.py."""

import pytest
from ollama_proxy.utils.json_utils import (
    safe_json_loads,
    safe_json_dumps,
    truncate_payload,
    jsonpath_get,
    jsonpath_set,
    merge_dicts,
)


# ---------------------------------------------------------------------------
# safe_json_loads
# ---------------------------------------------------------------------------

def test_safe_json_loads_string():
    assert safe_json_loads('{"key": "value"}') == {"key": "value"}


def test_safe_json_loads_passthrough_dict():
    data = {"key": "value"}
    assert safe_json_loads(data) is data


def test_safe_json_loads_passthrough_list():
    data = [1, 2, 3]
    assert safe_json_loads(data) is data


def test_safe_json_loads_invalid_returns_default():
    assert safe_json_loads("not json") is None
    assert safe_json_loads("not json", default={}) == {}


def test_safe_json_loads_none_returns_default():
    assert safe_json_loads(None) is None
    assert safe_json_loads(None, default=42) == 42


def test_safe_json_loads_bytes():
    assert safe_json_loads(b'{"x": 1}') == {"x": 1}


# ---------------------------------------------------------------------------
# safe_json_dumps
# ---------------------------------------------------------------------------

def test_safe_json_dumps_dict():
    result = safe_json_dumps({"a": 1})
    import json
    assert json.loads(result) == {"a": 1}


def test_safe_json_dumps_error_returns_empty():
    class Unserializable:
        pass
    assert safe_json_dumps(Unserializable()) == "{}"


# ---------------------------------------------------------------------------
# truncate_payload
# ---------------------------------------------------------------------------

def test_truncate_payload_small_unchanged():
    data = {"key": "value"}
    assert truncate_payload(data, max_bytes=10_000) == data


def test_truncate_payload_large_dict_marked():
    data = {"key": "x" * 2_000}
    result = truncate_payload(data, max_bytes=100)
    assert result["_truncated"] is True
    assert "_original_size_bytes" in result


def test_truncate_payload_large_non_dict():
    data = [{"key": "x" * 2_000}]
    result = truncate_payload(data, max_bytes=10)
    assert result["_truncated"] is True


def test_truncate_payload_none_passthrough():
    # None is not JSON-serialisable; handled gracefully
    result = truncate_payload(None, max_bytes=10)
    assert result is None


# ---------------------------------------------------------------------------
# jsonpath_get
# ---------------------------------------------------------------------------

def test_jsonpath_get_simple():
    data = {"model": "llama3"}
    assert jsonpath_get(data, "$.model") == ["llama3"]


def test_jsonpath_get_nested():
    data = {"options": {"temperature": 0.9}}
    assert jsonpath_get(data, "$.options.temperature") == [0.9]


def test_jsonpath_get_missing_returns_empty():
    assert jsonpath_get({}, "$.model") == []


def test_jsonpath_get_strips_optional_chaining():
    data = {"options": {"temperature": 0.5}}
    assert jsonpath_get(data, "$.options?.temperature") == [0.5]


def test_jsonpath_get_invalid_path_returns_empty():
    # Malformed expression should not raise
    result = jsonpath_get({"a": 1}, "$$$$")
    assert result == []


# ---------------------------------------------------------------------------
# jsonpath_set
# ---------------------------------------------------------------------------

def test_jsonpath_set_simple():
    data = {"model": "llama3"}
    jsonpath_set(data, "$.model", "deepseek")
    assert data["model"] == "deepseek"


def test_jsonpath_set_nested():
    data = {"options": {"temperature": 0.9}}
    jsonpath_set(data, "$.options.temperature", 0.1)
    assert data["options"]["temperature"] == 0.1


def test_jsonpath_set_missing_key_no_crash():
    data = {"other": "value"}
    jsonpath_set(data, "$.model", "x")  # key doesn't exist — no crash
    # jsonpath_ng silently does nothing when key is missing
    assert data.get("model") is None or data.get("model") == "x"


def test_jsonpath_set_returns_data():
    data = {"a": 1}
    result = jsonpath_set(data, "$.a", 99)
    assert result is data


# ---------------------------------------------------------------------------
# merge_dicts
# ---------------------------------------------------------------------------

def test_merge_dicts_flat():
    assert merge_dicts({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


def test_merge_dicts_override():
    assert merge_dicts({"a": 1}, {"a": 2}) == {"a": 2}


def test_merge_dicts_deep():
    base = {"a": {"b": 1, "c": 2}}
    override = {"a": {"c": 99, "d": 3}}
    result = merge_dicts(base, override)
    assert result == {"a": {"b": 1, "c": 99, "d": 3}}


def test_merge_dicts_non_dict_override_replaces():
    base = {"a": {"b": 1}}
    override = {"a": "string"}
    result = merge_dicts(base, override)
    assert result["a"] == "string"


def test_merge_dicts_does_not_mutate_base():
    base = {"a": 1}
    merge_dicts(base, {"b": 2})
    assert "b" not in base
