"""
JSON utility functions for PyProxy.

Provides safe JSON parsing, JSONPath helpers, and large-payload handling.
"""

import json
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Safe parsing
# ---------------------------------------------------------------------------

def safe_json_loads(data: Any, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any error."""
    if data is None:
        return default
    if isinstance(data, (dict, list)):
        return data
    try:
        return json.loads(data)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


def safe_json_dumps(data: Any, **kwargs) -> str:
    """Serialise to JSON string, returning '{}' on error."""
    try:
        return json.dumps(data, **kwargs)
    except (TypeError, ValueError):
        return "{}"


# ---------------------------------------------------------------------------
# Payload truncation
# ---------------------------------------------------------------------------

def truncate_payload(
    data: Any,
    max_bytes: int = 1024 * 1024,
    truncation_key: str = "_truncated",
) -> Any:
    """
    Return *data* unchanged when it fits within *max_bytes*.
    For dicts, add a ``_truncated`` marker instead of dropping data silently.
    """
    try:
        serialised = json.dumps(data, default=str)
    except (TypeError, ValueError):
        return data

    if len(serialised.encode()) <= max_bytes:
        return data

    if isinstance(data, dict):
        return {
            truncation_key: True,
            "_original_size_bytes": len(serialised.encode()),
            **{k: v for k, v in list(data.items())[:5]},  # keep first 5 keys as preview
        }

    return {truncation_key: True, "_original_size_bytes": len(serialised.encode())}


# ---------------------------------------------------------------------------
# JSONPath helpers
# ---------------------------------------------------------------------------

def jsonpath_get(data: Dict[str, Any], path: str) -> List[Any]:
    """
    Return all values matching *path* in *data*.

    Strips optional-chaining syntax ``?.`` before parsing since
    jsonpath_ng does not support it.

    Returns an empty list on any error.
    """
    path = path.replace("?.", ".")
    try:
        from jsonpath_ng import parse
        expr = parse(path)
        return [m.value for m in expr.find(data)]
    except Exception:
        return []


def jsonpath_set(data: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    """
    Set *value* at all locations matching *path* inside *data* (mutates in place).

    Returns *data* for convenience.
    Strips optional-chaining syntax before parsing.
    """
    path = path.replace("?.", ".")
    try:
        from jsonpath_ng import parse
        expr = parse(path)
        expr.update(data, value)
    except Exception:
        pass
    return data


def merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-merge *override* into *base*.

    Values in *override* take precedence; nested dicts are merged
    recursively rather than replaced.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
