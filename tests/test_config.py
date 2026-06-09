"""Tests for config.py."""

import json
from pathlib import Path

import pytest

from prompt_interceptor.config import (
    Config,
    load_config,
    save_config,
    get_config,
    get_global_config,
    reload_config,
    config_to_json,
)


# ---------------------------------------------------------------------------
# Config model
# ---------------------------------------------------------------------------

def test_config_defaults():
    c = Config()
    assert c.proxy_port == 8080
    assert c.target == "http://localhost:11434"
    assert c.mode == "passthrough"
    assert c.timeout == 120
    assert c.dashboard_port == 9090
    assert c.debug is False
    assert c.project_name == "PromptInterceptor"


def test_config_custom_values():
    c = Config(proxy_port=9000, mode="intercept", timeout=60)
    assert c.proxy_port == 9000
    assert c.mode == "intercept"
    assert c.timeout == 60


def test_config_rules_default_empty():
    assert Config().rules == []


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

def test_load_config_missing_file_returns_defaults():
    cfg = load_config(path=Path("/nonexistent/path/config.json"))
    assert cfg.proxy_port == 8080


def test_load_config_from_valid_file(tmp_path):
    data = {"proxy_port": 9999, "mode": "intercept", "timeout": 60}
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    cfg = load_config(path=p)
    assert cfg.proxy_port == 9999
    assert cfg.mode == "intercept"
    assert cfg.timeout == 60


def test_load_config_invalid_json_returns_defaults(tmp_path):
    p = tmp_path / "config.json"
    p.write_text("this is not json {{{")
    cfg = load_config(path=p)
    assert cfg.proxy_port == 8080


def test_load_config_with_rules(tmp_path):
    data = {
        "rules": [
            {"match": {"jsonpath": "$.model"}, "replace": {"jsonpath": "$.model", "value": "x"}}
        ]
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data))
    cfg = load_config(path=p)
    assert len(cfg.rules) == 1


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

def test_save_config_round_trip():
    """save_config writes valid JSON; restore config.json afterwards."""
    import prompt_interceptor.config as cfg_mod
    config_path = Path(cfg_mod.__file__).parent / "config.json"
    original = config_path.read_text()
    try:
        cfg = Config(proxy_port=7777)
        save_config(cfg)
        loaded = load_config()   # uses the same path
        assert loaded.proxy_port == 7777
    finally:
        config_path.write_text(original)


# ---------------------------------------------------------------------------
# get_config
# ---------------------------------------------------------------------------

def test_get_config_returns_config():
    cfg = get_config()
    assert isinstance(cfg, Config)


# ---------------------------------------------------------------------------
# get_global_config / reload_config
# ---------------------------------------------------------------------------

def test_get_global_config_caches(monkeypatch):
    import prompt_interceptor.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_config", None)
    first = get_global_config()
    second = get_global_config()
    assert first is second


def test_reload_config_clears_cache(monkeypatch):
    import prompt_interceptor.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "_config", None)
    first = get_global_config()
    reload_config()
    assert cfg_mod._config is not None
    # After reload, get_global_config returns newly loaded instance
    second = get_global_config()
    assert second is cfg_mod._config


# ---------------------------------------------------------------------------
# config_to_json
# ---------------------------------------------------------------------------

def test_config_to_json_is_dict():
    d = config_to_json(Config())
    assert isinstance(d, dict)
    assert d["proxy_port"] == 8080


def test_config_to_json_serialisable():
    import json as json_mod
    d = config_to_json(Config())
    dumped = json_mod.dumps(d)
    assert "proxy_port" in dumped


# ---------------------------------------------------------------------------
# load_config — OSError / PermissionError handling
# ---------------------------------------------------------------------------

def test_load_config_permission_error_returns_defaults(tmp_path, monkeypatch):
    """If the config file exists but cannot be opened (PermissionError), defaults are returned."""
    p = tmp_path / "config.json"
    p.write_text('{"proxy_port": 1234}')
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("denied")))
    cfg = load_config(path=p)
    assert cfg.proxy_port == 8080


def test_load_config_os_error_returns_defaults(tmp_path, monkeypatch):
    """If opening the config file raises OSError, defaults are returned."""
    p = tmp_path / "config.json"
    p.write_text('{"proxy_port": 5555}')
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("disk error")))
    cfg = load_config(path=p)
    assert cfg.proxy_port == 8080


# ---------------------------------------------------------------------------
# save_config — OSError / PermissionError handling
# ---------------------------------------------------------------------------

def test_save_config_permission_error_raises(monkeypatch):
    """save_config raises OSError with an informative message when write fails."""
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(PermissionError("read-only")))
    with pytest.raises(OSError, match="Failed to save configuration"):
        save_config(Config())


def test_save_config_os_error_raises(monkeypatch):
    """save_config re-raises OSError when the filesystem write fails."""
    monkeypatch.setattr("builtins.open", lambda *a, **kw: (_ for _ in ()).throw(OSError("no space left")))
    with pytest.raises(OSError):
        save_config(Config())


def test_debug_intercept_default_false():
    """debug_intercept defaults to False."""
    assert Config().debug_intercept is False


def test_debug_intercept_can_be_enabled():
    """debug_intercept can be set to True via config."""
    cfg = Config(debug_intercept=True)
    assert cfg.debug_intercept is True


def test_debug_intercept_loads_from_json(tmp_path):
    """debug_intercept is read from config.json when present."""
    p = tmp_path / "config.json"
    p.write_text('{"debug_intercept": true}')
    cfg = load_config(path=p)
    assert cfg.debug_intercept is True


def test_intercept_timeout_default():
    """intercept_timeout defaults to 30 seconds."""
    assert Config().intercept_timeout == 30.0


def test_intercept_timeout_configurable():
    """intercept_timeout can be overridden."""
    cfg = Config(intercept_timeout=120.0)
    assert cfg.intercept_timeout == 120.0


def test_intercept_timeout_loads_from_json(tmp_path):
    """intercept_timeout is read from config.json when present."""
    p = tmp_path / "config.json"
    p.write_text('{"intercept_timeout": 60.0}')
    cfg = load_config(path=p)
    assert cfg.intercept_timeout == 60.0
