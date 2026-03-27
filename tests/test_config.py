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
    assert c.project_name == "PyProxy"


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
