"""Tests for rules_engine.py."""

import logging
import pytest
from unittest.mock import MagicMock
from prompt_interceptor.config import Config
from prompt_interceptor.logger import TrafficLogger
from prompt_interceptor.rules_engine import Rule, RuleEngine, _jsonpath_get, _jsonpath_set


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def intercept_cfg(tmp_path):
    return Config(
        log_dir=str(tmp_path / "logs"),
        mode="intercept",
        rules=[
            {
                "match": {
                    "path": "/api/chat",
                    "jsonpath": "$.model",
                    "value": ["llama3"],
                },
                "replace": {"jsonpath": "$.model", "value": "deepseek-coder"},
            }
        ],
    )


@pytest.fixture
def passthrough_cfg(tmp_path):
    return Config(log_dir=str(tmp_path / "logs"), mode="passthrough", rules=[])


@pytest.fixture
def engine(intercept_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    logger = TrafficLogger(intercept_cfg)
    return RuleEngine(logger)


@pytest.fixture
def passthrough_engine(passthrough_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: passthrough_cfg)
    logger = TrafficLogger(passthrough_cfg)
    return RuleEngine(logger)


# ---------------------------------------------------------------------------
# Rule dataclass
# ---------------------------------------------------------------------------

def test_rule_normalises_value_to_list():
    r = Rule(match={"value": "llama3"}, replace={})
    assert r.match["value"] == ["llama3"]


def test_rule_keeps_existing_list():
    r = Rule(match={"value": ["a", "b"]}, replace={})
    assert r.match["value"] == ["a", "b"]


# ---------------------------------------------------------------------------
# _load_rules
# ---------------------------------------------------------------------------

def test_load_rules_from_config(engine):
    assert len(engine.rules) == 1
    assert engine.rules[0].replace["value"] == "deepseek-coder"


def test_load_rules_empty_when_no_rules(passthrough_engine):
    assert engine is not passthrough_engine
    assert len(passthrough_engine.rules) == 0


# ---------------------------------------------------------------------------
# _evaluate_match
# ---------------------------------------------------------------------------

def test_evaluate_match_path_filter_no_match(engine):
    rule = engine.rules[0]
    assert engine._evaluate_match(rule, {"model": "llama3"}, "/api/generate") is False


def test_evaluate_match_path_filter_match(engine):
    rule = engine.rules[0]
    assert engine._evaluate_match(rule, {"model": "llama3"}, "/api/chat") is True


def test_evaluate_match_jsonpath_not_found(engine):
    rule = Rule(match={"jsonpath": "$.model"}, replace={})
    assert engine._evaluate_match(rule, {"other": "x"}, "/") is False


def test_evaluate_match_value_filter_mismatch(engine):
    rule = Rule(
        match={"jsonpath": "$.model", "value": ["mistral"]},
        replace={},
    )
    assert engine._evaluate_match(rule, {"model": "llama3"}, "/") is False


def test_evaluate_match_value_filter_match(engine):
    rule = Rule(
        match={"jsonpath": "$.model", "value": ["llama3", "mistral"]},
        replace={},
    )
    assert engine._evaluate_match(rule, {"model": "mistral"}, "/") is True


def test_evaluate_match_no_jsonpath_returns_true(engine):
    rule = Rule(match={"path": "/api/chat"}, replace={})
    assert engine._evaluate_match(rule, {}, "/api/chat") is True


# ---------------------------------------------------------------------------
# _apply_replacement
# ---------------------------------------------------------------------------

def test_apply_replacement_modifies_model(engine):
    rule = engine.rules[0]
    data = {"model": "llama3", "messages": []}
    engine._apply_replacement(rule, data)
    assert data["model"] == "deepseek-coder"


def test_apply_replacement_nested(engine):
    rule = Rule(
        match={},
        replace={"jsonpath": "$.options.temperature", "value": 0.1},
    )
    data = {"options": {"temperature": 0.9}}
    engine._apply_replacement(rule, data)
    assert data["options"]["temperature"] == 0.1


def test_apply_replacement_missing_jsonpath_no_crash(engine):
    rule = Rule(match={}, replace={"value": "x"})  # no jsonpath
    data = {"model": "llama3"}
    engine._apply_replacement(rule, data)  # should not raise


# ---------------------------------------------------------------------------
# process_request
# ---------------------------------------------------------------------------

async def test_process_request_passthrough_returns_original(passthrough_engine, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: passthrough_engine.logger.config)
    body = {"model": "llama3"}
    modified, result, rid = await passthrough_engine.process_request("POST", "/api/chat", {}, body)
    assert modified is False  # passthrough_engine has rules=[], so nothing changes
    assert result is body
    assert isinstance(rid, str)


async def test_process_request_intercept_rule_matches(engine, intercept_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    body = {"model": "llama3", "messages": []}
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, body)
    assert modified is True
    assert result["model"] == "deepseek-coder"


async def test_process_request_no_rule_match(engine, intercept_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    body = {"model": "gemma", "messages": []}
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, body)
    assert modified is False
    assert result is body


async def test_process_request_none_body(engine, intercept_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, None)
    assert isinstance(rid, str)


# ---------------------------------------------------------------------------
# process_response
# ---------------------------------------------------------------------------

async def test_process_response_passthrough(passthrough_engine, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: passthrough_engine.logger.config)
    body = {"model": "llama3"}
    modified, result = await passthrough_engine.process_response("rid", "/", {}, body)
    assert modified is False
    assert result is body


async def test_process_response_intercept(engine, intercept_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    # Rules apply to the path /api/chat with model=llama3
    body = {"model": "llama3"}
    modified, result = await engine.process_response("rid", "/api/chat", {}, body)
    # model rule should fire
    assert modified is True
    assert result["model"] == "deepseek-coder"


# ---------------------------------------------------------------------------
# enable_rule / disable_rule / add_rule / get_rules / reload_rules
# ---------------------------------------------------------------------------

def test_enable_disable_rule(engine):
    assert engine.disable_rule(0) is True
    assert engine.rules[0].enabled is False
    assert engine.enable_rule(0) is True
    assert engine.rules[0].enabled is True


def test_enable_invalid_index(engine):
    assert engine.enable_rule(99) is False


def test_disable_invalid_index(engine):
    assert engine.disable_rule(-1) is False


def test_add_rule(engine):
    before = len(engine.rules)
    ok = engine.add_rule({
        "match": {"jsonpath": "$.prompt"},
        "replace": {"jsonpath": "$.prompt", "value": "hi"},
        "enabled": True,
    })
    assert ok is True
    assert len(engine.rules) == before + 1


def test_add_rule_invalid_missing_keys(engine, caplog):
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        result = engine.add_rule({"nope": True})
    assert result is False
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_get_rules_returns_list(engine):
    rules = engine.get_rules()
    assert isinstance(rules, list)
    assert len(rules) == 1
    assert "match" in rules[0]
    assert "replace" in rules[0]


def test_reload_rules(engine, intercept_cfg, monkeypatch):
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    engine.reload_rules()
    assert len(engine.rules) == 1


# ---------------------------------------------------------------------------
# _jsonpath_get exception path  (lines 38-39)
# ---------------------------------------------------------------------------

def test_jsonpath_get_invalid_expression_returns_empty(caplog):
    """Malformed JSONPath triggers the except clause → returns []."""
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        result = _jsonpath_get({"model": "x"}, "$$$$invalid")
    assert result == []


def test_jsonpath_get_invalid_via_evaluate_match(engine, caplog):
    """_evaluate_match with invalid jsonpath falls through to False."""
    rule = Rule(match={"jsonpath": "$$$$invalid"}, replace={})
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        result = engine._evaluate_match(rule, {"model": "llama3"}, "/")
    assert result is False


# ---------------------------------------------------------------------------
# _jsonpath_set exception path  (lines 50-51)
# ---------------------------------------------------------------------------

def test_jsonpath_set_invalid_expression_no_crash(caplog):
    """Malformed JSONPath in _jsonpath_set triggers except → data unchanged."""
    data = {"model": "llama3"}
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        result = _jsonpath_set(data, "$$$$invalid", "new")
    assert result is data
    assert data["model"] == "llama3"


def test_apply_replacement_invalid_jsonpath_no_crash(engine, caplog):
    """_apply_replacement with bad replace path does not raise."""
    rule = Rule(match={}, replace={"jsonpath": "$$$$invalid", "value": "x"})
    data = {"model": "llama3"}
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        engine._apply_replacement(rule, data)
    assert data["model"] == "llama3"


# ---------------------------------------------------------------------------
# Disabled rule skip in process_request  (line 136)
# ---------------------------------------------------------------------------

async def test_process_request_disabled_rule_skipped(engine, intercept_cfg, monkeypatch):
    """A disabled rule is skipped even if the body would match."""
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)

    engine.rules[0].enabled = False
    body = {"model": "llama3", "messages": []}
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, body)
    assert modified is False
    assert result is body


async def test_process_request_mix_disabled_enabled(intercept_cfg, monkeypatch):
    """First rule disabled, second enabled — only second applies."""
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    logger = TrafficLogger(intercept_cfg)
    eng = RuleEngine(logger)

    eng.add_rule({
        "match": {"path": "/api/chat", "jsonpath": "$.model", "value": ["llama3"]},
        "replace": {"jsonpath": "$.model", "value": "qwen"},
    })
    eng.rules[0].enabled = False

    body = {"model": "llama3", "messages": []}
    modified, result, rid = await eng.process_request("POST", "/api/chat", {}, body)
    assert modified is True
    assert result["model"] == "qwen"


# ---------------------------------------------------------------------------
# Exception in rule application in process_request  (lines 154-155)
# ---------------------------------------------------------------------------

async def test_process_request_rule_exception_logged(engine, intercept_cfg, monkeypatch):
    """When _apply_replacement raises, the exception is logged and processing continues."""
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)

    monkeypatch.setattr(engine, "_apply_replacement", MagicMock(side_effect=RuntimeError("boom")))

    body = {"model": "llama3", "messages": []}
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, body)
    assert isinstance(rid, str)


# ---------------------------------------------------------------------------
# Disabled rule skip in process_response  (line 185)
# ---------------------------------------------------------------------------

async def test_process_response_disabled_rule_skipped(engine, intercept_cfg, monkeypatch):
    """A disabled rule is skipped in process_response."""
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)

    engine.rules[0].enabled = False
    body = {"model": "llama3"}
    modified, result = await engine.process_response("rid", "/api/chat", {}, body)
    assert modified is False


# ---------------------------------------------------------------------------
# Exception in rule application in process_response  (lines 203-204)
# ---------------------------------------------------------------------------

async def test_process_response_rule_exception_logged(engine, intercept_cfg, monkeypatch):
    """When _apply_replacement raises in process_response, it logs and continues."""
    import prompt_interceptor.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)

    monkeypatch.setattr(engine, "_apply_replacement", MagicMock(side_effect=RuntimeError("crash")))

    body = {"model": "llama3"}
    modified, result = await engine.process_response("rid", "/api/chat", {}, body)
    assert isinstance(modified, bool)


# ---------------------------------------------------------------------------
# _jsonpath_get and _jsonpath_set emit warnings on exception
# ---------------------------------------------------------------------------

def test_jsonpath_get_invalid_expression_logs_warning(caplog):
    """_jsonpath_get emits a WARNING log when the expression is invalid."""
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        result = _jsonpath_get({"model": "llama3"}, "$$$$invalid")

    assert result == []
    assert any("$$$$invalid" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_jsonpath_set_invalid_expression_logs_warning(caplog):
    """_jsonpath_set emits a WARNING log when the expression is invalid."""
    data = {"model": "llama3"}
    with caplog.at_level(logging.WARNING, logger="prompt_interceptor.rules_engine"):
        result = _jsonpath_set(data, "$$$$invalid", "new_value")

    assert result is data
    assert data["model"] == "llama3"
    assert any("$$$$invalid" in r.message for r in caplog.records)
    assert any(r.levelno == logging.WARNING for r in caplog.records)
