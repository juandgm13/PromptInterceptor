"""Tests for rules_engine.py."""

import pytest
from ollama_proxy.config import Config
from ollama_proxy.logger import TrafficLogger
from ollama_proxy.rules_engine import Rule, RuleEngine


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
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    logger = TrafficLogger(intercept_cfg)
    return RuleEngine(logger)


@pytest.fixture
def passthrough_engine(passthrough_cfg, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
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
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: passthrough_engine.logger.config)
    body = {"model": "llama3"}
    modified, result, rid = await passthrough_engine.process_request("POST", "/api/chat", {}, body)
    assert modified is False
    assert result is body
    assert isinstance(rid, str)


async def test_process_request_intercept_rule_matches(engine, intercept_cfg, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    body = {"model": "llama3", "messages": []}
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, body)
    assert modified is True
    assert result["model"] == "deepseek-coder"


async def test_process_request_no_rule_match(engine, intercept_cfg, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    body = {"model": "gemma", "messages": []}
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, body)
    assert modified is False
    assert result is body


async def test_process_request_none_body(engine, intercept_cfg, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    modified, result, rid = await engine.process_request("POST", "/api/chat", {}, None)
    assert isinstance(rid, str)


# ---------------------------------------------------------------------------
# process_response
# ---------------------------------------------------------------------------

async def test_process_response_passthrough(passthrough_engine, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: passthrough_engine.logger.config)
    body = {"model": "llama3"}
    modified, result = await passthrough_engine.process_response("rid", "/", {}, body)
    assert modified is False
    assert result is body


async def test_process_response_intercept(engine, intercept_cfg, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
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


def test_add_rule_invalid_missing_keys(engine):
    assert engine.add_rule({"nope": True}) is False


def test_get_rules_returns_list(engine):
    rules = engine.get_rules()
    assert isinstance(rules, list)
    assert len(rules) == 1
    assert "match" in rules[0]
    assert "replace" in rules[0]


def test_reload_rules(engine, intercept_cfg, monkeypatch):
    import ollama_proxy.rules_engine as re_mod
    monkeypatch.setattr(re_mod, "get_config", lambda: intercept_cfg)
    engine.reload_rules()
    assert len(engine.rules) == 1
