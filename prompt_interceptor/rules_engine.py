"""
Rule engine for PromptInterceptor.

Handles rule-based request/response modifications using JSONPath expressions.
"""

import copy
import functools
import json
import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from .config import get_config
from .logger import TrafficLogger

_log = logging.getLogger(__name__)


@functools.lru_cache(maxsize=256)
def _compile_jsonpath(jsonpath: str):
    """Compile and cache a JSONPath expression to avoid repeated parsing."""
    from jsonpath_ng import parse
    return parse(jsonpath)


@dataclass
class Rule:
    """A single rule for matching and replacing."""

    match: Dict[str, Any]
    replace: Dict[str, Any]
    enabled: bool = True

    def __post_init__(self):
        # Normalise value filter to list
        if "value" in self.match and not isinstance(self.match["value"], list):
            self.match["value"] = [self.match["value"]]


def _jsonpath_get(data: Dict[str, Any], jsonpath: str) -> list:
    """Return all values matching a JSONPath expression."""
    # Strip optional-chaining syntax (not valid JSONPath)
    jsonpath = jsonpath.replace("?.", ".")
    try:
        expr = _compile_jsonpath(jsonpath)
        return [m.value for m in expr.find(data)]
    except Exception as exc:
        _log.warning("JSONPath expression failed (get): %r — %s", jsonpath, exc)
        return []


def _jsonpath_set(data: Dict[str, Any], jsonpath: str, value: Any) -> Dict[str, Any]:
    """Set a value at a JSONPath location, returning the (mutated) data."""
    # Strip optional-chaining syntax
    jsonpath = jsonpath.replace("?.", ".")
    try:
        expr = _compile_jsonpath(jsonpath)
        expr.update(data, value)
    except Exception as exc:
        _log.warning("JSONPath expression failed (set): %r — %s", jsonpath, exc)
    return data


class RuleEngine:
    """Engine for processing rules against requests/responses."""

    def __init__(self, logger: Optional[TrafficLogger] = None):
        self.logger = logger or TrafficLogger()
        self.rules: List[Rule] = []
        self._load_rules()

    def _load_rules(self) -> None:
        """Load rules from configuration."""
        config = get_config()
        self.rules = [
            Rule(
                match=rule["match"],
                replace=rule["replace"],
                enabled=rule.get("enabled", True),
            )
            for rule in config.rules
        ]

    def reload_rules(self) -> None:
        """Reload rules from config."""
        self._load_rules()

    def _evaluate_match(self, rule: Rule, data: Dict[str, Any], path: str) -> bool:
        """
        Check if a rule matches the given data.

        Returns True if rule matches.
        """
        match_config = rule.match

        # Check path filter
        if "path" in match_config and match_config["path"] != path:
            return False

        # Check jsonpath filter
        if "jsonpath" in match_config:
            matched_values = _jsonpath_get(data, match_config["jsonpath"])
            if not matched_values:
                return False

            # If value filter provided, at least one matched value must be in the list
            if "value" in match_config:
                filter_values = match_config["value"]
                if not any(str(v) in [str(f) for f in filter_values] for v in matched_values):
                    return False

        return True

    def _apply_replacement(self, rule: Rule, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply replacement to data. Returns mutated copy."""
        replace_config = rule.replace
        if "jsonpath" not in replace_config or "value" not in replace_config:
            return data
        return _jsonpath_set(data, replace_config["jsonpath"], replace_config["value"])

    async def process_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        Process a request through rules.

        Returns (modified, body, request_id).
        Always returns a 3-tuple — callers must not check for None.
        """
        request_id = self.logger.log_request(method, path, headers, body)

        config = get_config()
        if config.mode == "passthrough":
            return (False, body, request_id)

        current_body = copy.deepcopy(body) if body else {}
        modified = False

        for rule in self.rules:
            if not rule.enabled:
                continue

            if self._evaluate_match(rule, current_body, path):
                try:
                    before = copy.deepcopy(current_body)
                    self._apply_replacement(rule, current_body)
                    if current_body != before:
                        modified = True
                        self.logger.log_response(
                            request_id=request_id,
                            status_code=200,
                            headers={},
                            body={
                                "action": "rule_applied",
                                "jsonpath": rule.replace.get("jsonpath"),
                                "new_value": rule.replace.get("value"),
                            },
                        )
                except Exception as e:
                    self.logger.log_response(
                        request_id=request_id,
                        status_code=500,
                        headers={},
                        body={"action": "rule_error", "error": str(e)},
                    )

        return (modified, current_body if modified else body, request_id)

    async def process_response(
        self,
        request_id: str,
        path: str,
        headers: Dict[str, Any],
        body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Process a response through rules.

        Returns (modified, body).
        """
        config = get_config()
        if config.mode == "passthrough":
            return (False, body)

        current_body = copy.deepcopy(body) if body else {}
        modified = False

        for rule in self.rules:
            if not rule.enabled:
                continue

            if self._evaluate_match(rule, current_body, path):
                try:
                    before = copy.deepcopy(current_body)
                    self._apply_replacement(rule, current_body)
                    if current_body != before:
                        modified = True
                        self.logger.log_response(
                            request_id=request_id,
                            status_code=200,
                            headers={},
                            body={
                                "action": "response_rule_applied",
                                "jsonpath": rule.replace.get("jsonpath"),
                                "new_value": rule.replace.get("value"),
                            },
                        )
                except Exception as e:
                    self.logger.log_response(
                        request_id=request_id,
                        status_code=500,
                        headers={},
                        body={"action": "response_rule_error", "error": str(e)},
                    )

        return (modified, current_body if modified else body)

    def get_rules(self) -> List[Dict[str, Any]]:
        """Get current rules."""
        return [rule.__dict__ for rule in self.rules]

    def enable_rule(self, index: int) -> bool:
        """Enable a rule by index."""
        if 0 <= index < len(self.rules):
            self.rules[index].enabled = True
            return True
        return False

    def disable_rule(self, index: int) -> bool:
        """Disable a rule by index."""
        if 0 <= index < len(self.rules):
            self.rules[index].enabled = False
            return True
        return False

    def add_rule(self, rule_dict: Dict[str, Any]) -> bool:
        """Add a new rule."""
        try:
            rule = Rule(
                match=rule_dict["match"],
                replace=rule_dict["replace"],
                enabled=rule_dict.get("enabled", True),
            )
            self.rules.append(rule)
            return True
        except (KeyError, TypeError) as exc:
            _log.warning("Failed to add rule — missing or invalid keys: %s", exc)
            return False

    def delete_rule(self, index: int) -> bool:
        """Delete a rule by index."""
        if 0 <= index < len(self.rules):
            self.rules.pop(index)
            return True
        return False
