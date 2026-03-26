"""
Rule engine for PyProxy.

Handles rule-based request/response modifications using JSONPath expressions.
"""

import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

from .config import get_config
from .logger import TrafficLogger


@dataclass
class Rule:
    """A single rule for matching and replacing."""

    match: Dict[str, Any]
    replace: Dict[str, Any]
    enabled: bool = True

    def __post_init__(self):
        # Ensure lists are valid
        if "value" in self.match:
            self.match["value"] = self.match["value"] if isinstance(self.match["value"], list) else [self.match["value"]]


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
                enabled=rule.get("enabled", True)
            )
            for rule in config.get("rules", [])
        ]

    def reload_rules(self) -> None:
        """Reload rules from config."""
        self._load_rules()
        self.logger.log_response(
            request_id="reload",
            status_code=200,
            headers={},
            body={"action": "reload_rules", "rule_count": len(self.rules)}
        )

    def _evaluate_match(self, rule: Rule, data: Dict[str, Any], path: str) -> bool:
        """
        Check if a rule matches the given data.

        Args:
            rule: Rule to evaluate
            data: Request/response data
            path: Request path

        Returns:
            True if rule matches
        """
        match_config = rule.match

        # Check path filter
        if "path" in match_config:
            if match_config["path"] != path:
                return False

        # Check jsonpath filter
        if "jsonpath" in match_config:
            jsonpath = match_config["jsonpath"]
            value_filter = match_config.get("value")

            try:
                from jsonpath_ng import parse
                jsonpath_expr = parse(jsonpath)
                matches = jsonpath_expr.find(data)

                if not matches:
                    return False

                # Get matched values
                matched_values = [str(m.value) for m in matches]

                # If value filter is provided, check if any matched value is in the list
                if "value" in match_config:
                    filter_values = match_config["value"]
                    if isinstance(filter_values, str):
                        filter_values = [filter_values]
                    if not any(v in filter_values for v in matched_values):
                        return False

            except ImportError:
                # jsonpath-ng not installed, skip jsonpath filtering
                pass

        return True

    def _apply_replacement(self, rule: Rule, data: Dict[str, Any], path: str) -> Dict[str, Any]:
        """
        Apply a replacement to the data.

        Args:
            rule: Rule with replacement info
            data: Data to modify
            path: Request path

        Returns:
            Modified data
        """
        replace_config = rule.replace

        # Build JSONPath expression for replacement
        try:
            from jsonpath_ng import parse, set

            # Construct the set expression
            expr_parts = []
            for key in replace_config:
                if key == "jsonpath":
                    continue
                expr_parts.append(f"{'.' if '.' not in key else ''}{key}")

            expr_str = f"{replace_config['jsonpath']}={expr_parts}" if expr_parts else str(replace_config)

            # Parse the set expression
            set_expr = parse(expr_str)

            # Apply the set
            set_expr.find_and_replace(data, set_value=replace_config["value"])

            return data

        except (ImportError, KeyError, TypeError):
            # Handle simple key replacement
            if isinstance(replace_config, dict):
                for key, value in replace_config.items():
                    if key == "jsonpath":
                        continue
                    # Navigate to the key and set the value
                    keys = replace_config["jsonpath"].split(".")
                    current = data
                    for k in keys:
                        current = current.get(k, {})
                    current[key] = value

            return data

    async def process_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None
    ) -> Optional[tuple]:
        """
        Process a request through rules.

        Returns:
            (modified_data, rules_applied, reason) or None
        """
        # Skip in passthrough mode
        if get_config().mode == "passthrough":
            return None

        # Log request
        request_id = self.logger.log_request(method, path, headers, body)

        modified = False
        rule_applied = None

        for rule in self.rules:
            if not rule.enabled:
                continue

            if self._evaluate_match(rule, body or {}, path):
                try:
                    # Deep copy data to modify
                    import copy
                    new_body = copy.deepcopy(body) if body else {}

                    if new_body:
                        new_body = self._apply_replacement(rule, new_body, path)

                    if new_body != body:
                        modified = True
                        rule_applied = rule.replace["jsonpath"]
                        self.logger.log_response(
                            request_id=request_id,
                            status_code=200,
                            headers={},
                            body={
                                "action": "rule_applied",
                                "path": rule.replace["jsonpath"],
                                "new_value": new_body.get(rule.replace["jsonpath"], rule.replace.get("value", "N/A")),
                                "rule": rule.replace
                            }
                        )

                except Exception as e:
                    self.logger.log_response(
                        request_id=request_id,
                        status_code=500,
                        headers={},
                        body={"action": "rule_error", "error": str(e)}
                    )
                    raise

        return (modified, body, request_id)

    async def process_response(
        self,
        request_id: str,
        path: str,
        headers: Dict[str, Any],
        body: Optional[Dict[str, Any]] = None
    ) -> Optional[tuple]:
        """
        Process a response through rules.

        Returns:
            (modified_data, rules_applied) or None
        """
        # Skip in passthrough mode
        if get_config().mode == "passthrough":
            return None

        modified = False
        rule_applied = None

        for rule in self.rules:
            if not rule.enabled:
                continue

            if self._evaluate_match(rule, body or {}, path):
                try:
                    import copy
                    new_body = copy.deepcopy(body) if body else {}

                    if new_body:
                        new_body = self._apply_replacement(rule, new_body, path)

                        if new_body != body:
                            modified = True
                            rule_applied = rule.replace["jsonpath"]
                            self.logger.log_response(
                                request_id=request_id,
                                status_code=200,
                                headers={},
                                body={
                                    "action": "response_rule_applied",
                                    "path": rule.replace["jsonpath"],
                                    "new_value": new_body.get(rule.replace["jsonpath"], rule.replace.get("value", "N/A")),
                                    "rule": rule.replace
                                }
                            )

                except Exception as e:
                    self.logger.log_response(
                        request_id=request_id,
                        status_code=500,
                        headers={},
                        body={"action": "response_rule_error", "error": str(e)}
                    )
                    raise

        return (modified, body)

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
                enabled=rule_dict.get("enabled", True)
            )
            self.rules.append(rule)
            return True
        except (KeyError, TypeError):
            return False
