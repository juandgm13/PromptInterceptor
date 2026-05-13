"""
Traffic logging for PyProxy.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import hashlib
import shutil

import httpx

from .config import get_config


def _get_ollama_ps_info(model: str, target: str) -> Dict[str, Any]:
    """
    Query /api/ps for the running model's actual context size and GPU/CPU offload.

    Returns a dict with optional keys:
      context_size (int), offload ('gpu'|'partial'|'cpu'), size_vram (int), size_total (int)
    Returns {} on failure or when model is not loaded.
    """
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{target}/api/ps")
            if r.status_code != 200:
                return {}
            for m in r.json().get("models", []):
                name = m.get("name", "")
                # Match "llama3" == "llama3:latest", etc.
                if name != model and name.split(":")[0] != model.split(":")[0]:
                    continue
                result: Dict[str, Any] = {}
                # Context size: field name varies across Ollama versions
                for field in ("num_ctx", "context_length", "context"):
                    if m.get(field):
                        result["context_size"] = int(m[field])
                        break
                # GPU/CPU offload derived from size vs size_vram
                size_total = m.get("size", 0)
                size_vram = m.get("size_vram", 0)
                if size_total > 0:
                    result["size_total"] = size_total
                    result["size_vram"] = size_vram
                    ratio = size_vram / size_total
                    result["offload"] = "gpu" if ratio >= 0.99 else ("cpu" if size_vram == 0 else "partial")
                return result
    except Exception:
        pass
    return {}


class TrafficLogger:
    """Logger for capturing HTTP traffic."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.log_dir = Path(self.config.log_dir)
        self.log_size_limit = self.config.log_size_limit
        self.max_log_files = self.config.max_log_files
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Ensure log directory exists."""
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_date_dir(self) -> Path:
        """Get today's log directory."""
        today = datetime.now().strftime("%Y-%m-%d")
        date_dir = self.log_dir / today
        if not date_dir.exists():
            date_dir.mkdir(exist_ok=True)
        return date_dir

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        return hashlib.md5(
            f"{datetime.now().isoformat()}:{os.getpid()}".encode()
        ).hexdigest()[:12]

    def _truncate_body(self, body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Truncate large bodies for logging."""
        if body is None:
            return None

        body_str = json.dumps(body)
        if len(body_str) > self.log_size_limit:
            # Truncate large payloads
            return {
                **body,
                "_truncated": True,
                "_original_size": len(body_str),
            }
        return body

    def _get_headers_for_log(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Get headers for logging, excluding sensitive ones."""
        sensitive = {"authorization", "cookie", "set-cookie", "x-api-key"}
        return {
            k: v for k, v in headers.items()
            if k.lower() not in sensitive
        }

    def _extract_token_usage(
        self,
        response_body: Optional[Dict[str, Any]],
        model_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract and normalize token usage, querying Ollama /api/ps for actual context size."""
        if not response_body:
            return None

        prompt_tokens = None
        completion_tokens = None

        # Ollama native (/api/chat, /api/generate): prompt_eval_count + eval_count
        if "prompt_eval_count" in response_body:
            prompt_tokens = response_body["prompt_eval_count"]
            completion_tokens = response_body.get("eval_count", 0)
        # OpenAI-compat (/v1/chat/completions) and Anthropic (/v1/messages): usage object
        elif isinstance(response_body.get("usage"), dict):
            u = response_body["usage"]
            prompt_tokens = u.get("prompt_tokens") or u.get("input_tokens")
            completion_tokens = u.get("completion_tokens") or u.get("output_tokens")

        if prompt_tokens is None:
            return None

        total = (prompt_tokens or 0) + (completion_tokens or 0)

        # Query /api/ps for the loaded model's actual num_ctx and GPU/CPU offload
        ps_info: Dict[str, Any] = {}
        if model_name:
            ps_info = _get_ollama_ps_info(model_name, self.config.target)

        ctx = ps_info.get("context_size") or self.config.context_size
        # Use prompt_tokens as numerator: measures context fill *before* generation,
        # avoiding >100% artifacts from adding completion tokens to total.
        pct = round(((prompt_tokens or 0) / ctx) * 100, 1) if ctx else None

        result: Dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "context_size": ctx,
            "context_utilization_pct": pct,
        }
        if "offload" in ps_info:
            result["offload"] = ps_info["offload"]
            result["size_vram"] = ps_info.get("size_vram", 0)
            result["size_total"] = ps_info.get("size_total", 0)
        return result

    def log_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log a request.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: Request path
            headers: Request headers dict
            body: Request body (JSON)

        Returns:
            request_id: Unique ID for this request
        """
        request_id = self._generate_request_id()

        # Truncate large bodies
        safe_body = self._truncate_body(body)

        # Prepare headers for logging
        safe_headers = self._get_headers_for_log(headers)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_id": request_id,
            "type": "request",
            "method": method,
            "path": path,
            "headers": safe_headers,
            "body": safe_body,
            "proxy_target": self.config.target,
            "mode": self.config.mode,
        }

        # Write to log file
        self._write_log_file(log_entry)

        return request_id

    def log_response(
        self,
        request_id: str,
        status_code: int,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]] = None,
        correction_applied: Optional[str] = None,
        token_source_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log a response, merging into the existing request log entry if present.

        Args:
            request_id: ID from matching request
            status_code: HTTP status code
            headers: Response headers
            body: Response body (JSON)
            correction_applied: Description of auto-corrections applied, or None
            token_source_body: Alternative body to extract token usage from (e.g. the
                original Ollama response when body is an error dict like for 413s).
        """
        date_dir = self._get_date_dir()
        filepath = date_dir / f"req_{request_id}.json"

        # Start from existing request data so method/path/body are preserved
        existing: Dict[str, Any] = {}
        if filepath.exists():
            try:
                with open(filepath, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = {}

        log_entry = {
            **existing,
            "response_timestamp": datetime.now().isoformat(),
            "type": "response",
            "status_code": status_code,
            "response_headers": self._get_headers_for_log(headers),
            "response_body": self._truncate_body(body),
        }
        if correction_applied:
            log_entry["_correction_applied"] = correction_applied
        # Extract model name from response or request body for context size lookup
        model_name: Optional[str] = None
        token_src = token_source_body if token_source_body is not None else body
        if isinstance(token_src, dict):
            model_name = token_src.get("model")
        if not model_name and isinstance(existing.get("body"), dict):
            model_name = existing["body"].get("model")
        token_usage = self._extract_token_usage(token_src, model_name)
        if token_usage:
            log_entry["_tokens_usage"] = token_usage
        # Ensure request_id is always present
        log_entry["request_id"] = request_id

        self._write_log_file(log_entry)

    def log_raw_request(
        self,
        request_id: str,
        raw_body: str,
        raw_headers: Dict[str, str]
    ) -> None:
        """Log raw request/response before parsing."""
        body_obj = None
        try:
            body_obj = json.loads(raw_body) if raw_body else None
        except json.JSONDecodeError:
            pass

        self.log_request(
            method="UNKNOWN",
            path="/raw",
            headers=raw_headers,
            body=body_obj
        )
        self.log_response(
            request_id=request_id,
            status_code=200,
            headers={},
            body=None
        )

    def log_intercepted(self, request_id: str, modified: bool = True) -> None:
        """Log that a request was intercepted."""
        self.log_response(
            request_id=request_id,
            status_code=200 if modified else 0,
            headers={},
            body={"intercepted": True, "modified": modified}
        )

    def _write_log_file(self, entry: Dict[str, Any]) -> None:
        """Write log entry to file."""
        date_dir = self._get_date_dir()

        # Create filename with timestamp to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"req_{entry['request_id']}.json"

        filepath = date_dir / filename

        # Check if file exists and is within size limit
        if filepath.exists() and filepath.stat().st_size >= self.log_size_limit:
            # Rotate: delete oldest if at max files
            files = sorted(date_dir.glob("req_*.json"))
            if len(files) >= self.max_log_files:
                oldest = files[0]
                oldest.unlink()
                filepath = date_dir / f"req_{entry['request_id']}.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, default=str)

    def get_logs(self, limit: int = 10) -> list:
        """Get recent log entries."""
        date_dir = self._get_date_dir()
        files = sorted(date_dir.glob("req_*.json"), key=lambda f: f.stat().st_mtime)
        logs = []

        for f in files[-limit:]:
            try:
                with open(f) as fp:
                    logs.append(json.load(fp))
            except (json.JSONDecodeError, IOError):
                continue

        return logs

    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        date_dir = self._get_date_dir()
        if not date_dir.exists():
            return {"total_requests": 0, "total_responses": 0, "files": []}

        files = list(date_dir.glob("req_*.json"))
        return {
            "total_requests": len(files),
            "total_responses": len(files),  # Same as requests
            "files": [str(f) for f in files[-self.max_log_files:]]
        }

    def get_all_logs(self) -> list:
        """Get all logs from all directories."""
        import glob
        log_files = sorted(glob.glob(str(self.log_dir / "**" / "req_*.json"), recursive=True))
        all_logs = []

        for f in log_files:
            try:
                with open(f) as fp:
                    all_logs.append(json.load(fp))
            except (json.JSONDecodeError, IOError):
                continue

        return all_logs

    def clear_logs(self) -> int:
        """Delete all log files in today's directory. Returns number of files deleted."""
        date_dir = self._get_date_dir()
        if not date_dir.exists():
            return 0
        files = list(date_dir.glob("req_*.json"))
        for f in files:
            try:
                f.unlink()
            except OSError:
                pass
        return len(files)
