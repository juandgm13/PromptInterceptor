"""
Traffic logging for PyProxy.
"""

import copy
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import uuid
import shutil

import httpx

from .config import get_config, effective_context_size

# Max in-flight request timers kept in memory (see TrafficLogger.start_timer).
_MAX_TIMERS = 512

# Base64 signatures of the image formats vision models accept. Ollama's native
# API sends bare base64 (no data: URI), so the MIME type must be sniffed.
_IMAGE_SIGNATURES = (
    ("iVBOR", "image/png"),
    ("/9j/", "image/jpeg"),
    ("R0lG", "image/gif"),
    ("UklG", "image/webp"),
    ("Qk", "image/bmp"),
)

_DATA_URI_RE = re.compile(r"^data:(image/[\w.+-]+);base64,(.*)$", re.DOTALL)


def _sniff_image_mime(data: str) -> str:
    for prefix, mime in _IMAGE_SIGNATURES:
        if data.startswith(prefix):
            return mime
    return "image/png"


def _iter_image_slots(body: Any):
    """Yield (container, key, kind, location) for every image in a request body.

    kind is "ollama" (bare base64 string in an images list), "openai" (image_url
    block) or "anthropic" (image block with a source). container[key] is the
    value to read or replace.
    """
    if not isinstance(body, dict):
        return
    if isinstance(body.get("images"), list):
        for j in range(len(body["images"])):
            yield body["images"], j, "ollama", "prompt"
    messages = body.get("messages")
    if not isinstance(messages, list):
        return
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        location = f"messages[{i}]"
        if isinstance(msg.get("images"), list):
            for j in range(len(msg["images"])):
                yield msg["images"], j, "ollama", location
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image_url":
                if isinstance(block.get("image_url"), dict):
                    yield block["image_url"], "url", "openai", location
                elif "image_url" in block:
                    yield block, "image_url", "openai", location
            elif block.get("type") == "image" and isinstance(block.get("source"), dict):
                yield block, "source", "anthropic", location


def extract_images(body: Any) -> List[Dict[str, Any]]:
    """Return every image in an Ollama, OpenAI or Anthropic request body.

    Each item is {"mime", "location"} plus either "data" (base64) or "url".
    """
    images: List[Dict[str, Any]] = []
    for container, key, kind, location in _iter_image_slots(body):
        value = container[key]
        if kind == "anthropic":
            src = value
            if src.get("type") == "base64" and isinstance(src.get("data"), str):
                images.append({
                    "mime": src.get("media_type") or _sniff_image_mime(src["data"]),
                    "data": src["data"],
                    "location": location,
                })
            elif src.get("type") == "url" and isinstance(src.get("url"), str):
                images.append({"mime": None, "url": src["url"], "location": location})
            continue
        if not isinstance(value, str):
            continue
        match = _DATA_URI_RE.match(value)
        if match:
            images.append({"mime": match.group(1), "data": match.group(2), "location": location})
        elif kind == "openai" or value.startswith(("http://", "https://")):
            images.append({"mime": None, "url": value, "location": location})
        else:
            images.append({"mime": _sniff_image_mime(value), "data": value, "location": location})
    return images


def _image_placeholder(n: int, data: str) -> str:
    # base64 encodes 3 bytes in 4 chars
    return f"<image #{n}, {max(1, len(data) * 3 // 4 // 1024)} KB>"


def strip_images(body: Any) -> Tuple[Any, int]:
    """Return (copy of body with base64 image payloads replaced by placeholders, image count).

    Image URLs are kept as-is. The original body is never mutated; it is only
    copied when it actually contains images.
    """
    if not extract_images(body):
        return body, 0
    stripped = copy.deepcopy(body)
    count = 0
    for container, key, kind, _location in _iter_image_slots(stripped):
        value = container[key]
        # Count exactly the slots extract_images reports, so "#n" matches its order.
        if kind == "anthropic":
            if value.get("type") == "base64" and isinstance(value.get("data"), str):
                count += 1
                value["data"] = _image_placeholder(count, value["data"])
            elif value.get("type") == "url" and isinstance(value.get("url"), str):
                count += 1
            continue
        if not isinstance(value, str):
            continue
        count += 1
        match = _DATA_URI_RE.match(value)
        if match:
            container[key] = _image_placeholder(count, match.group(2))
        elif kind == "ollama" and not value.startswith(("http://", "https://")):
            container[key] = _image_placeholder(count, value)
    return stripped, count


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
        # request_id -> monotonic timestamp taken just before forwarding to Ollama
        self._forward_start: Dict[str, float] = {}
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
        """Generate a unique request ID (12 lowercase hex chars).

        Random rather than derived from the clock: on Windows datetime.now() can
        return the same value for requests in the same tick, which made two
        requests share an ID and overwrite each other's log file.
        """
        return uuid.uuid4().hex[:12]

    def start_timer(self, request_id: Optional[str]) -> None:
        """
        Mark the start of the forward to Ollama for this request.

        Called by the proxy handlers right before the outbound call, so the
        measured duration excludes the (potentially unbounded) wait for a
        dashboard decision in intercept mode. log_response later reads the
        mark to compute _duration_ms.
        """
        if not request_id:
            return
        # Bound the dict: requests that error out before log_response never
        # clear their mark, so drop the oldest entry once the cap is reached.
        while len(self._forward_start) >= _MAX_TIMERS:
            del self._forward_start[next(iter(self._forward_start))]
        self._forward_start[request_id] = time.monotonic()

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

        ctx = ps_info.get("context_size") or effective_context_size(self.config)
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
        started = self._forward_start.get(request_id)
        if started is not None:
            log_entry["_duration_ms"] = round((time.monotonic() - started) * 1000)
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

    def update_request_body(
        self,
        request_id: str,
        new_body: Optional[Dict[str, Any]],
        rule_applied: Optional[Dict[str, Any]] = None,
        intercepted_modified: bool = False,
        intercepted_forwarded: bool = False,
        num_ctx_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the body of an existing request log entry without changing its type to 'response'.

        Used when rules or intercept edits modify the body that is actually sent to Ollama,
        so the log reflects what was forwarded rather than the original request.
        """
        date_dir = self._get_date_dir()
        filepath = date_dir / f"req_{request_id}.json"

        existing: Dict[str, Any] = {}
        if filepath.exists():
            try:
                with open(filepath, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing = {}

        existing["body"] = self._truncate_body(new_body)
        if rule_applied:
            rules_list = existing.get("_rules_applied", [])
            rules_list.append(rule_applied)
            existing["_rules_applied"] = rules_list
        if intercepted_modified:
            existing["_intercepted_modified"] = True
        if intercepted_forwarded:
            existing["_intercepted_forwarded"] = True
        if num_ctx_override:
            # {"client": <num_ctx the client asked for, or None>, "sent": <num_ctx forwarded>}
            existing["_num_ctx_override"] = num_ctx_override

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, default=str)

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

    def get_log(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Return a single log entry by request id, or None if missing or malformed."""
        if not re.fullmatch(r"[0-9a-f]{12}", request_id or ""):
            return None
        filepath = self._get_date_dir() / f"req_{request_id}.json"
        try:
            with open(filepath, encoding="utf-8") as fp:
                return json.load(fp)
        except (OSError, json.JSONDecodeError):
            return None

    def delete_log(self, request_id: str) -> bool:
        """
        Delete a single log entry by request id.

        Returns True if a file was removed, False otherwise.
        """
        # request_id reaches us straight from a URL path and is used to build a
        # filesystem path, so reject anything that is not the exact shape
        # _generate_request_id produces (12 hex chars).
        if not re.fullmatch(r"[0-9a-f]{12}", request_id or ""):
            return False
        filepath = self._get_date_dir() / f"req_{request_id}.json"
        try:
            filepath.unlink()
        except OSError:
            return False
        return True

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
