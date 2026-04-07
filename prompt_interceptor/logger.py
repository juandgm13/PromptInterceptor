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

from .config import get_config


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
        body: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a response, merging into the existing request log entry if present.

        Args:
            request_id: ID from matching request
            status_code: HTTP status code
            headers: Response headers
            body: Response body (JSON)
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
        files = sorted(date_dir.glob("req_*.json"))
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
