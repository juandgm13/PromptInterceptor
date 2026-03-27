"""
Interceptor for PyProxy.

In intercept mode the proxy pauses each request, stores it in a queue,
and waits for a decision from the dashboard (or CLI):

  - forward  – send the original request unchanged
  - edit     – send a modified request body provided by the caller
  - drop     – respond with 204 No Content immediately

The Interceptor is a module-level singleton so both the proxy handlers
and the dashboard API share the same in-flight request map.
"""

import asyncio
import copy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Per-request state machine
# ---------------------------------------------------------------------------

class InterceptedRequest:
    """Holds a paused request and its resolution event."""

    def __init__(
        self,
        request_id: str,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]],
    ) -> None:
        self.request_id = request_id
        self.method = method
        self.path = path
        self.headers = headers
        self.body = copy.deepcopy(body)
        self.created_at = datetime.now(timezone.utc).isoformat()

        # Set by the resolver (forward / edit / drop)
        self._action: str = "forward"
        self._edited_body: Optional[Dict[str, Any]] = None
        self._event: asyncio.Event = asyncio.Event()

    # ------------------------------------------------------------------
    # Resolution methods – called by dashboard API or CLI
    # ------------------------------------------------------------------

    def forward(self) -> None:
        """Forward the request unchanged."""
        self._action = "forward"
        self._event.set()

    def edit(self, new_body: Dict[str, Any]) -> None:
        """Forward the request with a modified body."""
        self._action = "edit"
        self._edited_body = new_body
        self._event.set()

    def drop(self) -> None:
        """Drop the request (return 204 to the client)."""
        self._action = "drop"
        self._event.set()

    # ------------------------------------------------------------------
    # Awaitable by the proxy handler
    # ------------------------------------------------------------------

    async def wait(self, timeout: float = 30.0) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Block until a decision is made or *timeout* seconds elapse.

        Returns:
            (action, body) where action is 'forward', 'edit', or 'drop',
            and body is the (possibly modified) request body.
        """
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Auto-forward on timeout so the client isn't left hanging
            self._action = "forward"

        final_body = self._edited_body if self._action == "edit" else self.body
        return self._action, final_body

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "headers": self.headers,
            "body": self.body,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Singleton interceptor
# ---------------------------------------------------------------------------

class Interceptor:
    """
    Manages in-flight intercepted requests.

    One global instance is shared between the proxy and the dashboard.
    """

    def __init__(self, intercept_timeout: float = 30.0) -> None:
        self._pending: Dict[str, InterceptedRequest] = {}
        self.intercept_timeout = intercept_timeout

    # ------------------------------------------------------------------
    # Called by proxy handlers
    # ------------------------------------------------------------------

    async def intercept(
        self,
        request_id: str,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Optional[Dict[str, Any]],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Pause the request and wait for a resolution.

        Returns:
            (action, body) – see InterceptedRequest.wait()
        """
        req = InterceptedRequest(request_id, method, path, headers, body)
        self._pending[request_id] = req
        try:
            return await req.wait(timeout=self.intercept_timeout)
        finally:
            self._pending.pop(request_id, None)

    # ------------------------------------------------------------------
    # Called by dashboard / CLI
    # ------------------------------------------------------------------

    def get_pending(self) -> list:
        """Return a list of all pending intercepted requests."""
        return [r.to_dict() for r in self._pending.values()]

    def resolve_forward(self, request_id: str) -> bool:
        """Forward a pending request unchanged."""
        req = self._pending.get(request_id)
        if req:
            req.forward()
            return True
        return False

    def resolve_edit(self, request_id: str, new_body: Dict[str, Any]) -> bool:
        """Forward a pending request with *new_body*."""
        req = self._pending.get(request_id)
        if req:
            req.edit(new_body)
            return True
        return False

    def resolve_drop(self, request_id: str) -> bool:
        """Drop a pending request."""
        req = self._pending.get(request_id)
        if req:
            req.drop()
            return True
        return False

    def __len__(self) -> int:
        return len(self._pending)


# Module-level singleton shared across all imports
interceptor = Interceptor()
