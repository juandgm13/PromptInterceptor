"""
PyProxy - Ollama Traffic Interceptor & Model Switcher

Main entry point for the proxy application.
"""

import time
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import get_config
from .logger import TrafficLogger
from .rules_engine import RuleEngine
from .proxy import (
    handle_chat_request, handle_generate_request,
    handle_stream_chat, handle_stream_generate,
)
from .health import check_target_health, get_status as _get_status
from .cors_middleware import add_cors_middleware


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    config = get_config()
    logger = TrafficLogger()
    rule_engine = RuleEngine(logger)

    app = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=config.debug,
    )

    add_cors_middleware(app)

    @app.middleware("http")
    async def log_timing(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        response.headers["x-proxy-time"] = f"{time.time() - start:.4f}s"
        return response

    @app.get("/health")
    async def health_check():
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "target": config.target,
                "mode": config.mode,
            },
        )

    @app.get("/status")
    async def status():
        return await _get_status()

    @app.get("/dashboard")
    async def dashboard():
        return JSONResponse(
            status_code=307,
            headers={"location": f"http://localhost:{config.dashboard_port}/"},
        )

    @app.get("/api/models")
    async def list_models():
        try:
            response = await check_target_health()
            import json
            return json.loads(response.body)
        except Exception as e:
            return {"error": str(e)}

    @app.post("/api/chat")
    async def chat_endpoint(request: Request):
        return await handle_chat_request(request, rule_engine, logger)

    @app.post("/api/generate")
    async def generate_endpoint(request: Request):
        return await handle_generate_request(request, rule_engine, logger)

    @app.post("/api/chat/stream")
    async def stream_chat_endpoint(request: Request):
        return await handle_stream_chat(request, rule_engine, logger)

    @app.post("/api/generate/stream")
    async def stream_generate_endpoint(request: Request):
        return await handle_stream_generate(request, rule_engine, logger)

    return app


# Module-level app so uvicorn can reference "ollama_proxy.main:app"
app = create_app()


def main() -> None:
    """Main entry point."""
    config = get_config()

    if config.dashboard_enabled:
        def _run_dashboard():
            uvicorn.run(
                "ollama_proxy.dashboard:app",
                host="0.0.0.0",
                port=config.dashboard_port,
                log_level="warning",
            )

        threading.Thread(target=_run_dashboard, daemon=True).start()

    uvicorn.run(
        "ollama_proxy.main:app",
        host=config.proxy_host,
        port=config.proxy_port,
        reload=config.debug,
        log_level="info",
    )


if __name__ == "__main__":
    main()
