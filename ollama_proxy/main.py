"""
PyProxy - Ollama Traffic Interceptor & Model Switcher

Main entry point for the proxy application.
"""

import asyncio
import json
import time
import sys
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .config import get_config
from .logger import TrafficLogger
from .rules_engine import RuleEngine
from .proxy import (
    handle_chat_request, handle_generate_request,
    handle_stream_chat, handle_stream_generate
)
from .health import (
    check_proxy_health, check_target_health, get_status,
    check_dashboard_health
)
from .cors_middleware import add_cors_middleware, setup_cors_config
from .dashboard import app as dashboard_app


async def on_startup():
    """Run on startup."""
    print("PyProxy started")


async def on_shutdown():
    """Run on shutdown."""
    print("PyProxy shutting down")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance
    """
    config = get_config()
    logger = TrafficLogger(log_dir=config.log_dir)
    rule_engine = RuleEngine(logger)

    # Create app
    app = FastAPI(
        title=config.project_name,
        description=config.project_description,
        version=config.project_version,
        debug=config.debug
    )

    # Add middleware
    @app.middleware("http")
    async def log_requests(request, call_next):
        start_time = time.time()

        response = await call_next(request)

        process_time = time.time() - start_time

        if response.status_code == 200:
            logger.log_response(
                request_id=request.headers.get("x-request-id", ""),
                status_code=response.status_code,
                headers=dict(request.headers),
                body=request.json() if request.headers.get("content-type") == "application/json" else None
            )

        return response

    # Add CORS middleware
    add_cors_middleware(app)

    # Add health endpoints
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "proxy": config.target,
                "mode": config.mode
            }
        )

    @app.get("/status")
    async def get_status():
        """Get proxy status."""
        return await get_status()

    @app.get("/dashboard")
    async def dashboard():
        """Redirect to dashboard."""
        return JSONResponse(
            status_code=307,
            headers={"location": f"http://localhost:{config.dashboard_port}/"}
        )

    @app.get("/api/models")
    async def list_models():
        """List available models."""
        try:
            response = await check_target_health()
            return response.content
        except Exception as e:
            return {"error": str(e)}

    @app.get("/api/prompts")
    async def get_prompts():
        """Get prompt templates."""
        return ["Prompt template here..."]

    # Proxy endpoints
    @app.post("/api/chat")
    async def chat_endpoint(request):
        """Handle chat requests."""
        return await handle_chat_request(request, rule_engine, logger)

    @app.post("/api/generate")
    async def generate_endpoint(request):
        """Handle generate requests."""
        return await handle_generate_request(request, rule_engine, logger)

    @app.post("/api/chat/stream")
    async def stream_chat_endpoint(request):
        """Handle streaming chat requests."""
        return await handle_stream_chat(request, rule_engine, logger)

    @app.post("/api/generate/stream")
    async def stream_generate_endpoint(request):
        """Handle streaming generate requests."""
        return await handle_stream_generate(request, rule_engine, logger)

    return app


def main() -> None:
    """
    Main entry point for the application.
    """
    app = create_app()
    config = get_config()

    # Start dashboard in a separate thread
    if config.dashboard_enabled:
        import threading

        def run_dashboard():
            uvicorn.run(
                "ollama_proxy.dashboard:get_app",
                host="0.0.0.0",
                port=config.dashboard_port,
                reload=config.debug
            )

        threading.Thread(
            target=run_dashboard,
            daemon=True
        ).start()

    # Run proxy server
    uvicorn.run(
        "ollama_proxy.main:app",
        host="0.0.0.0",
        port=config.proxy_port,
        reload=config.debug,
        log_level="info"
    )


if __name__ == "__main__":
    main()
