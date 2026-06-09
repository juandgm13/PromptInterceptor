"""
PromptInterceptor - Ollama Traffic Interceptor for AI Clients

Main entry point for the proxy application.
"""

import shutil
import time
import threading

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .config import get_config
from .logger import TrafficLogger
from .rules_engine import RuleEngine
from .proxy import (
    handle_chat_request, handle_generate_request,
    handle_stream_chat, handle_stream_generate,
    handle_v1_messages, handle_v1_chat_completions,
    handle_passthrough,
)
from .health import get_models as _get_models, get_status as _get_status
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
        return RedirectResponse(
            url=f"http://localhost:{config.dashboard_port}/",
            status_code=307,
        )

    @app.get("/api/models")
    async def list_models():
        try:
            response = await _get_models()
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

    @app.post("/v1/messages")
    async def v1_messages_endpoint(request: Request):
        return await handle_v1_messages(request, rule_engine, logger)

    @app.post("/v1/chat/completions")
    async def v1_chat_completions_endpoint(request: Request):
        return await handle_v1_chat_completions(request, rule_engine, logger)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
    async def passthrough_endpoint(request: Request):
        return await handle_passthrough(request, logger)

    return app


# Module-level app so uvicorn can reference "prompt_interceptor.main:app"
app = create_app()


def _warn_if_bash_missing() -> None:
    """Show a warning popup if windows_bash_mode is enabled but bash is not in PATH."""
    if not get_config().windows_bash_mode:
        return
    if shutil.which('bash'):
        return
    msg = (
        "Bash no encontrado en el sistema.\n\n"
        "windows_bash_mode está activado pero bash no está disponible en PATH.\n\n"
        "El comportamiento de los comandos bash puede ser incorrecto.\n\n"
        "Se recomienda instalar Git Bash (https://gitforwindows.org/) o habilitar WSL."
    )
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("PromptInterceptor — Bash no encontrado", msg)
        root.destroy()
    except Exception:
        print(
            "[PromptInterceptor] WARNING: windows_bash_mode activo pero bash no encontrado. "
            "Se recomienda instalar Git Bash o WSL."
        )


def main() -> None:
    """Main entry point."""
    config = get_config()
    _warn_if_bash_missing()

    if config.dashboard_enabled:
        def _run_dashboard():
            uvicorn.run(
                "prompt_interceptor.dashboard:app",
                host="0.0.0.0",
                port=config.dashboard_port,
                log_level="warning",
            )

        threading.Thread(target=_run_dashboard, daemon=True).start()

    uvicorn.run(
        "prompt_interceptor.main:app",
        host=config.proxy_host,
        port=config.proxy_port,
        reload=config.debug,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
