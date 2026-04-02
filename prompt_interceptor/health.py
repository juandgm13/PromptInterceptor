"""
Health check endpoints for PyProxy.
"""

import json
from datetime import datetime, timezone

import httpx
from fastapi.responses import JSONResponse

from .config import get_config
from .logger import TrafficLogger
from .rules_engine import RuleEngine


async def check_proxy_health() -> JSONResponse:
    """Check proxy server health."""
    config = get_config()

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "proxy_port": config.proxy_port,
            "target": config.target,
            "mode": config.mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def check_target_health() -> JSONResponse:
    """Check target Ollama server health."""
    config = get_config()

    try:
        async with httpx.AsyncClient(timeout=config.health_timeout) as client:
            response = await client.get(config.target + "/api/tags")
            if response.status_code == 200:
                return JSONResponse(
                    status_code=200,
                    content={
                        "proxy": "healthy",
                        "target": "healthy",
                        "message": f"Connected to {config.target}",
                        "models": response.json().get("models", []),
                    },
                )
            else:
                return JSONResponse(
                    status_code=response.status_code,
                    content={"target": "unhealthy", "error": response.text},
                )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={"target": "timeout"},
        )
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"target": "unhealthy", "error": str(e)},
        )


async def check_dashboard_health() -> JSONResponse:
    """Check dashboard health."""
    config = get_config()

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "dashboard_port": config.dashboard_port,
            "dashboard_enabled": config.dashboard_enabled,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )


async def get_status() -> JSONResponse:
    """Get comprehensive status information."""
    config = get_config()
    logger = TrafficLogger()
    rule_engine = RuleEngine(logger)

    target_response = await check_target_health()
    target_data = json.loads(target_response.body)

    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "proxy": {
                "port": config.proxy_port,
                "target": config.target,
                "mode": config.mode,
            },
            "target": target_data,
            "dashboard": {
                "enabled": config.dashboard_enabled,
                "port": config.dashboard_port,
            },
            "rules": {
                "count": len(rule_engine.rules),
                "enabled_count": sum(1 for r in rule_engine.rules if r.enabled),
            },
            "logs": {
                "dir": config.log_dir,
                "max_files": config.max_log_files,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
