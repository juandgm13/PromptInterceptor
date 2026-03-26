"""
Dashboard module for PyProxy.
"""

import asyncio
import json
import sys
import os

from fastapi import FastAPI, APIRouter, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse

from .config import get_config
from .logger import TrafficLogger
from .rules_engine import RuleEngine
from .health import check_proxy_health, check_target_health, get_status

# Create app
app = FastAPI(
    title="PyProxy Dashboard",
    description="Dashboard for PyProxy - Ollama Traffic Intercepter & Model Switcher",
    version="0.1.0",
)

config = get_config()
logger = TrafficLogger()
rule_engine = RuleEngine(logger)

# Create router
router = APIRouter()


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve dashboard HTML."""
    dashboard_dir = os.path.join(os.path.dirname(__file__), "static")

    if os.path.exists(dashboard_dir):
        with open(os.path.join(dashboard_dir, "index.html")) as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PyProxy Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        h1 { color: #fff; margin-bottom: 20px; }
        .status { padding: 15px; border-radius: 8px; margin-bottom: 20px; }
        .healthy { background: #0f0; }
        .unhealthy { background: #f00; }
        .section { background: #16213e; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        .rules { display: grid; gap: 10px; }
        .rule { background: #1a1a2e; padding: 15px; border-radius: 6px; }
        .rule button { margin: 5px; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; }
        .enable { background: #0f0; }
        .disable { background: #f00; }
        .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 20px; }
        .stat { background: #1a1a2e; padding: 15px; border-radius: 6px; text-align: center; }
        .stat-value { font-size: 2em; color: #0f0; }
        .stat-label { color: #888; }
        footer { text-align: center; margin-top: 30px; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>PyProxy Dashboard</h1>

        <div class="status healthy">
            <strong>Status:</strong> Running
        </div>

        <div class="section">
            <h2>Rules</h2>
            <div class="rules">
                <div class="rule">
                    <strong>Model Switcher</strong>
                    <p>Switch models between requests</p>
                    <button class="enable" onclick="toggleRule(0)">✓ Enable</button>
                </div>
                <div class="rule">
                    <strong>Temperature Tuner</strong>
                    <p>Adjust temperature to 0.5</p>
                    <button class="enable" onclick="toggleRule(1)">✓ Enable</button>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Statistics</h2>
            <div class="stats">
                <div class="stat">
                    <div class="stat-value">0</div>
                    <div class="stat-label">Total Requests</div>
                </div>
                <div class="stat">
                    <div class="stat-value">0</div>
                    <div class="stat-label">Total Responses</div>
                </div>
                <div class="stat">
                    <div class="stat-value">2</div>
                    <div class="stat-label">Active Rules</div>
                </div>
            </div>
        </div>

        <footer>
            PyProxy Dashboard v0.1.0
        </footer>
    </div>
</body>
</html>
        """
        )


@router.get("/status")
async def status():
    """Get proxy status."""
    return await get_status()


@router.get("/health")
async def health():
    """Get health status."""
    return await check_proxy_health()


@router.get("/target-health")
async def target_health():
    """Get target health status."""
    return await check_target_health()


@router.get("/stats")
async def stats():
    """Get statistics."""
    return await get_status()


@router.get("/logs")
async def logs(limit: int = 10):
    """Get recent logs."""
    logs = logger.get_logs(limit=limit)
    return {"logs": logs}


@router.get("/raw-logs")
async def raw_logs():
    """Get all raw logs."""
    logs = logger.get_all_logs()
    return {"logs": logs}


@router.get("/rules")
async def get_rules():
    """Get current rules."""
    return {"rules": rule_engine.get_rules()}


@router.get("/enable-rule/{index:int}")
async def enable_rule(index: int):
    """Enable a rule."""
    if rule_engine.enable_rule(index):
        return JSONResponse(status_code=200, content={"status": "enabled"})
    return JSONResponse(status_code=400, content={"status": "not_found"})


@router.get("/disable-rule/{index:int}")
async def disable_rule(index: int):
    """Disable a rule."""
    if rule_engine.disable_rule(index):
        return JSONResponse(status_code=200, content={"status": "disabled"})
    return JSONResponse(status_code=400, content={"status": "not_found"})


def get_app() -> FastAPI:
    """Get the dashboard app instance."""
    return app


# Mount static files if they exist
if not os.path.exists(os.path.join(os.path.dirname(__file__), "static")):
    os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Export
__all__ = ["app", "get_app"]
