"""
Request/Response models for PyProxy.

Defines Pydantic models for:
- Chat requests/responses
- Generate requests/responses
- Stream responses
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""

    model: str
    messages: List[dict]
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    options: Optional[dict] = None


class GenerateRequest(BaseModel):
    """Generate request model."""

    model: str
    prompt: str
    stream: bool = False
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    options: Optional[dict] = None


class SystemPrompt(BaseModel):
    """System prompt to inject."""

    prompt: str
    optional: bool = True
    forced: bool = False


class MatchRule(BaseModel):
    """Rule for matching incoming requests."""

    path: Optional[str] = None
    header: Optional[str] = None
    value: str
    operation: str = "equals"


class ReplaceRule(BaseModel):
    """Rule for replacing values in responses."""

    jsonpath: str
    value: Optional[str] = None
    optional: bool = True
    forced: bool = False


class ProxyConfig(BaseModel):
    """Configuration for proxy rules."""

    proxy_port: int = 8080
    target: str = "http://localhost:11434"
    mode: str = "intercept"
    timeout: int = 120
    log_dir: str = "logs"
    max_log_files: int = 10
    dashboard_enabled: bool = True
    dashboard_port: int = 9090
    log_size_limit: int = 1024 * 1024

    # Rules
    rules: List[dict] = Field(default_factory=list)

    # CORS
    allow_origins: List[str] = Field(default_factory=lambda: ["*"])
    allow_methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "OPTIONS"])
    allow_headers: List[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    max_age: int = 86400


class ChatResponse(BaseModel):
    """Chat response model."""

    model: str
    choices: List[dict]
    usage: dict
    created: int


class GenerateResponse(BaseModel):
    """Generate response model."""

    model: str
    choices: List[dict]
    usage: dict
    created: int


class StreamResponse(BaseModel):
    """Stream response model."""

    model: str
    choices: List[dict]
