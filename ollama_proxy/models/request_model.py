"""
Request/Response models for PyProxy.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing_extensions import Annotated


class Message(BaseModel):
    """Ollama message model."""

    role: Literal["user", "assistant", "system"]
    content: str
    images: Optional[List[str]] = None

    model_config = ConfigDict(extra="allow")


class ChatRequest(BaseModel):
    """Ollama /api/chat request model."""

    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: List[Message]) -> List[Message]:
        if not v:
            raise ValueError("messages list cannot be empty")
        return v


class GenerateRequest(BaseModel):
    """Ollama /api/generate request model."""

    model: str
    prompt: str
    stream: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str

    model_config = ConfigDict(extra="allow")


class ResponseMessage(BaseModel):
    """Response message."""

    role: str
    content: str
    images: Optional[List[str]] = None

    model_config = ConfigDict(extra="allow")


class ChatResponse(BaseModel):
    """Ollama /api/chat response model."""

    model: str
    created_at: str
    message: Optional[ResponseMessage] = None
    done: bool = False
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
        load time = Optional[int] = None
    duration: Optional[float] = None

    model_config = ConfigDict(extra="allow")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: ResponseMessage) -> ResponseMessage:
        if v is None:
            raise ValueError("message cannot be None")
        return v


class GenerateResponse(BaseModel):
    """Ollama /api/generate response model."""

    model: str
    created_at: str
    response: str
    done: bool = False
    done_reason: Optional[str] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    duration: Optional[float] = None
    context: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class RawChatRequest(BaseModel):
    """Raw chat request (for logging)."""

    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    format: str = None
    options: Dict[str, Any] = None


class RawGenerateRequest(BaseModel):
    """Raw generate request (for logging)."""

    model: str
    prompt: str
    stream: bool = False
    format: str = None
    options: Dict[str, Any] = None


# Helper functions

def parse_chat_request(body: Dict[str, Any]) -> ChatRequest:
    """Parse raw chat request to model."""
    messages = [
        Message(
            role=msg["role"],
            content=msg["content"],
            images=msg.get("images")
        )
        for msg in body.get("messages", [])
    ]
    return ChatRequest(
        model=body.get("model", "llama3"),
        messages=messages,
        stream=body.get("stream", False),
        format=body.get("format"),
        options=body.get("options") or {}
    )


def parse_generate_request(body: Dict[str, Any]) -> GenerateRequest:
    """Parse raw generate request to model."""
    return GenerateRequest(
        model=body.get("model", "llama3"),
        prompt=body.get("prompt", ""),
        stream=body.get("stream", False),
        format=body.get("format"),
        options=body.get("options") or {}
    )


def create_chat_response(
    message: Message,
    model: str,
    created_at: str,
    done: bool = True
) -> ChatResponse:
    """Create chat response."""
    return ChatResponse(
        model=model,
        created_at=created_at,
        message=ResponseMessage(
            role=message.role,
            content=message.content,
            images=message.images
        ),
        done=done
    )


def create_generate_response(
    response: str,
    model: str,
    created_at: str,
    done: bool = True
) -> GenerateResponse:
    """Create generate response."""
    return GenerateResponse(
        model=model,
        created_at=created_at,
        response=response,
        done=done
    )
