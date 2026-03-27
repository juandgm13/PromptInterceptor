"""Tests for models/request_model.py."""

import pytest
from pydantic import ValidationError

from prompt_interceptor.models.request_model import (
    Message,
    ChatRequest,
    GenerateRequest,
    ChatResponse,
    GenerateResponse,
    ResponseMessage,
    ErrorResponse,
    RawChatRequest,
    RawGenerateRequest,
    parse_chat_request,
    parse_generate_request,
    create_chat_response,
    create_generate_response,
)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

def test_message_valid():
    m = Message(role="user", content="hello")
    assert m.role == "user"
    assert m.content == "hello"
    assert m.images is None


def test_message_invalid_role():
    with pytest.raises(ValidationError):
        Message(role="unknown", content="hi")


def test_message_with_images():
    m = Message(role="user", content="look", images=["base64data"])
    assert m.images == ["base64data"]


# ---------------------------------------------------------------------------
# ChatRequest
# ---------------------------------------------------------------------------

def test_chat_request_valid():
    r = ChatRequest(
        model="llama3",
        messages=[Message(role="user", content="hi")],
    )
    assert r.model == "llama3"
    assert len(r.messages) == 1


def test_chat_request_empty_messages_raises():
    with pytest.raises(ValidationError):
        ChatRequest(model="llama3", messages=[])


def test_chat_request_optional_fields():
    r = ChatRequest(
        model="llama3",
        messages=[Message(role="user", content="hi")],
        stream=True,
        format="json",
        options={"temperature": 0.5},
    )
    assert r.stream is True
    assert r.options == {"temperature": 0.5}


# ---------------------------------------------------------------------------
# GenerateRequest
# ---------------------------------------------------------------------------

def test_generate_request_valid():
    r = GenerateRequest(model="llama3", prompt="Say hello")
    assert r.prompt == "Say hello"
    assert r.stream is False


def test_generate_request_with_options():
    r = GenerateRequest(model="llama3", prompt="Hi", options={"seed": 42})
    assert r.options["seed"] == 42


# ---------------------------------------------------------------------------
# ChatResponse
# ---------------------------------------------------------------------------

def test_chat_response_minimal():
    r = ChatResponse(model="llama3", created_at="2026-01-01T00:00:00Z", done=True)
    assert r.model == "llama3"
    assert r.done is True
    assert r.load_time is None  # was `load time` before fix


def test_chat_response_with_message():
    r = ChatResponse(
        model="llama3",
        created_at="2026-01-01T00:00:00Z",
        message=ResponseMessage(role="assistant", content="Hello!"),
        done=True,
    )
    assert r.message.content == "Hello!"


def test_chat_response_message_can_be_none():
    r = ChatResponse(model="llama3", created_at="now", message=None)
    assert r.message is None


# ---------------------------------------------------------------------------
# GenerateResponse
# ---------------------------------------------------------------------------

def test_generate_response_valid():
    r = GenerateResponse(model="llama3", created_at="now", response="Hi", done=True)
    assert r.response == "Hi"
    assert r.load_duration is None


# ---------------------------------------------------------------------------
# ErrorResponse
# ---------------------------------------------------------------------------

def test_error_response():
    e = ErrorResponse(error="something went wrong")
    assert e.error == "something went wrong"


# ---------------------------------------------------------------------------
# Raw models
# ---------------------------------------------------------------------------

def test_raw_chat_request():
    r = RawChatRequest(
        model="llama3",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert r.model == "llama3"
    assert r.stream is False
    assert r.format is None


def test_raw_generate_request():
    r = RawGenerateRequest(model="llama3", prompt="hello")
    assert r.options is None


# ---------------------------------------------------------------------------
# parse_chat_request
# ---------------------------------------------------------------------------

def test_parse_chat_request():
    body = {
        "model": "llama3",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "options": {"temperature": 0.7},
    }
    r = parse_chat_request(body)
    assert r.model == "llama3"
    assert r.messages[0].role == "user"
    assert r.stream is True


def test_parse_chat_request_defaults():
    r = parse_chat_request({"messages": [{"role": "user", "content": "hi"}]})
    assert r.model == "llama3"


# ---------------------------------------------------------------------------
# parse_generate_request
# ---------------------------------------------------------------------------

def test_parse_generate_request():
    body = {"model": "mistral", "prompt": "Hello"}
    r = parse_generate_request(body)
    assert r.model == "mistral"
    assert r.prompt == "Hello"


def test_parse_generate_request_defaults():
    r = parse_generate_request({})
    assert r.model == "llama3"
    assert r.prompt == ""


# ---------------------------------------------------------------------------
# create_chat_response / create_generate_response
# ---------------------------------------------------------------------------

def test_create_chat_response():
    msg = Message(role="assistant", content="Hi!")
    r = create_chat_response(msg, model="llama3", created_at="now")
    assert r.model == "llama3"
    assert r.message.content == "Hi!"
    assert r.done is True


def test_create_generate_response():
    r = create_generate_response("Hello world", model="llama3", created_at="now")
    assert r.response == "Hello world"
    assert r.done is True
