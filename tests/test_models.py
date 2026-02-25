"""Tests for LLM Gateway models."""

from __future__ import annotations

import pytest

from llm_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    Message,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    Usage,
)


def test_message_creation() -> None:
    """Test Message model creation."""
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"


def test_chat_completion_request_defaults() -> None:
    """Test ChatCompletionRequest default values."""
    req = ChatCompletionRequest(model="gpt-4", messages=[])
    assert req.temperature == 0.7
    assert req.top_p == 1.0
    assert req.stream is False
    assert req.max_tokens is None


def test_chat_completion_request_validation() -> None:
    """Test ChatCompletionRequest validation."""
    with pytest.raises(ValueError):
        ChatCompletionRequest(model="gpt-4", temperature=3.0, messages=[])

    with pytest.raises(ValueError):
        ChatCompletionRequest(model="gpt-4", top_p=1.5, messages=[])


def test_chat_completion_response() -> None:
    """Test ChatCompletionResponse model."""
    response = ChatCompletionResponse(
        id="test-id",
        created=1234567890,
        model="gpt-4",
        choices=[
            Choice(
                index=0,
                message={"role": "assistant", "content": "Hello!"},
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    assert response.id == "test-id"
    assert len(response.choices) == 1
    assert response.choices[0].message["content"] == "Hello!"
    assert response.usage.total_tokens == 15


def test_model_info() -> None:
    """Test ModelInfo model."""
    model = ModelInfo(id="gpt-4", owned_by="openai")
    assert model.id == "gpt-4"
    assert model.owned_by == "openai"
    assert model.object == "model"


def test_model_list_response() -> None:
    """Test ModelListResponse model."""
    response = ModelListResponse(
        data=[
            ModelInfo(id="gpt-4"),
            ModelInfo(id="gpt-3.5-turbo"),
        ]
    )
    assert len(response.data) == 2
    assert response.object == "list"


def test_stream_choice() -> None:
    """Test StreamChoice model."""
    choice = StreamChoice(index=0, delta={"content": "Hello"})
    assert choice.delta["content"] == "Hello"
    assert choice.finish_reason is None


def test_message_with_tool_role() -> None:
    """Test Message model with tool role."""
    msg = Message(
        role="tool",
        content='{"temperature": 20}',
        tool_call_id="call_abc123",
    )
    assert msg.role == "tool"
    assert msg.content == '{"temperature": 20}'
    assert msg.tool_call_id == "call_abc123"


def test_message_with_tool_calls() -> None:
    """Test Message model with tool_calls."""
    tool_calls = [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Beijing"}',
            },
        }
    ]
    msg = Message(role="assistant", content=None, tool_calls=tool_calls)
    assert msg.role == "assistant"
    assert msg.tool_calls == tool_calls
    assert msg.tool_calls[0]["function"]["name"] == "get_weather"


def test_chat_completion_request_with_tools() -> None:
    """Test ChatCompletionRequest with tools parameter."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "City name"}},
                    "required": ["location"],
                },
            },
        }
    ]
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[Message(role="user", content="What's the weather?")],
        tools=tools,
    )
    assert req.tools is not None
    assert len(req.tools) == 1
    assert req.tools[0]["function"]["name"] == "get_weather"


def test_chat_completion_request_with_tool_choice() -> None:
    """Test ChatCompletionRequest with tool_choice parameter."""
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[Message(role="user", content="Use the weather tool")],
        tool_choice="auto",
    )
    assert req.tool_choice == "auto"

    req2 = ChatCompletionRequest(
        model="gpt-4",
        messages=[Message(role="user", content="Use specific function")],
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
    )
    assert req2.tool_choice["function"]["name"] == "get_weather"


def test_chat_completion_request_with_response_format() -> None:
    """Test ChatCompletionRequest with response_format parameter."""
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[Message(role="user", content="Give me JSON")],
        response_format={"type": "json_object"},
    )
    assert req.response_format == {"type": "json_object"}

    req2 = ChatCompletionRequest(
        model="gpt-4",
        messages=[Message(role="user", content="Give me structured data")],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "person", "schema": {"type": "object"}},
        },
    )
    assert req2.response_format["type"] == "json_schema"


def test_choice_with_tool_calls() -> None:
    """Test Choice model with tool_calls in message."""
    message_with_tool_calls = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location": "Beijing"}',
                },
            }
        ],
    }
    choice = Choice(
        index=0,
        message=message_with_tool_calls,
        finish_reason="tool_calls",
    )
    assert choice.finish_reason == "tool_calls"
    assert choice.message["tool_calls"][0]["function"]["name"] == "get_weather"


def test_stream_choice_with_tool_calls() -> None:
    """Test StreamChoice model with tool_calls."""
    delta = {
        "tool_calls": [
            {
                "id": "call_abc123",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"location":',
                },
            }
        ]
    }
    choice = StreamChoice(index=0, delta=delta, finish_reason="tool_calls")
    assert choice.finish_reason == "tool_calls"
    assert choice.delta["tool_calls"][0]["function"]["name"] == "get_weather"


def test_message_serialization_with_tool_fields() -> None:
    """Test Message model serialization includes tool fields."""
    msg = Message(
        role="tool",
        content='{"result": "success"}',
        tool_call_id="call_xyz789",
    )
    data = msg.model_dump()
    assert data["role"] == "tool"
    assert data["content"] == '{"result": "success"}'
    assert data["tool_call_id"] == "call_xyz789"


def test_request_serialization_passes_through() -> None:
    """Test ChatCompletionRequest serialization passes through tool fields."""
    tools = [{"type": "function", "function": {"name": "test", "parameters": {}}}]
    req = ChatCompletionRequest(
        model="gpt-4",
        messages=[],
        tools=tools,
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    data = req.model_dump()
    assert "tools" in data
    assert "tool_choice" in data
    assert "response_format" in data
