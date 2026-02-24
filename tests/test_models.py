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
                message=Message(role="assistant", content="Hello!"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )

    assert response.id == "test-id"
    assert len(response.choices) == 1
    assert response.choices[0].message.content == "Hello!"
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
