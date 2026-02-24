"""Tests for LLM Gateway adapters."""

from __future__ import annotations

import pytest
import respx
from httpx import Response

from llm_gateway.adapters.openai import OPENAI_MODELS, OpenAIAdapter
from llm_gateway.models import ChatCompletionRequest, Message


@pytest.fixture
def openai_adapter() -> OpenAIAdapter:
    """Create OpenAI adapter for testing."""
    return OpenAIAdapter(api_key="test-key")


@pytest.mark.asyncio
async def test_openai_adapter_chat_completion(openai_adapter: OpenAIAdapter) -> None:
    """Test OpenAI adapter chat completion."""
    with respx.mock:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(
                200,
                json={
                    "id": "test-id",
                    "created": 1234567890,
                    "model": "gpt-4",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                },
            )
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )

        response = await openai_adapter.chat_completion(request)

        assert response.id == "test-id"
        assert response.model == "gpt-4"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_openai_adapter_list_models(openai_adapter: OpenAIAdapter) -> None:
    """Test OpenAI adapter list models."""
    with respx.mock:
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-4", "created": 1234567890, "owned_by": "openai"},
                        {"id": "gpt-3.5-turbo", "created": 1234567890, "owned_by": "openai"},
                    ]
                },
            )
        )

        models = await openai_adapter.list_models()

        assert len(models.data) == 2
        assert models.data[0].id == "gpt-4"


def test_openai_adapter_supports_model(openai_adapter: OpenAIAdapter) -> None:
    """Test OpenAI adapter supports model."""
    assert openai_adapter.supports_model("gpt-4")
    assert openai_adapter.supports_model("gpt-3.5-turbo")
    assert openai_adapter.supports_model("gpt-4-turbo")
    assert not openai_adapter.supports_model("claude-3")
    assert not openai_adapter.supports_model("llama2")


@pytest.mark.asyncio
async def test_openai_adapter_chat_completion_error(openai_adapter: OpenAIAdapter) -> None:
    """Test OpenAI adapter error handling."""
    with respx.mock:
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=Response(401, json={"error": "Invalid API key"})
        )

        request = ChatCompletionRequest(
            model="gpt-4",
            messages=[Message(role="user", content="Hello")],
        )

        with pytest.raises(Exception):
            await openai_adapter.chat_completion(request)
