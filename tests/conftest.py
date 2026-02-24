"""Test fixtures and utilities."""

from __future__ import annotations

import pytest

from llm_gateway.adapters.openai import OpenAIAdapter
from llm_gateway.models import ChatCompletionRequest, Message


@pytest.fixture
def sample_chat_request() -> ChatCompletionRequest:
    """Create a sample chat completion request."""
    return ChatCompletionRequest(
        model="gpt-4",
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello!"),
        ],
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def openai_adapter_mock() -> OpenAIAdapter:
    """Create a mock OpenAI adapter."""
    return OpenAIAdapter(api_key="test-api-key")
