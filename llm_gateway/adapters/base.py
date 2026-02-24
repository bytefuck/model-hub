"""Base adapter interface for LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from llm_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ModelListResponse,
)


class BaseAdapter(ABC):
    """Base class for LLM provider adapters."""

    def __init__(self, name: str, api_key: str | None = None, base_url: str | None = None) -> None:
        self.name = name
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Perform chat completion."""
        raise NotImplementedError

    @abstractmethod
    def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Perform streaming chat completion."""
        raise NotImplementedError

    @abstractmethod
    async def list_models(self) -> ModelListResponse:
        """List available models."""
        raise NotImplementedError

    @abstractmethod
    def supports_model(self, model: str) -> bool:
        """Check if this adapter supports the given model."""
        raise NotImplementedError

    def _get_headers(self) -> dict[str, str]:
        """Get default headers for API requests."""
        headers: dict[str, str] = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _map_model_name(self, model: str) -> str:
        """Map model name to provider-specific format. Override if needed."""
        return model
