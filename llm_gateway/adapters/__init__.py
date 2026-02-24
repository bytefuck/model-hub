"""Adapter registry and factory."""

from __future__ import annotations

from typing import TypeVar

from llm_gateway.adapters.anthropic import AnthropicAdapter
from llm_gateway.adapters.base import BaseAdapter
from llm_gateway.adapters.ollama import OllamaAdapter
from llm_gateway.adapters.openai import OpenAIAdapter
from llm_gateway.config import settings

T = TypeVar("T", bound=BaseAdapter)


class AdapterRegistry:
    """Registry for LLM provider adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, BaseAdapter] = {}

    async def initialize(self) -> None:
        """Initialize adapters based on configuration."""
        # OpenAI
        if settings.openai_api_key:
            self._adapters["openai"] = OpenAIAdapter(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )

        # Anthropic
        if settings.anthropic_api_key:
            self._adapters["anthropic"] = AnthropicAdapter(
                api_key=settings.anthropic_api_key,
                base_url=settings.anthropic_base_url,
            )

        # Ollama (always try to initialize as it may be running locally)
        self._adapters["ollama"] = OllamaAdapter(host=settings.ollama_host)

    def get_adapter(self, model: str) -> BaseAdapter:
        """Get adapter for a specific model."""
        # Try to find adapter that supports this model
        for adapter in self._adapters.values():
            if adapter.supports_model(model):
                return adapter

        # Default to OpenAI if available, otherwise first available
        if "openai" in self._adapters:
            return self._adapters["openai"]

        if not self._adapters:
            raise RuntimeError("No adapters configured")

        return next(iter(self._adapters.values()))

    def list_adapters(self) -> list[str]:
        """List available adapter names."""
        return list(self._adapters.keys())

    async def close_all(self) -> None:
        """Close all adapter connections."""
        for adapter in self._adapters.values():
            if hasattr(adapter, "close"):
                await adapter.close()
        self._adapters.clear()


# Global registry instance
registry = AdapterRegistry()
