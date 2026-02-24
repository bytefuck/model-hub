"""Anthropic Claude adapter."""

from __future__ import annotations

import time
from typing import AsyncIterator

import httpx
import structlog

from llm_gateway.adapters.base import BaseAdapter
from llm_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    Usage,
)

logger = structlog.get_logger()

# Known Anthropic models
ANTHROPIC_MODELS = {
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
}


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic Claude API."""

    def __init__(self, api_key: str, base_url: str = "https://api.anthropic.com") -> None:
        super().__init__("anthropic", api_key, base_url)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            }
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=60.0,
            )
        return self._client

    def _convert_messages(
        self,
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert OpenAI format messages to Anthropic format."""
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )

        return system_message, anthropic_messages

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Perform chat completion via Anthropic API."""
        client = await self._get_client()

        system, messages = self._convert_messages(request.messages)

        payload: dict[str, object] = {
            "model": self._map_model_name(request.model),
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if system:
            payload["system"] = system
        if request.stop:
            payload["stop_sequences"] = (
                [request.stop] if isinstance(request.stop, str) else request.stop
            )

        logger.info("anthropic_chat_request", model=request.model)

        response = await client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        # Map Anthropic response to OpenAI format
        content = ""
        if data.get("content"):
            for block in data["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")

        return ChatCompletionResponse(
            id=data["id"],
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=content),
                    finish_reason=data.get("stop_reason", "stop"),
                ),
            ],
            usage=Usage(
                prompt_tokens=data["usage"]["input_tokens"],
                completion_tokens=data["usage"]["output_tokens"],
                total_tokens=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
            ),
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Perform streaming chat completion via Anthropic API."""
        client = await self._get_client()

        system, messages = self._convert_messages(request.messages)

        payload: dict[str, object] = {
            "model": self._map_model_name(request.model),
            "messages": messages,
            "max_tokens": request.max_tokens or 4096,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }
        if system:
            payload["system"] = system

        logger.info("anthropic_chat_stream_request", model=request.model)

        created = int(time.time())
        async with client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    import json

                    event = json.loads(line[6:])
                    event_type = event.get("type")

                    if event_type == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            yield ChatCompletionStreamResponse(
                                id=event.get("message", {}).get("id", "anthropic-"),
                                created=created,
                                model=request.model,
                                choices=[
                                    StreamChoice(
                                        index=0,
                                        delta={"content": delta.get("text", "")},
                                    ),
                                ],
                            )

    async def list_models(self) -> ModelListResponse:
        """List available Anthropic models."""
        # Anthropic doesn't have a models endpoint, return known models
        return ModelListResponse(
            data=[ModelInfo(id=model, owned_by="anthropic") for model in ANTHROPIC_MODELS],
        )

    def supports_model(self, model: str) -> bool:
        """Check if this adapter supports the given model."""
        return model in ANTHROPIC_MODELS or model.startswith("claude-")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
