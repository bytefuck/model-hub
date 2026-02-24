"""OpenAI adapter."""

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

# Known OpenAI models
OPENAI_MODELS = {
    "gpt-4",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k",
}


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI API."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1") -> None:
        super().__init__("openai", api_key, base_url)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._get_headers(),
                timeout=60.0,
            )
        return self._client

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Perform chat completion via OpenAI API."""
        client = await self._get_client()

        payload = {
            "model": self._map_model_name(request.model),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.stop:
            payload["stop"] = request.stop

        logger.info("openai_chat_request", model=request.model)

        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        return ChatCompletionResponse(
            id=data["id"],
            created=data["created"],
            model=data["model"],
            choices=[
                Choice(
                    index=c["index"],
                    message=Message(
                        role=c["message"]["role"],
                        content=c["message"]["content"],
                    ),
                    finish_reason=c.get("finish_reason"),
                )
                for c in data["choices"]
            ],
            usage=Usage(
                prompt_tokens=data["usage"]["prompt_tokens"],
                completion_tokens=data["usage"]["completion_tokens"],
                total_tokens=data["usage"]["total_tokens"],
            ),
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Perform streaming chat completion via OpenAI API."""
        client = await self._get_client()

        payload = {
            "model": self._map_model_name(request.model),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "stream": True,
        }
        if request.stop:
            payload["stop"] = request.stop

        logger.info("openai_chat_stream_request", model=request.model)

        async with client.stream(
            "POST",
            "/chat/completions",
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    import json

                    chunk = json.loads(data)
                    yield ChatCompletionStreamResponse(
                        id=chunk["id"],
                        created=chunk["created"],
                        model=chunk["model"],
                        choices=[
                            StreamChoice(
                                index=c["index"],
                                delta=c.get("delta", {}),
                                finish_reason=c.get("finish_reason"),
                            )
                            for c in chunk["choices"]
                        ],
                    )

    async def list_models(self) -> ModelListResponse:
        """List available OpenAI models."""
        client = await self._get_client()

        response = await client.get("/models")
        response.raise_for_status()
        data = response.json()

        return ModelListResponse(
            data=[
                ModelInfo(
                    id=m["id"],
                    created=m.get("created", 0),
                    owned_by=m.get("owned_by", "openai"),
                )
                for m in data["data"]
            ],
        )

    def supports_model(self, model: str) -> bool:
        """Check if this adapter supports the given model."""
        return model in OPENAI_MODELS or model.startswith("gpt-")

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
