"""Ollama adapter for local models."""

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


class OllamaAdapter(BaseAdapter):
    """Adapter for Ollama local LLM server."""

    def __init__(self, host: str = "http://localhost:11434") -> None:
        super().__init__("ollama", None, host)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=120.0,  # Local models may take longer
            )
        return self._client

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Perform chat completion via Ollama API."""
        client = await self._get_client()

        # Ollama uses its own format but supports OpenAI-compatible API
        payload = {
            "model": self._map_model_name(request.model),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {},
        }

        if request.temperature is not None:
            payload["options"]["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens
        if request.top_p is not None:
            payload["options"]["top_p"] = request.top_p
        if request.stop:
            payload["options"]["stop"] = request.stop

        logger.info("ollama_chat_request", model=request.model)

        response = await client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})

        return ChatCompletionResponse(
            id=f"ollama-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(
                        role=message.get("role", "assistant"),
                        content=message.get("content", ""),
                    ),
                    finish_reason="stop",
                ),
            ],
            usage=Usage(
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            ),
        )

    async def chat_completion_stream(
        self,
        request: ChatCompletionRequest,
    ) -> AsyncIterator[ChatCompletionStreamResponse]:
        """Perform streaming chat completion via Ollama API."""
        client = await self._get_client()

        payload = {
            "model": self._map_model_name(request.model),
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": True,
            "options": {},
        }

        if request.temperature is not None:
            payload["options"]["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["options"]["num_predict"] = request.max_tokens

        logger.info("ollama_chat_stream_request", model=request.model)

        created = int(time.time())
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line:
                    import json

                    try:
                        data = json.loads(line)
                        message = data.get("message", {})
                        yield ChatCompletionStreamResponse(
                            id=f"ollama-{created}",
                            created=created,
                            model=request.model,
                            choices=[
                                StreamChoice(
                                    index=0,
                                    delta={"content": message.get("content", "")},
                                    finish_reason="stop" if data.get("done") else None,
                                ),
                            ],
                        )
                    except json.JSONDecodeError:
                        continue

    async def list_models(self) -> ModelListResponse:
        """List available Ollama models."""
        client = await self._get_client()

        try:
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            return ModelListResponse(
                data=[
                    ModelInfo(
                        id=model["name"],
                        owned_by="ollama",
                    )
                    for model in data.get("models", [])
                ],
            )
        except Exception:
            logger.warning("ollama_list_models_failed")
            return ModelListResponse(data=[])

    def supports_model(self, model: str) -> bool:
        """Check if this adapter supports the given model."""
        # Ollama supports any locally available model
        # We'll try to route common model names and any name with ':'
        return ":" in model or model in [
            "llama2",
            "llama3",
            "mistral",
            "mixtral",
            "codellama",
            "phi",
            "qwen",
            "gemma",
        ]

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
