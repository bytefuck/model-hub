"""Worker proxy handlers."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import structlog

from llm_gateway.config import settings

if TYPE_CHECKING:
    from llm_gateway.worker.registration import RegistrationClient

logger = structlog.get_logger()


class ProxyHandler:
    """Handles proxying requests to backend."""

    def __init__(
        self,
        backend_url: str,
        registration_client: RegistrationClient,
        timeout: int = 120,
    ) -> None:
        self.backend_url = backend_url
        self.registration_client = registration_client
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.backend_url,
            timeout=self.timeout,
        )
        logger.info("proxy_handler_started", backend_url=self.backend_url)

    async def stop(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
        logger.info("proxy_handler_stopped")

    async def proxy_chat_completion(self, payload: dict) -> dict:
        """Proxy non-streaming chat completion request."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        self.registration_client.increment_load()
        try:
            response = await self._client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(
                "backend_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except httpx.TimeoutException:
            logger.error("backend_timeout")
            raise
        finally:
            self.registration_client.decrement_load()

    async def proxy_chat_completion_stream(self, payload: dict):
        """Proxy streaming chat completion request."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        self.registration_client.increment_load()
        try:
            async with self._client.stream(
                "POST", "/v1/chat/completions", json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk
        except httpx.HTTPStatusError as e:
            logger.error(
                "backend_stream_error",
                status_code=e.response.status_code,
                error=str(e),
            )
            raise
        except httpx.TimeoutException:
            logger.error("backend_stream_timeout")
            raise
        finally:
            self.registration_client.decrement_load()

    async def check_backend_health(self) -> tuple[bool, str]:
        """Check if backend is healthy."""
        if not self._client:
            return False, "Client not initialized"

        try:
            response = await self._client.get("/health", timeout=5.0)
            if response.status_code == 200:
                return True, "healthy"
            return False, f"Backend returned {response.status_code}"
        except httpx.ConnectError:
            return False, "backend unreachable"
        except httpx.TimeoutException:
            return False, "backend timeout"
        except Exception as e:
            return False, str(e)
