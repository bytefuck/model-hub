"""Worker registration client."""

from __future__ import annotations

import asyncio
import signal
from typing import TYPE_CHECKING, Any

import httpx
import structlog

from llm_gateway.config import settings

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class RegistrationClient:
    """Client for worker registration with controller."""

    def __init__(
        self,
        worker_id: str,
        model_id: str,
        controller_url: str,
        backend_url: str,
        capacity: int = 10,
        heartbeat_interval: int = 10,
        retry_count: int = 30,
        retry_delay: int = 5,
    ) -> None:
        self.worker_id = worker_id
        self.model_id = model_id
        self.controller_url = controller_url
        self.backend_url = backend_url
        self.capacity = capacity
        self.heartbeat_interval = heartbeat_interval
        self.retry_count = retry_count
        self.retry_delay = retry_delay

        self._client: httpx.AsyncClient | None = None
        self._current_load: int = 0
        self._running: bool = False
        self._heartbeat_task: asyncio.Task | None = None
        self._registered: bool = False

    async def start(self) -> None:
        """Start the registration client and register with controller."""
        self._client = httpx.AsyncClient(timeout=10.0)
        self._running = True

        await self._register_with_retry()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("registration_client_started", worker_id=self.worker_id)

    async def stop(self) -> None:
        """Stop the registration client and deregister."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._registered:
            await self._deregister()

        if self._client:
            await self._client.aclose()

        logger.info("registration_client_stopped", worker_id=self.worker_id)

    async def _register_with_retry(self) -> None:
        """Register with exponential backoff retry."""
        delay = self.retry_delay

        for attempt in range(1, self.retry_count + 1):
            try:
                await self._register()
                self._registered = True
                logger.info(
                    "worker_registered",
                    worker_id=self.worker_id,
                    attempt=attempt,
                )
                return
            except Exception as e:
                logger.warning(
                    "registration_failed",
                    worker_id=self.worker_id,
                    attempt=attempt,
                    error=str(e),
                    next_delay=delay,
                )

                if attempt == self.retry_count:
                    raise RuntimeError(f"Failed to register after {self.retry_count} attempts: {e}")

                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    async def _register(self) -> None:
        """Send registration request to controller."""
        if not self._client:
            raise RuntimeError("Client not initialized")

        payload = {
            "worker_id": self.worker_id,
            "model_id": self.model_id,
            "endpoint": f"http://localhost:{settings.listen_port}",
            "capacity": self.capacity,
            "metadata": {"backend_url": self.backend_url},
        }

        headers = self._get_headers()
        response = await self._client.post(
            f"{self.controller_url}/internal/workers/register",
            json=payload,
            headers=headers,
        )
        logger.info(
            "registration_response",
            worker_id=self.worker_id,
            status_code=response.status_code,
            response=response.json(),
        )
        response.raise_for_status()

    async def _deregister(self) -> None:
        """Send deregistration request to controller."""
        if not self._client:
            return

        try:
            headers = self._get_headers()
            response = await self._client.delete(
                f"{self.controller_url}/internal/workers/{self.worker_id}",
                headers=headers,
            )
            response.raise_for_status()
            logger.info("worker_deregistered", worker_id=self.worker_id)
        except Exception as e:
            logger.warning(
                "deregistration_failed",
                worker_id=self.worker_id,
                error=str(e),
            )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to controller."""
        while self._running:
            try:
                await self._send_heartbeat()
            except Exception as e:
                logger.warning(
                    "heartbeat_failed",
                    worker_id=self.worker_id,
                    error=str(e),
                )

            await asyncio.sleep(self.heartbeat_interval)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to controller."""
        if not self._client:
            return

        payload = {
            "worker_id": self.worker_id,
            "current_load": self._current_load,
            "status": "healthy",
        }

        headers = self._get_headers()
        response = await self._client.post(
            f"{self.controller_url}/internal/workers/heartbeat",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        logger.debug(
            "heartbeat_sent",
            worker_id=self.worker_id,
            current_load=self._current_load,
        )

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including auth if configured."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if settings.internal_api_key:
            headers["Authorization"] = f"Bearer {settings.internal_api_key}"
        return headers

    def increment_load(self) -> None:
        """Increment current load counter."""
        self._current_load += 1

    def decrement_load(self) -> None:
        """Decrement current load counter."""
        self._current_load = max(0, self._current_load - 1)

    @property
    def current_load(self) -> int:
        """Get current load."""
        return self._current_load
