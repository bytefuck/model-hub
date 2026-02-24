"""Health checker for workers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import structlog

if TYPE_CHECKING:
    from llm_gateway.controller.registry import WorkerRegistry

logger = structlog.get_logger()


class HealthChecker:
    """Background task to check worker health."""

    def __init__(
        self,
        registry: WorkerRegistry,
        heartbeat_timeout: int = 60,
        check_interval: int = 10,
        probe_failures_threshold: int = 3,
    ) -> None:
        self._registry = registry
        self._heartbeat_timeout = heartbeat_timeout
        self._check_interval = check_interval
        self._probe_failures_threshold = probe_failures_threshold
        self._probe_failures: dict[str, int] = {}
        self._running = False
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Start the health check background task."""
        self._running = True
        self._client = httpx.AsyncClient(timeout=5.0)
        self._task = asyncio.create_task(self._check_loop())
        logger.info("health_checker_started")

    async def stop(self) -> None:
        """Stop the health check background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._client:
            await self._client.aclose()
        logger.info("health_checker_stopped")

    async def _check_loop(self) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._check_workers()
            except Exception as e:
                logger.error("health_check_error", error=str(e))

            await asyncio.sleep(self._check_interval)

    async def _check_workers(self) -> None:
        """Check all registered workers for heartbeat timeout."""
        response = await self._registry.list_workers()

        for worker_info in response.workers:
            if worker_info.status == "draining":
                continue

            if worker_info.last_heartbeat is None:
                continue

            elapsed = datetime.utcnow() - worker_info.last_heartbeat
            if elapsed > timedelta(seconds=self._heartbeat_timeout):
                await self._handle_timeout(worker_info.worker_id, worker_info.endpoint)

    async def _handle_timeout(self, worker_id: str, endpoint: str) -> None:
        """Handle a worker that has timed out."""
        logger.warning("worker_heartbeat_timeout", worker_id=worker_id)

        healthy = await self._probe_worker(endpoint)

        if healthy:
            await self._registry.mark_unhealthy(worker_id)
            self._probe_failures.pop(worker_id, None)
            logger.info(
                "worker_restored_after_probe",
                worker_id=worker_id,
            )
        else:
            failures = self._probe_failures.get(worker_id, 0) + 1
            self._probe_failures[worker_id] = failures

            if failures >= self._probe_failures_threshold:
                await self._registry.remove_worker(worker_id)
                self._probe_failures.pop(worker_id, None)
                logger.warning(
                    "worker_removed_after_failed_probes",
                    worker_id=worker_id,
                    failures=failures,
                )
            else:
                await self._registry.mark_unhealthy(worker_id)
                logger.warning(
                    "worker_probe_failed",
                    worker_id=worker_id,
                    failures=failures,
                )

    async def _probe_worker(self, endpoint: str) -> bool:
        """Probe a worker's health endpoint."""
        if not self._client:
            return False

        try:
            response = await self._client.get(f"{endpoint}/health")
            return response.status_code == 200
        except Exception as e:
            logger.debug("worker_probe_failed", endpoint=endpoint, error=str(e))
            return False
