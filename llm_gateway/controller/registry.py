"""Worker registry for controller."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING, Literal, cast

import structlog

from llm_gateway.controller.models import (
    WorkerInfo,
    WorkerListResponse,
    WorkerRecord,
)

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class WorkerRegistry:
    """Registry for managing worker records."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._by_model: dict[str, dict[str, WorkerRecord]] = {}
        self._by_id: dict[str, WorkerRecord] = {}

    async def register_worker(self, record: WorkerRecord) -> None:
        """Register a new worker."""
        async with self._lock:
            if record.worker_id in self._by_id:
                raise ValueError(f"Worker {record.worker_id} already registered")

            self._by_id[record.worker_id] = record

            if record.model_id not in self._by_model:
                self._by_model[record.model_id] = {}
            self._by_model[record.model_id][record.worker_id] = record

            logger.info(
                "worker_registered",
                worker_id=record.worker_id,
                model_id=record.model_id,
                endpoint=record.endpoint,
            )

    async def unregister_worker(self, worker_id: str, force: bool = False) -> None:
        """Unregister a worker.

        Args:
            worker_id: Worker to unregister
            force: If True, immediately remove; if False, mark as draining
        """
        async with self._lock:
            record = self._by_id.get(worker_id)
            if not record:
                raise KeyError(f"Worker {worker_id} not found")

            if force:
                del self._by_id[worker_id]
                if record.model_id in self._by_model:
                    self._by_model[record.model_id].pop(worker_id, None)
                    if not self._by_model[record.model_id]:
                        del self._by_model[record.model_id]
                logger.info("worker_removed", worker_id=worker_id)
            else:
                record.status = "draining"
                logger.info("worker_draining", worker_id=worker_id)

    async def get_workers_for_model(self, model_id: str) -> list[WorkerRecord]:
        """Get all workers for a specific model."""
        async with self._lock:
            workers = self._by_model.get(model_id, {})
            return list(workers.values())

    async def update_heartbeat(
        self,
        worker_id: str,
        current_load: int,
        status: Literal["healthy", "unhealthy", "draining"],
    ) -> None:
        """Update worker heartbeat."""
        async with self._lock:
            record = self._by_id.get(worker_id)
            if not record:
                raise KeyError(f"Worker {worker_id} not found")

            record.last_heartbeat = datetime.utcnow()
            record.current_load = current_load
            record.status = cast(Literal["healthy", "unhealthy", "draining"], status)
            logger.debug(
                "heartbeat_updated",
                worker_id=worker_id,
                current_load=current_load,
                status=status,
            )

    async def list_workers(self, model_id: str | None = None) -> WorkerListResponse:
        """List all workers, optionally filtered by model."""
        async with self._lock:
            if model_id:
                workers = self._by_model.get(model_id, {}).values()
            else:
                workers = self._by_id.values()

            worker_infos = [
                WorkerInfo(
                    worker_id=w.worker_id,
                    model_id=w.model_id,
                    endpoint=w.endpoint,
                    status=w.status,
                    current_load=w.current_load,
                    capacity=w.capacity,
                    circuit_state=w.circuit_state,
                    last_heartbeat=w.last_heartbeat,
                )
                for w in workers
            ]

            return WorkerListResponse(
                workers=worker_infos,
                total=len(worker_infos),
            )

    async def get_worker(self, worker_id: str) -> WorkerRecord | None:
        """Get a specific worker by ID."""
        async with self._lock:
            return self._by_id.get(worker_id)

    async def mark_unhealthy(self, worker_id: str) -> None:
        """Mark a worker as unhealthy."""
        async with self._lock:
            record = self._by_id.get(worker_id)
            if record:
                record.status = cast(Literal["healthy", "unhealthy", "draining"], "unhealthy")
                logger.warning("worker_marked_unhealthy", worker_id=worker_id)

    async def remove_worker(self, worker_id: str) -> None:
        """Remove a worker from registry completely."""
        async with self._lock:
            record = self._by_id.pop(worker_id, None)
            if record and record.model_id in self._by_model:
                self._by_model[record.model_id].pop(worker_id, None)
                if not self._by_model[record.model_id]:
                    del self._by_model[record.model_id]
            logger.info("worker_removed_from_registry", worker_id=worker_id)
