"""Request router for controller."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from llm_gateway.controller.circuit_breaker import CircuitBreaker
from llm_gateway.exceptions import LLMGatewayError

if TYPE_CHECKING:
    from llm_gateway.controller.registry import WorkerRegistry
    from llm_gateway.controller.models import WorkerRecord

logger = structlog.get_logger()


class NoWorkerAvailableError(LLMGatewayError):
    """No worker available for the requested model."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"No worker available for model: {model_id}",
            status_code=404,
        )


class AllWorkersAtCapacityError(LLMGatewayError):
    """All workers are at full capacity."""

    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"All workers for model {model_id} are at full capacity",
            status_code=503,
        )


class Router:
    """Routes requests to appropriate workers."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry
        self._circuit_breakers: dict[str, CircuitBreaker] = {}

    def _get_circuit_breaker(self, worker_id: str) -> CircuitBreaker:
        """Get or create circuit breaker for a worker."""
        if worker_id not in self._circuit_breakers:
            self._circuit_breakers[worker_id] = CircuitBreaker()
        return self._circuit_breakers[worker_id]

    async def select_worker(self, model_id: str) -> WorkerRecord:
        """Select the best worker for a model using Least-Loaded strategy.

        Raises:
            NoWorkerAvailableError: No workers registered for the model
            AllWorkersAtCapacityError: All workers at full capacity
        """
        workers = await self._registry.get_workers_for_model(model_id)

        if not workers:
            raise NoWorkerAvailableError(model_id)

        available_workers = []
        for worker in workers:
            cb = self._get_circuit_breaker(worker.worker_id)
            if cb.is_available() and worker.status == "healthy":
                available_workers.append(worker)

        if not available_workers:
            raise NoWorkerAvailableError(model_id)

        at_capacity = [w for w in available_workers if w.current_load < w.capacity]

        if not at_capacity:
            raise AllWorkersAtCapacityError(model_id)

        selected = min(at_capacity, key=lambda w: w.load_ratio)
        logger.debug(
            "worker_selected",
            worker_id=selected.worker_id,
            model_id=model_id,
            load_ratio=selected.load_ratio,
        )
        return selected

    def record_success(self, worker_id: str) -> None:
        """Record a successful request to a worker."""
        cb = self._get_circuit_breaker(worker_id)
        cb.record_success()

    def record_failure(self, worker_id: str) -> None:
        """Record a failed request to a worker."""
        cb = self._get_circuit_breaker(worker_id)
        cb.record_failure()
