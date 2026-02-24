"""Tests for Router."""

import pytest

from llm_gateway.controller.models import WorkerRecord
from llm_gateway.controller.registry import WorkerRegistry
from llm_gateway.controller.router import (
    AllWorkersAtCapacityError,
    NoWorkerAvailableError,
    Router,
)


@pytest.fixture
def registry():
    """Create a fresh registry."""
    return WorkerRegistry()


@pytest.fixture
def router(registry):
    """Create a router with the registry."""
    return Router(registry)


@pytest.mark.asyncio
async def test_select_worker_no_workers(router):
    """Test that selecting worker with no workers raises error."""
    with pytest.raises(NoWorkerAvailableError):
        await router.select_worker("llama3")


@pytest.mark.asyncio
async def test_select_worker_least_loaded(router, registry):
    """Test that router selects least loaded worker."""
    worker1 = WorkerRecord(
        worker_id="worker-1",
        model_id="llama3",
        endpoint="http://localhost:8001",
        capacity=10,
        current_load=5,
    )
    worker2 = WorkerRecord(
        worker_id="worker-2",
        model_id="llama3",
        endpoint="http://localhost:8002",
        capacity=10,
        current_load=2,
    )

    await registry.register_worker(worker1)
    await registry.register_worker(worker2)

    selected = await router.select_worker("llama3")
    assert selected.worker_id == "worker-2"


@pytest.mark.asyncio
async def test_select_worker_all_at_capacity(router, registry):
    """Test that router raises error when all workers at capacity."""
    worker1 = WorkerRecord(
        worker_id="worker-1",
        model_id="llama3",
        endpoint="http://localhost:8001",
        capacity=10,
        current_load=10,
    )

    await registry.register_worker(worker1)

    with pytest.raises(AllWorkersAtCapacityError):
        await router.select_worker("llama3")


@pytest.mark.asyncio
async def test_select_worker_respects_circuit_breaker(router, registry):
    """Test that router doesn't select workers with open circuit."""
    worker1 = WorkerRecord(
        worker_id="worker-1",
        model_id="llama3",
        endpoint="http://localhost:8001",
        capacity=10,
        current_load=0,
    )
    worker2 = WorkerRecord(
        worker_id="worker-2",
        model_id="llama3",
        endpoint="http://localhost:8002",
        capacity=10,
        current_load=0,
    )

    await registry.register_worker(worker1)
    await registry.register_worker(worker2)

    for _ in range(5):
        router.record_failure("worker-1")

    selected = await router.select_worker("llama3")
    assert selected.worker_id == "worker-2"


@pytest.mark.asyncio
async def test_record_success_and_failure(router):
    """Test recording success and failure."""
    router.record_success("any-worker")
    router.record_failure("any-worker")

    cb = router._get_circuit_breaker("any-worker")
    assert cb.failure_count == 1
