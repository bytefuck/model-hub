"""Tests for WorkerRegistry."""

import pytest

from llm_gateway.controller.models import WorkerRecord
from llm_gateway.controller.registry import WorkerRegistry


@pytest.fixture
def registry():
    """Create a fresh registry for each test."""
    return WorkerRegistry()


@pytest.fixture
def sample_worker():
    """Create a sample worker record."""
    return WorkerRecord(
        worker_id="test-worker-001",
        model_id="llama3",
        endpoint="http://localhost:8001",
        capacity=10,
    )


@pytest.mark.asyncio
async def test_register_worker(registry, sample_worker):
    """Test worker registration."""
    await registry.register_worker(sample_worker)

    worker = await registry.get_worker("test-worker-001")
    assert worker is not None
    assert worker.worker_id == "test-worker-001"
    assert worker.model_id == "llama3"


@pytest.mark.asyncio
async def test_register_duplicate_worker(registry, sample_worker):
    """Test that duplicate worker_id raises error."""
    await registry.register_worker(sample_worker)

    with pytest.raises(ValueError, match="already registered"):
        await registry.register_worker(sample_worker)


@pytest.mark.asyncio
async def test_unregister_worker(registry, sample_worker):
    """Test worker unregistration."""
    await registry.register_worker(sample_worker)
    await registry.unregister_worker("test-worker-001", force=True)

    worker = await registry.get_worker("test-worker-001")
    assert worker is None


@pytest.mark.asyncio
async def test_unregister_nonexistent_worker(registry):
    """Test that unregistering nonexistent worker raises error."""
    with pytest.raises(KeyError):
        await registry.unregister_worker("nonexistent", force=True)


@pytest.mark.asyncio
async def test_get_workers_for_model(registry):
    """Test getting workers for a specific model."""
    worker1 = WorkerRecord(
        worker_id="worker-1",
        model_id="llama3",
        endpoint="http://localhost:8001",
    )
    worker2 = WorkerRecord(
        worker_id="worker-2",
        model_id="llama3",
        endpoint="http://localhost:8002",
    )
    worker3 = WorkerRecord(
        worker_id="worker-3",
        model_id="mistral",
        endpoint="http://localhost:8003",
    )

    await registry.register_worker(worker1)
    await registry.register_worker(worker2)
    await registry.register_worker(worker3)

    llama3_workers = await registry.get_workers_for_model("llama3")
    assert len(llama3_workers) == 2

    mistral_workers = await registry.get_workers_for_model("mistral")
    assert len(mistral_workers) == 1

    unknown_workers = await registry.get_workers_for_model("unknown")
    assert len(unknown_workers) == 0


@pytest.mark.asyncio
async def test_update_heartbeat(registry, sample_worker):
    """Test heartbeat update."""
    await registry.register_worker(sample_worker)

    await registry.update_heartbeat(
        worker_id="test-worker-001",
        current_load=5,
        status="healthy",
    )

    worker = await registry.get_worker("test-worker-001")
    assert worker.current_load == 5


@pytest.mark.asyncio
async def test_list_workers(registry):
    """Test listing all workers."""
    worker1 = WorkerRecord(
        worker_id="worker-1",
        model_id="llama3",
        endpoint="http://localhost:8001",
    )
    worker2 = WorkerRecord(
        worker_id="worker-2",
        model_id="mistral",
        endpoint="http://localhost:8002",
    )

    await registry.register_worker(worker1)
    await registry.register_worker(worker2)

    response = await registry.list_workers()
    assert response.total == 2
    assert len(response.workers) == 2


@pytest.mark.asyncio
async def test_list_workers_filter_by_model(registry):
    """Test listing workers filtered by model."""
    worker1 = WorkerRecord(
        worker_id="worker-1",
        model_id="llama3",
        endpoint="http://localhost:8001",
    )
    worker2 = WorkerRecord(
        worker_id="worker-2",
        model_id="mistral",
        endpoint="http://localhost:8002",
    )

    await registry.register_worker(worker1)
    await registry.register_worker(worker2)

    response = await registry.list_workers(model_id="llama3")
    assert response.total == 1
    assert response.workers[0].model_id == "llama3"


@pytest.mark.asyncio
async def test_draining_worker(registry, sample_worker):
    """Test that unregistering without force marks worker as draining."""
    await registry.register_worker(sample_worker)
    await registry.unregister_worker("test-worker-001", force=False)

    worker = await registry.get_worker("test-worker-001")
    assert worker is not None
    assert worker.status == "draining"
