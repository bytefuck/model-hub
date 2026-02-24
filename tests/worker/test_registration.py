"""Tests for RegistrationClient."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llm_gateway.worker.registration import RegistrationClient


@pytest.fixture
def registration_client():
    """Create a registration client."""
    return RegistrationClient(
        worker_id="test-worker-001",
        model_id="llama3",
        controller_url="http://localhost:8000",
        backend_url="http://localhost:11434",
        capacity=10,
        heartbeat_interval=1,
        retry_count=3,
        retry_delay=0.1,
    )


@pytest.mark.asyncio
async def test_start_registers_with_controller(registration_client):
    """Test that start registers with controller."""
    with patch("httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client.post.return_value = MagicMock(status_code=201)
        mock_client_class.return_value = mock_client

        await registration_client.start()

        mock_client.post.assert_called_once()
        call_args = mock_client.post.call_args
        assert "/internal/workers/register" in call_args[0][0]
        assert call_args[1]["json"]["worker_id"] == "test-worker-001"

        await registration_client.stop()


@pytest.mark.asyncio
async def test_increment_decrement_load(registration_client):
    """Test load tracking."""
    assert registration_client.current_load == 0

    registration_client.increment_load()
    assert registration_client.current_load == 1

    registration_client.increment_load()
    assert registration_client.current_load == 2

    registration_client.decrement_load()
    assert registration_client.current_load == 1


@pytest.mark.asyncio
async def test_load_never_negative(registration_client):
    """Test that load never goes negative."""
    registration_client.decrement_load()
    assert registration_client.current_load == 0

    registration_client.decrement_load()
    assert registration_client.current_load == 0
