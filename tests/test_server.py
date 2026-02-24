"""Tests for LLM Gateway server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from llm_gateway.server import app


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


def test_health_check(client: TestClient) -> None:
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root_endpoint(client: TestClient) -> None:
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "LLM Gateway"
    assert "endpoints" in data


def test_list_models(client: TestClient) -> None:
    """Test list models endpoint."""
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert "data" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_chat_completion_validation() -> None:
    """Test chat completion request validation."""
    from httpx import ASGITransport

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Test missing required field
        response = await ac.post("/v1/chat/completions", json={})
        assert response.status_code == 422

        # Test invalid model type
        response = await ac.post(
            "/v1/chat/completions",
            json={"model": "gpt-4", "messages": "invalid"},
        )
        assert response.status_code == 422
