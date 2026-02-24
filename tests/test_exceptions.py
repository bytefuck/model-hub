"""Tests for LLM Gateway exceptions."""

from __future__ import annotations

import pytest

from llm_gateway.exceptions import (
    AuthenticationError,
    ConfigurationError,
    LLMGatewayError,
    ModelNotFoundError,
    ProviderError,
    RateLimitError,
    ValidationError,
)


def test_llm_gateway_error_base() -> None:
    """Test base LLMGatewayError."""
    error = LLMGatewayError("Something went wrong", status_code=500)
    assert str(error) == "Something went wrong"
    assert error.status_code == 500
    assert error.message == "Something went wrong"


def test_authentication_error() -> None:
    """Test AuthenticationError."""
    error = AuthenticationError()
    assert error.status_code == 401
    assert "Authentication failed" in error.message

    error_custom = AuthenticationError("Custom auth error")
    assert error_custom.message == "Custom auth error"


def test_configuration_error() -> None:
    """Test ConfigurationError."""
    error = ConfigurationError("Invalid config")
    assert error.status_code == 500
    assert error.message == "Invalid config"


def test_provider_error() -> None:
    """Test ProviderError."""
    error = ProviderError("API unavailable", status_code=503)
    assert error.status_code == 503
    assert error.message == "API unavailable"


def test_model_not_found_error() -> None:
    """Test ModelNotFoundError."""
    error = ModelNotFoundError("gpt-99")
    assert error.status_code == 404
    assert "gpt-99" in error.message


def test_rate_limit_error() -> None:
    """Test RateLimitError."""
    error = RateLimitError()
    assert error.status_code == 429
    assert "Rate limit exceeded" in error.message


def test_validation_error() -> None:
    """Test ValidationError."""
    error = ValidationError("Invalid request")
    assert error.status_code == 400
    assert error.message == "Invalid request"
