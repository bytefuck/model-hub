"""Tests for LLM Gateway configuration."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from llm_gateway.config import Settings


def test_settings_defaults() -> None:
    """Test Settings default values."""
    settings = Settings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.openai_base_url == "https://api.openai.com/v1"
    assert settings.ollama_host == "http://localhost:11434"


def test_settings_from_env() -> None:
    """Test Settings from environment variables."""
    with patch.dict(
        "os.environ",
        {
            "HOST": "127.0.0.1",
            "PORT": "9000",
            "LOG_LEVEL": "DEBUG",
            "OPENAI_API_KEY": "test-key",
        },
    ):
        settings = Settings()
        assert settings.host == "127.0.0.1"
        assert settings.port == 9000
        assert settings.log_level == "DEBUG"
        assert settings.openai_api_key == "test-key"


def test_configured_providers() -> None:
    """Test configured providers detection."""
    settings = Settings()
    # Ollama is always considered configured (has default host)
    assert "ollama" in settings.configured_providers

    with patch.dict("os.environ", {"OPENAI_API_KEY": "test"}):
        settings = Settings()
        assert "openai" in settings.configured_providers
        assert "ollama" in settings.configured_providers

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test"}):
        settings = Settings()
        assert "anthropic" in settings.configured_providers
        assert "ollama" in settings.configured_providers
