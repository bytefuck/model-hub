"""LLM Gateway configuration."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Security
    api_key: str | None = Field(default=None, description="API key for authentication")

    # Provider settings
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI base URL",
    )

    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        description="Anthropic base URL",
    )

    ollama_host: str = Field(
        default="http://localhost:11434",
        description="Ollama host URL",
    )

    azure_openai_api_key: str | None = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: str | None = Field(default=None, description="Azure OpenAI endpoint")

    # Request settings
    default_timeout: int = Field(default=60, description="Default request timeout in seconds")
    max_tokens: int = Field(default=4096, description="Default max tokens")

    # Controller settings
    controller_host: str = Field(default="0.0.0.0", description="Controller server host")
    controller_port: int = Field(default=8000, description="Controller server port")
    internal_api_key: str | None = Field(default=None, description="API key for internal endpoints")
    heartbeat_timeout: int = Field(default=60, description="Seconds before worker considered stale")
    heartbeat_check_interval: int = Field(
        default=10, description="Interval to check worker heartbeats"
    )

    # Worker settings
    worker_id: str | None = Field(default=None, description="Unique worker identifier")
    model_id: str | None = Field(default=None, description="Model this worker serves")
    controller_url: str = Field(default="http://localhost:8000", description="Controller URL")
    backend_url: str | None = Field(default=None, description="Backend model service URL")
    listen_port: int = Field(default=8001, description="Worker listen port")
    heartbeat_interval: int = Field(default=10, description="Heartbeat interval in seconds")
    capacity: int = Field(default=10, description="Maximum concurrent requests")
    registry_retry_count: int = Field(default=30, description="Registration retry attempts")
    registry_retry_delay: int = Field(default=5, description="Initial retry delay in seconds")

    @property
    def configured_providers(self) -> list[str]:
        """Return list of configured providers."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.ollama_host:
            providers.append("ollama")
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            providers.append("azure")
        return providers


# Global settings instance
settings = Settings()
