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

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging level",
    )

    # Request settings
    default_timeout: int = Field(
        default=60, description="Default request timeout in seconds"
    )
    max_tokens: int = Field(default=4096, description="Default max tokens")

    # Controller settings
    controller_host: str = Field(
        default="0.0.0.0", description="Controller server host"
    )
    controller_port: int = Field(default=8000, description="Controller server port")
    internal_api_key: str | None = Field(
        default=None, description="API key for internal endpoints"
    )
    heartbeat_timeout: int = Field(
        default=60, description="Seconds before worker considered stale"
    )
    heartbeat_check_interval: int = Field(
        default=10, description="Interval to check worker heartbeats"
    )

    # Worker settings
    worker_id: str | None = Field(default=None, description="Unique worker identifier")
    model_id: str | None = Field(default=None, description="Model this worker serves")
    controller_url: str = Field(
        default="http://localhost:8000", description="Controller URL"
    )
    backend_url: str | None = Field(
        default=None, description="Backend model service URL"
    )
    listen_port: int = Field(default=8001, description="Worker listen port")
    heartbeat_interval: int = Field(
        default=10, description="Heartbeat interval in seconds"
    )
    capacity: int = Field(default=10, description="Maximum concurrent requests")
    registry_retry_count: int = Field(
        default=30, description="Registration retry attempts"
    )
    registry_retry_delay: int = Field(
        default=5, description="Initial retry delay in seconds"
    )


# Global settings instance
settings = Settings()
