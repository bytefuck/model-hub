"""Pydantic models for LLM Gateway."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant"] = Field(
        description="Role of the message sender",
    )
    content: str = Field(description="Content of the message")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(description="Model to use for completion")
    messages: list[Message] = Field(description="List of messages")
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = Field(default=None, ge=1)
    top_p: float | None = Field(default=1.0, ge=0, le=1)
    stream: bool = Field(default=False)
    stop: list[str] | str | None = Field(default=None)


class Choice(BaseModel):
    """Completion choice."""

    index: int = Field(default=0)
    message: Message = Field(...)
    finish_reason: str | None = Field(default="stop")


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(...)
    object: str = Field(default="chat.completion")
    created: int = Field(...)
    model: str = Field(...)
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)


class StreamChoice(BaseModel):
    """Streaming completion choice."""

    index: int = Field(default=0)
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = Field(default=None)


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response chunk."""

    id: str = Field(...)
    object: str = Field(default="chat.completion.chunk")
    created: int = Field(...)
    model: str = Field(...)
    choices: list[StreamChoice] = Field(default_factory=list)


class ModelInfo(BaseModel):
    """Model information."""

    id: str = Field(...)
    object: str = Field(default="model")
    created: int = Field(default=0)
    owned_by: str = Field(default="llm-gateway")


class ModelListResponse(BaseModel):
    """List of available models."""

    object: str = Field(default="list")
    data: list[ModelInfo] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Error response."""

    error: dict[str, Any] = Field(...)
