"""Custom exceptions for LLM Gateway."""

from __future__ import annotations


class LLMGatewayError(Exception):
    """Base exception for LLM Gateway."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ConfigurationError(LLMGatewayError):
    """Raised when there's a configuration error."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=500)


class AuthenticationError(LLMGatewayError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed") -> None:
        super().__init__(message, status_code=401)


class ProviderError(LLMGatewayError):
    """Raised when a provider API call fails."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message, status_code=status_code)


class ModelNotFoundError(LLMGatewayError):
    """Raised when a model is not found."""

    def __init__(self, model: str) -> None:
        super().__init__(f"Model '{model}' not found", status_code=404)


class RateLimitError(LLMGatewayError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, status_code=429)


class ValidationError(LLMGatewayError):
    """Raised when request validation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status_code=400)
