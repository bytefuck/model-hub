"""LLM Gateway main module."""

from __future__ import annotations

from llm_gateway.config import settings
from llm_gateway.server import app, main

__version__ = "0.1.0"
__all__ = ["app", "main", "settings"]
