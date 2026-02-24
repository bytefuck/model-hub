"""Controller models for worker management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


@dataclass
class WorkerRecord:
    """Represents a registered worker in the registry."""

    worker_id: str
    model_id: str
    endpoint: str
    status: Literal["healthy", "unhealthy", "draining"] = "healthy"
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    capacity: int = 10
    current_load: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Circuit breaker state
    circuit_state: Literal["closed", "open", "half_open"] = "closed"
    failure_count: int = 0
    last_failure: datetime | None = None

    @property
    def load_ratio(self) -> float:
        """Return current load as ratio of capacity."""
        if self.capacity <= 0:
            return float("inf")
        return self.current_load / self.capacity

    @property
    def is_available(self) -> bool:
        """Check if worker is available for routing."""
        return self.status == "healthy" and self.circuit_state != "open"


class WorkerRegisterRequest(BaseModel):
    """Request body for worker registration."""

    worker_id: str
    model_id: str
    endpoint: str
    capacity: int = 10
    metadata: dict[str, Any] = {}


class WorkerHeartbeatRequest(BaseModel):
    """Request body for worker heartbeat."""

    worker_id: str
    current_load: int
    status: Literal["healthy", "unhealthy", "draining"] = "healthy"


class WorkerInfo(BaseModel):
    """Worker information in list response."""

    worker_id: str
    model_id: str
    endpoint: str
    status: str
    current_load: int
    capacity: int
    circuit_state: str
    last_heartbeat: datetime | None = None


class WorkerListResponse(BaseModel):
    """Response for listing workers."""

    workers: list[WorkerInfo]
    total: int
