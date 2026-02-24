"""Circuit breaker implementation."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal

import structlog

logger = structlog.get_logger()


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state: Literal["closed", "open", "half_open"] = "closed"
        self.failure_count: int = 0
        self.last_failure: datetime | None = None
        self._last_state_change: datetime = datetime.utcnow()

    def record_success(self) -> None:
        """Record a successful operation."""
        if self.state == "half_open":
            self._transition_to("closed")
            logger.info("circuit_breaker_closed")
        elif self.state == "closed":
            self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure = datetime.utcnow()

        if self.state == "half_open":
            self._transition_to("open")
            logger.warning("circuit_breaker_reopened")
        elif self.state == "closed" and self.failure_count >= self.failure_threshold:
            self._transition_to("open")
            logger.warning(
                "circuit_breaker_opened",
                failure_count=self.failure_count,
                threshold=self.failure_threshold,
            )

    def is_available(self) -> bool:
        """Check if the circuit allows requests."""
        if self.state == "closed":
            return True

        if self.state == "open":
            if self._should_attempt_recovery():
                self._transition_to("half_open")
                logger.info("circuit_breaker_half_open")
                return True
            return False

        return True

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure is None:
            return True
        elapsed = datetime.utcnow() - self.last_failure
        return elapsed >= timedelta(seconds=self.recovery_timeout)

    def _transition_to(self, new_state: Literal["closed", "open", "half_open"]) -> None:
        """Transition to a new state."""
        self.state = new_state
        self._last_state_change = datetime.utcnow()
        if new_state == "closed":
            self.failure_count = 0
            self.last_failure = None

    def reset(self) -> None:
        """Reset the circuit breaker to closed state."""
        self._transition_to("closed")
