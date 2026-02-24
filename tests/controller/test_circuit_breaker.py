"""Tests for CircuitBreaker."""

import asyncio

import pytest

from llm_gateway.controller.circuit_breaker import CircuitBreaker


@pytest.fixture
def circuit_breaker():
    """Create a circuit breaker with default settings."""
    return CircuitBreaker(failure_threshold=3, recovery_timeout=1)


def test_initial_state(circuit_breaker):
    """Test initial state is closed."""
    assert circuit_breaker.state == "closed"
    assert circuit_breaker.is_available() is True


def test_record_success_resets_failure_count(circuit_breaker):
    """Test that success resets failure count."""
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker.failure_count == 2

    circuit_breaker.record_success()
    assert circuit_breaker.failure_count == 0


def test_circuit_opens_after_threshold(circuit_breaker):
    """Test that circuit opens after failure threshold."""
    assert circuit_breaker.state == "closed"

    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker.state == "closed"

    circuit_breaker.record_failure()
    assert circuit_breaker.state == "open"
    assert circuit_breaker.is_available() is False


def test_circuit_transitions_to_half_open_after_timeout(circuit_breaker):
    """Test that circuit transitions to half_open after timeout."""
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker.state == "open"

    asyncio.run(asyncio.sleep(1.1))

    assert circuit_breaker.is_available() is True
    assert circuit_breaker.state == "half_open"


def test_circuit_closes_on_success_in_half_open(circuit_breaker):
    """Test that circuit closes on success in half_open state."""
    circuit_breaker.state = "half_open"

    circuit_breaker.record_success()

    assert circuit_breaker.state == "closed"
    assert circuit_breaker.failure_count == 0


def test_circuit_reopens_on_failure_in_half_open(circuit_breaker):
    """Test that circuit reopens on failure in half_open state."""
    circuit_breaker.state = "half_open"

    circuit_breaker.record_failure()

    assert circuit_breaker.state == "open"


def test_reset(circuit_breaker):
    """Test that reset returns to closed state."""
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    circuit_breaker.record_failure()
    assert circuit_breaker.state == "open"

    circuit_breaker.reset()

    assert circuit_breaker.state == "closed"
    assert circuit_breaker.failure_count == 0
