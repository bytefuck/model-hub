"""Worker server for LLM Gateway."""

from __future__ import annotations

import asyncio
import signal
import sys
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from llm_gateway.config import settings
from llm_gateway.worker.proxy import ProxyHandler
from llm_gateway.worker.registration import RegistrationClient

logger = structlog.get_logger()

registration_client: RegistrationClient
proxy_handler: ProxyHandler
shutdown_event: asyncio.Event


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global registration_client, proxy_handler, shutdown_event

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(settings.log_level),
    )
    logger.info(
        "worker_starting",
        worker_id=settings.worker_id,
        model_id=settings.model_id,
        port=settings.listen_port,
    )

    if not settings.worker_id or not settings.model_id or not settings.backend_url:
        raise RuntimeError("WORKER_ID, MODEL_ID, and BACKEND_URL must be set for worker mode")

    registration_client = RegistrationClient(
        worker_id=settings.worker_id,
        model_id=settings.model_id,
        controller_url=settings.controller_url,
        backend_url=settings.backend_url,
        capacity=settings.capacity,
        heartbeat_interval=settings.heartbeat_interval,
        retry_count=settings.registry_retry_count,
        retry_delay=settings.registry_retry_delay,
    )

    proxy_handler = ProxyHandler(
        backend_url=settings.backend_url,
        registration_client=registration_client,
        timeout=settings.default_timeout,
    )

    shutdown_event = asyncio.Event()

    await proxy_handler.start()
    await registration_client.start()

    loop = asyncio.get_event_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.create_task(_handle_shutdown()),
            )

    yield

    logger.info("worker_shutting_down")
    shutdown_event.set()
    await registration_client.stop()
    await proxy_handler.stop()


async def _handle_shutdown():
    """Handle shutdown signal gracefully."""
    logger.info("shutdown_signal_received")
    shutdown_event.set()


app = FastAPI(
    title="LLM Gateway Worker",
    description="Worker for processing LLM requests",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to each request."""
    request_id = str(uuid.uuid4())
    structlog.contextvars.bind_contextvars(request_id=request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    """Proxy chat completion request to backend."""
    payload = await request.json()
    stream = payload.get("stream", False)

    request_id = str(uuid.uuid4())
    logger.info(
        "chat_completion_request",
        model=payload.get("model"),
        stream=stream,
        request_id=request_id,
    )

    try:
        if stream:
            return StreamingResponse(
                proxy_handler.proxy_chat_completion_stream(payload),
                media_type="text/event-stream",
                headers={"X-Request-ID": request_id},
            )

        return await proxy_handler.proxy_chat_completion(payload)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_completion_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    healthy, reason = await proxy_handler.check_backend_health()

    if healthy:
        return {"status": "healthy"}
    else:
        return {"status": "unhealthy", "reason": reason}


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "LLM Gateway Worker",
        "version": "0.1.0",
        "worker_id": settings.worker_id,
        "model_id": settings.model_id,
        "current_load": registration_client.current_load,
        "capacity": settings.capacity,
    }
