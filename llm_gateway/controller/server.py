"""Controller server for LLM Gateway."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
import structlog
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_gateway.config import settings
from llm_gateway.controller.health import HealthChecker
from llm_gateway.controller.models import (
    WorkerHeartbeatRequest,
    WorkerInfo,
    WorkerListResponse,
    WorkerRecord,
    WorkerRegisterRequest,
)
from llm_gateway.controller.registry import WorkerRegistry
from llm_gateway.controller.router import (
    AllWorkersAtCapacityError,
    NoWorkerAvailableError,
    Router,
)
from llm_gateway.exceptions import LLMGatewayError
from llm_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelListResponse,
)

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)

registry: WorkerRegistry
router: Router
health_checker: HealthChecker
client: httpx.AsyncClient


async def verify_internal_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str | None:
    """Verify internal API key if configured."""
    if settings.internal_api_key is None:
        return None

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    if credentials.credentials != settings.internal_api_key:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global registry, router, health_checker, client

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(settings.log_level),
    )
    logger.info(
        "controller_starting",
        host=settings.controller_host,
        port=settings.controller_port,
    )

    registry = WorkerRegistry()
    router = Router(registry)
    health_checker = HealthChecker(
        registry=registry,
        heartbeat_timeout=settings.heartbeat_timeout,
        check_interval=settings.heartbeat_check_interval,
    )
    client = httpx.AsyncClient(timeout=settings.default_timeout)

    await health_checker.start()
    logger.info("health_checker_started")

    yield

    logger.info("controller_shutting_down")
    await health_checker.stop()
    await client.aclose()


app = FastAPI(
    title="LLM Gateway Controller",
    description="Controller for managing workers and routing requests",
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


# ============================================================================
# Internal API Endpoints
# ============================================================================


@app.post("/internal/workers/register", status_code=201)
async def register_worker(
    request: WorkerRegisterRequest,
    _: str | None = Depends(verify_internal_api_key),
) -> dict[str, Any]:
    """Register a new worker."""
    logger.info(
        "worker_registration_request",
        worker_id=request.worker_id,
        model_id=request.model_id,
    )

    record = WorkerRecord(
        worker_id=request.worker_id,
        model_id=request.model_id,
        endpoint=request.endpoint,
        capacity=request.capacity,
        metadata=request.metadata,
    )

    try:
        await registry.register_worker(record)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    return {
        "worker_id": request.worker_id,
        "status": "registered",
    }


@app.post("/internal/workers/heartbeat")
async def worker_heartbeat(
    request: WorkerHeartbeatRequest,
    _: str | None = Depends(verify_internal_api_key),
) -> dict[str, str]:
    """Update worker heartbeat."""
    try:
        await registry.update_heartbeat(
            worker_id=request.worker_id,
            current_load=request.current_load,
            status=request.status,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Worker {request.worker_id} not found")

    return {"status": "ok"}


@app.get("/internal/workers", response_model=WorkerListResponse)
async def list_workers(
    model_id: str | None = Query(default=None),
    _: str | None = Depends(verify_internal_api_key),
) -> WorkerListResponse:
    """List all registered workers."""
    return await registry.list_workers(model_id=model_id)


@app.delete("/internal/workers/{worker_id}")
async def deregister_worker(
    worker_id: str,
    force: bool = Query(default=False),
    _: str | None = Depends(verify_internal_api_key),
) -> dict[str, str]:
    """Deregister a worker."""
    try:
        await registry.unregister_worker(worker_id, force=force)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Worker {worker_id} not found")

    return {"status": "ok"}


# ============================================================================
# Public API Endpoints
# ============================================================================


@app.post("/v1/chat/completions",response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | StreamingResponse:
    """Create chat completion by routing to a worker."""
    request_id = str(uuid.uuid4())
    logger.info(
        "chat_completion_request",
        model=request.model,
        stream=request.stream,
        request_id=request_id,
    )

    try:
        worker = await router.select_worker(request.model)
    except NoWorkerAvailableError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except AllWorkersAtCapacityError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)

    logger.info(
        "routing_to_worker",
        worker_id=worker.worker_id,
        endpoint=worker.endpoint,
    )

    try:
        url = f"{worker.endpoint}/v1/chat/completions"

        if request.stream:
            return StreamingResponse(
                _proxy_stream(url, request.model_dump(), worker.worker_id),
                media_type="text/event-stream",
                headers={"X-Request-ID": request_id},
            )

        response = await client.post(url, json=request.model_dump())
        response.raise_for_status()
        router.record_success(worker.worker_id)
        return response.json()

    except httpx.HTTPStatusError as e:
        router.record_failure(worker.worker_id)
        raise HTTPException(
            status_code=e.response.status_code,
            detail=str(e.response.text),
        )
    except Exception as e:
        router.record_failure(worker.worker_id)
        logger.error("worker_request_failed", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))


async def _proxy_stream(url: str, payload: dict, worker_id: str):
    """Proxy streaming response from worker."""
    try:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk
        router.record_success(worker_id)
    except Exception as e:
        router.record_failure(worker_id)
        logger.error("stream_proxy_failed", worker_id=worker_id, error=str(e))
        raise


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """List all available models from workers."""
    workers_response = await registry.list_workers()
    models: dict[str, ModelInfo] = {}

    for worker_info in workers_response.workers:
        if worker_info.model_id not in models:
            models[worker_info.model_id] = ModelInfo(
                id=worker_info.model_id,
                owned_by="llm-gateway",
            )

    return ModelListResponse(data=list(models.values()))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "name": "LLM Gateway Controller",
        "version": "0.1.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
            "workers": "/internal/workers",
        },
    }

def main() -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "llm_gateway.controller.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()