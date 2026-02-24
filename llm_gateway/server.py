"""FastAPI server for LLM Gateway."""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from llm_gateway.adapters import registry
from llm_gateway.config import settings
from llm_gateway.exceptions import LLMGatewayError
from llm_gateway.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ErrorResponse,
    ModelListResponse,
)

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Security(security),
) -> str | None:
    """Verify API token if authentication is enabled."""
    if settings.api_key is None:
        return None

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing authentication token")

    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid authentication token")

    return credentials.credentials


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(settings.log_level),
    )
    logger.info("starting_server", host=settings.host, port=settings.port)
    await registry.initialize()
    logger.info("adapters_initialized", adapters=registry.list_adapters())

    yield

    # Shutdown
    logger.info("shutting_down_server")
    await registry.close_all()


app = FastAPI(
    title="LLM Gateway",
    description="Unified LLM interface proxy with OpenAI-compatible endpoints",
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


@app.get("/v1/models", response_model=ModelListResponse)
async def list_models(
    token: str | None = Security(verify_token),
) -> ModelListResponse:
    """List available models from all providers."""
    logger.info("list_models_request")

    all_models = []
    for adapter_name in registry.list_adapters():
        adapter = registry._adapters.get(adapter_name)
        if adapter:
            try:
                models = await adapter.list_models()
                all_models.extend(models.data)
            except Exception as e:
                logger.warning("adapter_list_models_failed", adapter=adapter_name, error=str(e))

    return ModelListResponse(data=all_models)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    token: str | None = Security(verify_token),
) -> ChatCompletionResponse | StreamingResponse:
    """Create chat completion."""
    request_id = str(uuid.uuid4())
    logger.info(
        "chat_completion_request",
        model=request.model,
        stream=request.stream,
        request_id=request_id,
    )

    try:
        adapter = registry.get_adapter(request.model)

        if request.stream:

            async def generate():
                async for chunk in adapter.chat_completion_stream(request):
                    data = chunk.model_dump_json()
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate(),
                media_type="text/event-stream",
                headers={"X-Request-ID": request_id},
            )

        response = await adapter.chat_completion(request)
        return response

    except LLMGatewayError as e:
        logger.error("gateway_error", error=str(e), status_code=e.status_code)
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        logger.error("unexpected_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/")
async def root() -> dict[str, object]:
    """Root endpoint with basic info."""
    return {
        "name": "LLM Gateway",
        "version": "0.1.0",
        "endpoints": {
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health",
        },
        "providers": registry.list_adapters(),
    }


def main() -> None:
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "llm_gateway.server:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()
