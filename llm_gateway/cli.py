"""CLI entry points for LLM Gateway."""

from __future__ import annotations

import typer
import uvicorn

from llm_gateway.config import settings

app = typer.Typer(name="llm-gateway", help="LLM Gateway - Unified LLM interface proxy")


@app.command()
def controller(
    host: str | None = typer.Option(None, "--host", "-h", help="Host to bind to"),
    port: int | None = typer.Option(None, "--port", "-p", help="Port to bind to"),
    log_level: str | None = typer.Option(None, "--log-level", help="Log level"),
) -> None:
    """Start the controller server."""
    host = host or settings.controller_host
    port = port or settings.controller_port
    log_level = log_level or settings.log_level.lower()

    typer.echo(f"Starting controller on {host}:{port}")
    uvicorn.run(
        "llm_gateway.controller.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


@app.command()
def worker(
    worker_id: str | None = typer.Option(None, "--worker-id", "-w", help="Worker ID"),
    model_id: str | None = typer.Option(None, "--model-id", "-m", help="Model ID"),
    controller_url: str | None = typer.Option(
        None, "--controller-url", "-c", help="Controller URL"
    ),
    backend_url: str | None = typer.Option(
        None, "--backend-url", "-b", help="Backend URL"
    ),
    port: int | None = typer.Option(None, "--port", "-p", help="Port to bind to"),
    capacity: int | None = typer.Option(
        None, "--capacity", help="Max concurrent requests"
    ),
    log_level: str | None = typer.Option(None, "--log-level", help="Log level"),
) -> None:
    """Start a worker server."""
    if worker_id:
        settings.worker_id = worker_id
    if model_id:
        settings.model_id = model_id
    if controller_url:
        settings.controller_url = controller_url
    if backend_url:
        settings.backend_url = backend_url
    if capacity:
        settings.capacity = capacity

    host = "0.0.0.0"
    port = port or settings.listen_port
    log_level = log_level or settings.log_level.lower()

    if not settings.worker_id:
        typer.echo("Error: --worker-id or WORKER_ID environment variable is required")
        raise typer.Exit(1)
    if not settings.model_id:
        typer.echo("Error: --model-id or MODEL_ID environment variable is required")
        raise typer.Exit(1)
    if not settings.backend_url:
        typer.echo(
            "Error: --backend-url or BACKEND_URL environment variable is required"
        )
        raise typer.Exit(1)

    typer.echo(
        f"Starting worker {settings.worker_id} for model {settings.model_id} on port {port}"
    )
    typer.echo(f"Controller: {settings.controller_url}")
    typer.echo(f"Backend: {settings.backend_url}")

    uvicorn.run(
        "llm_gateway.worker.server:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    app()
