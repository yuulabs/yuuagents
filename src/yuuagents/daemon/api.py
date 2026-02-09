"""REST API routes (Starlette)."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Callable

import msgspec
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from yuuagents.daemon.manager import AgentManager
from yuuagents.types import TaskRequest

_encoder = msgspec.json.Encoder()
_task_decoder = msgspec.json.Decoder(TaskRequest)


def _json(obj: Any, status: int = 200) -> Response:
    """Encode *obj* via msgspec and return a JSON response."""
    body = _encoder.encode(obj)
    return Response(content=body, status_code=status, media_type="application/json")


def create_app(
    manager: AgentManager,
    request_shutdown: Callable[[], None] | None = None,
    *,
    manage_lifecycle: bool = False,
) -> Starlette:
    """Build the Starlette ASGI app wired to *manager*."""
    if request_shutdown is None:

        def request_shutdown() -> None:
            return None

    lifespan = None
    if manage_lifecycle:

        @asynccontextmanager
        async def lifespan(app: Starlette):  # noqa: ARG001
            await manager.start()
            try:
                yield
            finally:
                await manager.stop()

    async def health(request: Request) -> Response:
        return _json({"status": "ok"})

    async def shutdown(request: Request) -> Response:
        request_shutdown()
        return _json({"ok": True})

    async def create_agent(request: Request) -> Response:
        body = await request.body()
        try:
            req = _task_decoder.decode(body)
        except (msgspec.DecodeError, msgspec.ValidationError) as exc:
            return _json({"error": str(exc)}, status=400)

        try:
            task_id = await manager.submit(req)
        except ValueError as exc:
            return _json({"error": str(exc)}, status=400)

        return _json({"task_id": task_id, "agent_id": req.agent}, status=201)

    async def list_agents(request: Request) -> Response:
        return _json(await manager.list_agents())

    async def get_agent(request: Request) -> Response:
        task_id = request.path_params["task_id"]
        try:
            info = await manager.status(task_id)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        return _json(info)

    async def get_history(request: Request) -> Response:
        task_id = request.path_params["task_id"]
        try:
            hist = await manager.history(task_id)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        serializable = [{"role": role, "items": items} for role, items in hist]
        return JSONResponse(serializable)

    async def post_input(request: Request) -> Response:
        task_id = request.path_params["task_id"]
        body = await request.json()
        content = body.get("content", "")
        try:
            await manager.respond(task_id, content)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        return _json({"ok": True})

    async def delete_agent(request: Request) -> Response:
        task_id = request.path_params["task_id"]
        try:
            await manager.cancel(task_id)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        return _json({"ok": True})

    async def list_skills(request: Request) -> Response:
        return _json(manager.skills())

    async def scan_skills(request: Request) -> Response:
        refreshed = manager.rescan_skills()
        return _json(refreshed)

    async def get_config(request: Request) -> Response:
        # Return sanitised config (no secrets)
        cfg = manager.config
        providers_summary = {
            name: {"api_type": p.api_type, "default_model": p.default_model}
            for name, p in cfg.providers.items()
        }
        agents_summary = {
            name: {"provider": a.provider, "model": a.model}
            for name, a in cfg.agents.items()
        }
        return _json(
            {
                "socket": cfg.daemon.socket,
                "docker_image": cfg.docker.image,
                "providers": providers_summary,
                "agents": agents_summary,
                "skill_paths": cfg.skills.paths,
            }
        )

    async def reload_config(request: Request) -> Response:
        """Hot-reload configuration from disk."""
        from yuuagents.config import load as load_config

        try:
            new_config = load_config()
            manager.reload_config(new_config)
            return _json({"ok": True, "message": "Configuration reloaded successfully"})
        except Exception as e:
            return _json({"error": str(e)}, status=500)

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/api/shutdown", shutdown, methods=["POST"]),
        Route("/api/agents", create_agent, methods=["POST"]),
        Route("/api/agents", list_agents, methods=["GET"]),
        Route("/api/agents/{task_id}", get_agent, methods=["GET"]),
        Route("/api/agents/{task_id}/history", get_history, methods=["GET"]),
        Route("/api/agents/{task_id}/input", post_input, methods=["POST"]),
        Route("/api/agents/{task_id}", delete_agent, methods=["DELETE"]),
        Route("/api/skills", list_skills, methods=["GET"]),
        Route("/api/skills/scan", scan_skills, methods=["POST"]),
        Route("/api/config", get_config, methods=["GET"]),
        Route("/api/config/reload", reload_config, methods=["POST"]),
    ]

    app = Starlette(routes=routes, lifespan=lifespan)
    return app
