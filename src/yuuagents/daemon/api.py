"""REST API routes (Starlette)."""

from __future__ import annotations

from typing import Any

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


def create_app(manager: AgentManager) -> Starlette:
    """Build the Starlette ASGI app wired to *manager*."""

    async def health(request: Request) -> Response:
        return _json({"status": "ok"})

    async def create_agent(request: Request) -> Response:
        body = await request.body()
        req = _task_decoder.decode(body)
        agent_id = await manager.submit(req)
        return _json({"agent_id": agent_id}, status=201)

    async def list_agents(request: Request) -> Response:
        return _json(manager.list_agents())

    async def get_agent(request: Request) -> Response:
        agent_id = request.path_params["agent_id"]
        try:
            info = manager.status(agent_id)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        return _json(info)

    async def get_history(request: Request) -> Response:
        agent_id = request.path_params["agent_id"]
        try:
            hist = manager.history(agent_id)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        # History is list[Message] = list[tuple[str, list[Item]]]
        serializable = [{"role": role, "items": items} for role, items in hist]
        return JSONResponse(serializable)

    async def post_input(request: Request) -> Response:
        agent_id = request.path_params["agent_id"]
        body = await request.json()
        content = body.get("content", "")
        try:
            await manager.respond(agent_id, content)
        except KeyError:
            return _json({"error": "not found"}, status=404)
        return _json({"ok": True})

    async def delete_agent(request: Request) -> Response:
        agent_id = request.path_params["agent_id"]
        try:
            await manager.cancel(agent_id)
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
        return _json(
            {
                "socket": cfg.daemon.socket,
                "docker_image": cfg.docker.image,
                "llm_provider": cfg.llm.provider,
                "llm_default_model": cfg.llm.default_model,
                "skill_paths": cfg.skills.paths,
            }
        )

    routes = [
        Route("/health", health, methods=["GET"]),
        Route("/api/agents", create_agent, methods=["POST"]),
        Route("/api/agents", list_agents, methods=["GET"]),
        Route("/api/agents/{agent_id}", get_agent, methods=["GET"]),
        Route("/api/agents/{agent_id}/history", get_history, methods=["GET"]),
        Route("/api/agents/{agent_id}/input", post_input, methods=["POST"]),
        Route("/api/agents/{agent_id}", delete_agent, methods=["DELETE"]),
        Route("/api/skills", list_skills, methods=["GET"]),
        Route("/api/skills/scan", scan_skills, methods=["POST"]),
        Route("/api/config", get_config, methods=["GET"]),
    ]

    app = Starlette(routes=routes)
    return app
