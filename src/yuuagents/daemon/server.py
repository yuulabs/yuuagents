"""Daemon server — Unix Domain Socket HTTP server."""

from __future__ import annotations

import asyncio
import signal

import uvicorn

from yuuagents.config import Config
from yuuagents.daemon.api import create_app
from yuuagents.daemon.docker import DockerManager
from yuuagents.daemon.manager import AgentManager


async def serve(config: Config) -> None:
    """Start the daemon: Docker manager, agent manager, HTTP server."""
    docker = DockerManager(image=config.docker.image)
    manager = AgentManager(config=config, docker=docker)
    await manager.start()

    app = create_app(manager)
    sock_path = config.socket_path
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        sock_path.unlink()

    uds_config = uvicorn.Config(
        app,
        uds=str(sock_path),
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(uds_config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(server, manager)))

    try:
        await server.serve()
    finally:
        await manager.stop()
        if sock_path.exists():
            sock_path.unlink()


async def _shutdown(server: uvicorn.Server, manager: AgentManager) -> None:
    await manager.stop()
    server.should_exit = True
