"""Daemon server — Unix Domain Socket HTTP server."""

from __future__ import annotations

import asyncio
import errno
import signal
import socket
from pathlib import Path

import uvicorn

from yuuagents.config import Config
from yuuagents.daemon.api import create_app
from yuuagents.daemon.docker import DockerManager
from yuuagents.daemon.manager import AgentManager


async def serve(config: Config) -> None:
    """Start the daemon: Docker manager, agent manager, HTTP server."""
    docker = DockerManager(image=config.docker.image)
    manager = AgentManager(config=config, docker=docker, db_url=config.db_url)
    await manager.start()

    server: uvicorn.Server | None = None

    def request_shutdown() -> None:
        nonlocal server
        if server is not None:
            server.should_exit = True

    app = create_app(manager, request_shutdown=request_shutdown)
    sock_path = config.socket_path
    sock_path.parent.mkdir(parents=True, exist_ok=True)
    if sock_path.exists():
        if _can_connect(sock_path):
            raise RuntimeError(f"daemon already running (socket: {sock_path})")
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
        loop.add_signal_handler(sig, lambda: asyncio.create_task(_shutdown(server)))

    try:
        await server.serve()
    finally:
        await manager.stop()
        if sock_path.exists():
            sock_path.unlink()


async def _shutdown(server: uvicorn.Server) -> None:
    server.should_exit = True


def _can_connect(sock_path: str | Path) -> bool:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.2)
    try:
        sock.connect(str(sock_path))
    except FileNotFoundError:
        return False
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK, errno.EINVAL):
            return False
        raise
    else:
        return True
    finally:
        sock.close()
