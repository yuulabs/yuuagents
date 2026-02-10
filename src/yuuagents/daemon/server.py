"""Daemon server — Unix Domain Socket HTTP server."""

from __future__ import annotations

import asyncio
import errno
import os
import shutil
import signal
import socket
from pathlib import Path

import uvicorn

import yuutrace as ytrace

from yuuagents.config import Config
from yuuagents.daemon.api import create_app
from yuuagents.daemon.docker import DockerManager
from yuuagents.daemon.manager import AgentManager


async def serve(config: Config) -> None:
    """Start the daemon: Docker manager, agent manager, HTTP server."""
    trace_proc = await _ensure_yuutrace_server(config)
    ytrace.init(
        endpoint=f"http://127.0.0.1:{config.yuutrace.server_port}/v1/traces",
        service_name="yuuagents",
    )

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
        if trace_proc is not None:
            trace_proc.terminate()
            try:
                await asyncio.wait_for(trace_proc.wait(), timeout=5)
            except TimeoutError:
                trace_proc.kill()
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


def _tcp_port_open(host: str, port: int) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.2)
    try:
        s.connect((host, port))
    except OSError:
        return False
    else:
        return True
    finally:
        s.close()


async def _ensure_yuutrace_server(
    config: Config,
) -> asyncio.subprocess.Process | None:
    host = "127.0.0.1"
    port = config.yuutrace.server_port
    if _tcp_port_open(host, port):
        db_path = Path(config.yuutrace.db_path).expanduser()
        raise RuntimeError(
            f"Port {host}:{port} is already in use. "
            "yagents expects to own the ytrace server process. "
            "Stop the existing process or choose a different yuutrace.server_port. "
            "Expected startup command: "
            f"ytrace server --db {db_path} --port {port}"
        )

    ytrace_bin = shutil.which("ytrace")
    if ytrace_bin is None:
        raise RuntimeError("ytrace not found in PATH")

    db_path = Path(config.yuutrace.db_path).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    proc = await asyncio.create_subprocess_exec(
        ytrace_bin,
        "server",
        "--db",
        str(db_path),
        "--port",
        str(port),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=str(Path.cwd()),
        env=dict(os.environ),
    )

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while loop.time() < deadline:
        if _tcp_port_open(host, port):
            return proc
        if proc.returncode is not None:
            raise RuntimeError(
                f"ytrace server exited early with code {proc.returncode}"
            )
        await asyncio.sleep(0.05)

    proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=2)
    except TimeoutError:
        proc.kill()
    raise TimeoutError(
        f"Timed out waiting for ytrace server to listen on {host}:{port}"
    )
