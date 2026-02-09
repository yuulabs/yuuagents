"""AgentContext — dependency-injection context for builtin tools."""

from __future__ import annotations

import asyncio
from typing import Protocol

from attrs import define, field


class DockerExecutor(Protocol):
    async def exec(self, container_id: str, command: str, timeout: int) -> str: ...


@define
class AgentContext:
    """Injected into tools via ``yt.depends()``.

    Each agent gets its own context instance.
    """

    agent_id: str
    workdir: str
    docker_container: str
    docker: DockerExecutor | None = None
    input_queue: asyncio.Queue[str] = field(factory=asyncio.Queue)
    tavily_api_key: str = ""
