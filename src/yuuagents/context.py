"""AgentContext — dependency-injection context for builtin tools."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from attrs import define, field

if TYPE_CHECKING:
    from yuuagents.daemon.docker import DockerManager


@define
class AgentContext:
    """Injected into tools via ``yt.depends()``.

    Each agent gets its own context instance.
    """

    agent_id: str
    workdir: str
    docker_container: str
    docker: Any = None  # DockerManager — typed as Any to avoid circular import
    input_queue: asyncio.Queue[str] = field(factory=asyncio.Queue)
    tavily_api_key: str = ""
