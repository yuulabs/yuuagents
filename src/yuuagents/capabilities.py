"""Optional runtime capabilities injected into tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from attrs import define, field

if TYPE_CHECKING:
    from yuuagents.context import AgentContext
    from yuuagents.basin import Basin
    from yuuagents.core.flow import Flow
    from yuuagents.input import AgentInput


class DockerExecutor(Protocol):
    async def exec(self, container_id: str, command: str, timeout: int) -> str: ...
    async def exec_terminal(
        self,
        container_id: str,
        session_id: str,
        command: str,
        timeout: int,
        *,
        soft_timeout: int | None = None,
    ) -> str: ...
    def get_pending(
        self,
        container_id: str,
        session_id: str,
    ) -> Any: ...
    async def resume_pending(
        self,
        container_id: str,
        session_id: str,
        timeout: int,
    ) -> str: ...
    async def write_terminal(
        self,
        container_id: str,
        session_id: str,
        data: str,
        *,
        append_newline: bool = True,
    ) -> str: ...
    async def capture_terminal(
        self,
        container_id: str,
        session_id: str,
    ) -> str: ...


@define(frozen=True)
class DockerCapability:
    executor: DockerExecutor
    container_id: str


@define(frozen=True)
class WebCapability:
    api_key: str


SpawnAgent = Callable[
    ["Flow[Any, Any]", str, "AgentInput", list[str] | None, int],
    Awaitable[Any],
]


@define
class AgentCapabilities:
    docker: DockerCapability | None = None
    web: WebCapability | None = None
    basin: Basin | None = None
    spawn_agent: SpawnAgent | None = None
    subprocess_env: dict[str, str] = field(factory=dict)


def require_docker(ctx: AgentContext) -> DockerCapability:
    capability = ctx.capabilities.docker
    if capability is None:
        raise RuntimeError("docker capability unavailable for this agent")
    return capability


def require_basin(ctx: AgentContext) -> Basin:
    basin = ctx.capabilities.basin
    if basin is None:
        raise RuntimeError("basin capability unavailable for this agent")
    return basin


def require_spawn_agent(ctx: AgentContext) -> SpawnAgent:
    spawn_agent = ctx.capabilities.spawn_agent
    if spawn_agent is None:
        raise RuntimeError("spawn_agent capability unavailable for this agent")
    return spawn_agent


def require_web(ctx: AgentContext) -> WebCapability:
    capability = ctx.capabilities.web
    if capability is None or not capability.api_key:
        raise RuntimeError("web capability unavailable for this agent")
    return capability
