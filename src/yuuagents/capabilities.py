"""Optional runtime capabilities injected into tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from attrs import define, field

if TYPE_CHECKING:
    from yuuagents.context import AgentContext
    from yuuagents.pool import AgentPool


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


@define
class AgentCapabilities:
    docker: DockerCapability | None = None
    web: WebCapability | None = None
    subprocess_env: dict[str, str] = field(factory=dict)


def require_docker(ctx: AgentContext) -> DockerCapability:
    capability = ctx.capabilities.docker
    if capability is None:
        raise RuntimeError("docker capability unavailable for this agent")
    return capability


def require_agent_pool(ctx: AgentContext) -> AgentPool:
    pool = ctx.pool
    if pool is None:
        raise RuntimeError("agent pool unavailable for this agent")
    return pool  # type: ignore[return-value]


def require_web(ctx: AgentContext) -> WebCapability:
    capability = ctx.capabilities.web
    if capability is None or not capability.api_key:
        raise RuntimeError("web capability unavailable for this agent")
    return capability
