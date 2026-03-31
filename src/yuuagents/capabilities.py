"""Optional runtime capabilities injected into tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from attrs import define, field

if TYPE_CHECKING:
    from yuuagents.context import AgentContext
    from yuuagents.input import AgentInput
    from yuuagents.runtime_session import Session


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


class DelegateManager(Protocol):
    async def start_delegate(
        self,
        *,
        parent: Session,
        parent_run_id: str,
        agent: str,
        input: AgentInput,
        tools: list[str] | None,
        delegate_depth: int,
    ) -> Session: ...

    def inspect_run(
        self,
        *,
        parent: Session,
        run_id: str,
        limit: int = 200,
        max_chars: int = 4000,
    ) -> str: ...

    def cancel_run(
        self,
        *,
        parent: Session,
        run_id: str,
    ) -> str: ...

    def defer_run(
        self,
        *,
        parent: Session,
        run_id: str,
        message: str,
    ) -> str: ...

    async def input_run(
        self,
        *,
        parent: Session,
        run_id: str,
        data: str,
        append_newline: bool = True,
    ) -> str: ...

    async def wait_runs(
        self,
        *,
        parent: Session,
        run_ids: list[str],
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
    delegate: DelegateManager | None = None
    web: WebCapability | None = None
    subprocess_env: dict[str, str] = field(factory=dict)


def require_docker(ctx: AgentContext) -> DockerCapability:
    capability = ctx.capabilities.docker
    if capability is None:
        raise RuntimeError("docker capability unavailable for this agent")
    return capability


def require_delegate_manager(ctx: AgentContext) -> DelegateManager:
    capability = ctx.capabilities.delegate
    if capability is None:
        raise RuntimeError("delegate capability unavailable for this agent")
    return capability


def require_web(ctx: AgentContext) -> WebCapability:
    capability = ctx.capabilities.web
    if capability is None or not capability.api_key:
        raise RuntimeError("web capability unavailable for this agent")
    return capability
