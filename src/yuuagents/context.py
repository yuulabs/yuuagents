"""AgentContext — dependency-injection context for builtin tools."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from attrs import define, evolve

if TYPE_CHECKING:
    from yuuagents.core.flow import Flow
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


class DelegateDepthExceededError(RuntimeError):
    def __init__(
        self,
        *,
        max_depth: int,
        current_depth: int,
        target_agent: str,
    ) -> None:
        msg = (
            "delegate depth limit exceeded "
            f"(max_depth={max_depth}, current_depth={current_depth}, target_agent={target_agent!r}). "
            "Stop delegating and complete the task in the current agent."
        )
        super().__init__(msg)


@define
class AgentContext:
    """Injected into tools via ``yt.depends()``.

    Each agent gets its own context instance.
    """

    task_id: str
    agent_id: str
    workdir: str
    docker_container: str
    delegate_depth: int = 0
    manager: DelegateManager | None = None
    docker: DockerExecutor | None = None
    tavily_api_key: str = ""
    subprocess_env: dict | None = None
    addon_context: object | None = None  # yuubot AddonContext, opaque to yuuagents
    session: Session | None = None
    current_run_id: str = ""
    current_flow: Flow | None = None

    def evolve(self, **changes: object) -> AgentContext:
        return evolve(self, **changes)
