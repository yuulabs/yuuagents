"""AgentContext — dependency-injection context for builtin tools."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable, Protocol

from attrs import define, field

if TYPE_CHECKING:
    from yuuagents.agent import AgentState


class DockerExecutor(Protocol):
    async def exec(self, container_id: str, command: str, timeout: int) -> str: ...
    async def exec_terminal(
        self,
        container_id: str,
        session_id: str,
        command: str,
        timeout: int,
    ) -> str: ...


class DelegateManager(Protocol):
    async def delegate(
        self,
        *,
        caller_agent: str,
        agent: str,
        first_user_message: str,
        tools: list[str] | None,
        delegate_depth: int,
    ) -> str: ...


CliGuard = Callable[[list[str]], None]
"""Receive argv; raise ValueError to block execution."""


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
    state: AgentState | None = None
    input_queue: asyncio.Queue[str] = field(factory=asyncio.Queue)
    tavily_api_key: str = ""
    cli_guard: CliGuard | None = None
