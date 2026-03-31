"""AgentContext — minimal runtime context shared by SDK and daemon mode."""

from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import define, evolve, field

from yuuagents.capabilities import AgentCapabilities

if TYPE_CHECKING:
    from yuuagents.core.flow import Flow
    from yuuagents.runtime_session import Session


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
    capabilities: AgentCapabilities = field(factory=AgentCapabilities)
    delegate_depth: int = 0
    session: Session | None = None
    current_run_id: str = ""
    current_flow: Flow | None = None

    def evolve(self, **changes: object) -> AgentContext:
        return evolve(self, **changes)
