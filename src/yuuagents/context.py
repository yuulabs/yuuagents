"""AgentContext — minimal runtime context shared by SDK and daemon mode."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from attrs import define, field

from yuuagents.capabilities import AgentCapabilities

if TYPE_CHECKING:
    from yuuagents.core.flow import Flow
    from yuuagents.pool import AgentPool
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


_MISSING = object()
T = TypeVar("T")


def _replace(current: T, value: T | object) -> T:
    if value is _MISSING:
        return current
    return cast(T, value)


@define
class AgentContext:
    """Injected into tools via ``yt.depends()``.

    Each agent gets its own context instance.
    """

    task_id: str
    agent_id: str
    workdir: str
    pool: AgentPool | None = None
    capabilities: AgentCapabilities = field(factory=AgentCapabilities)
    delegate_depth: int = 0
    session: Session | None = None
    current_run_id: str = ""
    current_flow: Flow | None = None

    def evolve(
        self,
        *,
        task_id: str | object = _MISSING,
        agent_id: str | object = _MISSING,
        workdir: str | object = _MISSING,
        pool: AgentPool | None | object = _MISSING,
        capabilities: AgentCapabilities | object = _MISSING,
        delegate_depth: int | object = _MISSING,
        session: Session | None | object = _MISSING,
        current_run_id: str | object = _MISSING,
        current_flow: Flow | None | object = _MISSING,
    ) -> AgentContext:
        return AgentContext(
            task_id=_replace(self.task_id, task_id),
            agent_id=_replace(self.agent_id, agent_id),
            workdir=_replace(self.workdir, workdir),
            pool=_replace(self.pool, pool),
            capabilities=_replace(self.capabilities, capabilities),
            delegate_depth=_replace(self.delegate_depth, delegate_depth),
            session=_replace(self.session, session),
            current_run_id=_replace(self.current_run_id, current_run_id),
            current_flow=_replace(self.current_flow, current_flow),
        )
