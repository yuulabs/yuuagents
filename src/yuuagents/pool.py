"""AgentPool — concrete session pool."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import msgspec
import yuullm
from attrs import define, field

from yuuagents.input import AgentInput
from yuuagents.persistence import EphemeralPersistence, Persistence
from yuuagents.runtime_session import Session
from yuuagents.types import AgentStatus, ErrorInfo

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # add if needed to break circular imports


def _make_error(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        message=traceback.format_exc(),
        error_type=type(exc).__name__,
        timestamp=datetime.now(timezone.utc),
    )


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 32:
        return text[:max_chars]
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


@define
class AgentPool:
    """Concrete session pool — manages a collection of running agents.

    ``spawn`` creates a child agent linked to a parent run. The background
    control methods (inspect/cancel/defer/send_input/wait) operate on any
    run tracked by the pool — both delegated agents and background tool runs.
    """

    persistence: Persistence = field(factory=EphemeralPersistence)
    session_builder: Any = None  # Callable[..., Awaitable[Session]] | None
    _sessions: dict[str, Session] = field(factory=dict, init=False)
    _tasks: dict[str, asyncio.Task[None]] = field(factory=dict, init=False)
    _delegate_sessions_by_run_id: dict[str, Session] = field(factory=dict, init=False)
    _snapshot_turn: dict[str, int] = field(factory=dict, init=False)

    async def run(self, session: Session, input: AgentInput | None = None) -> None:
        """Register session and launch it as an asyncio.Task."""
        self._sessions[session.task_id] = session
        self._tasks[session.task_id] = asyncio.create_task(self._run(session, input))

    def get_session(self, task_id: str) -> Session | None:
        return self._sessions.get(task_id)

    def iter_sessions(self) -> Iterable[Session]:
        return self._sessions.values()

    async def stop(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._sessions.clear()
        self._snapshot_turn.clear()
        self._delegate_sessions_by_run_id.clear()

    async def spawn(
        self,
        *,
        parent: Session,
        parent_run_id: str,
        agent: str,
        input: AgentInput,
        tools: list[str] | None,
        delegate_depth: int,
    ) -> Session:
        if self.session_builder is None:
            raise RuntimeError(
                "AgentPool has no session_builder; "
                "for multi-agent SDK use, provide a session_builder when constructing AgentPool"
            )
        task_id = uuid4().hex
        child = await self.session_builder(
            parent=parent,
            task_id=task_id,
            parent_run_id=parent_run_id,
            agent=agent,
            input=input,
            tools=tools,
            delegate_depth=delegate_depth,
        )
        self._sessions[task_id] = child
        self._delegate_sessions_by_run_id[parent_run_id] = child
        # Important: don't create an asyncio.Task; delegate tool runs child.step_iter() itself
        parent_agent = parent.agent
        child_agent = child.agent
        if parent_agent is not None and child_agent is not None:
            parent_flow = parent_agent.flow.find(parent_run_id)
            if parent_flow is not None and child_agent.flow not in parent_flow.children:
                parent_flow.children.append(child_agent.flow)
        return child

    def inspect(
        self,
        *,
        parent: Session,
        run_id: str,
        limit: int = 200,
        max_chars: int = 4000,
    ) -> str:
        from yuuagents.core.flow import render_agent_event

        agent = parent.agent
        if agent is None:
            return f"[ERROR] parent session not started for run {run_id}"
        flow = agent.flow.find(run_id)
        if flow is None:
            return f"[ERROR] unknown run id {run_id!r}"
        lines = [f"run_id: {flow.id}", f"kind: {flow.kind}"]
        tool_name = flow.info.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            lines.append(f"tool_name: {tool_name}")
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is not None:
            lines.append(f"delegate_task_id: {delegate.task_id}")
            lines.append(f"delegate_status: {delegate.status.value}")
            delegate_agent = delegate.agent
            if delegate_agent is not None:
                lines.append("delegate_stem:")
                lines.append(delegate_agent.render(limit=limit) or "<empty>")
        rendered = flow.render(render_agent_event if flow.kind == "agent" else str, limit=limit)
        lines.append("stem:")
        lines.append(rendered or "<empty>")
        return _truncate_text("\n".join(lines), max_chars)

    def cancel(
        self,
        *,
        parent: Session,
        run_id: str,
    ) -> str:
        agent = parent.agent
        if agent is None:
            return f"[ERROR] parent session not started for run {run_id}"
        flow = agent.flow.find(run_id)
        if flow is None:
            return f"[ERROR] unknown run id {run_id!r}"
        flow.cancel()
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is not None:
            delegate.status = AgentStatus.CANCELLED
        return f"Cancelled run {run_id}"

    def defer(
        self,
        *,
        parent: Session,
        run_id: str,
        message: str,
    ) -> str:
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is None:
            return f"[ERROR] run {run_id!r} is not a delegated agent"
        prompt = (
            message.strip()
            or "请立即停止等待中的前台工具，把当前工作移到后台，并先汇报简短进展。"
        )
        delegate.send(yuullm.user(prompt), defer_tools=True)
        return f"Sent defer signal to delegated run {run_id}"

    async def send_input(
        self,
        *,
        parent: Session,
        run_id: str,
        data: str,
        append_newline: bool = True,
    ) -> str:
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is not None:
            delegate.send(yuullm.user(data), defer_tools=False)
            return f"Input sent to delegated run {run_id}"
        agent = parent.agent
        if agent is None:
            return f"[ERROR] parent session not started for run {run_id}"
        flow = agent.flow.find(run_id)
        if flow is None:
            return f"[ERROR] unknown run id {run_id!r}"
        tool_name = flow.info.get("tool_name")
        if tool_name != "execute_bash":
            return f"[ERROR] run {run_id!r} does not accept input"
        docker = parent.context.capabilities.docker
        if docker is None or not docker.container_id:
            return f"[ERROR] docker terminal unavailable for run {run_id!r}"
        return await docker.executor.write_terminal(
            docker.container_id,
            run_id,
            data,
            append_newline=append_newline,
        )

    async def wait(
        self,
        *,
        parent: Session,
        run_ids: list[str],
    ) -> str:
        agent = parent.agent
        if agent is None:
            return "[ERROR] parent session not started for wait"
        if not run_ids:
            return "[ERROR] run_ids must not be empty"

        waits: list[Any] = []
        for run_id in run_ids:
            delegate = self._delegate_sessions_by_run_id.get(run_id)
            if delegate is not None:
                task = self._tasks.get(delegate.task_id)
                if task is None:
                    return f"[ERROR] delegated run {run_id!r} not started"
                waits.append(task)
                continue
            flow = agent.flow.find(run_id)
            if flow is None:
                return f"[ERROR] unknown run id {run_id!r}"
            waits.append(flow.wait())

        await asyncio.gather(*waits)
        return f"Wait finished for runs: {', '.join(run_ids)}"

    async def _run(
        self,
        session: Session,
        agent_input: AgentInput | None,
    ) -> None:
        from loguru import logger

        session.status = AgentStatus.RUNNING
        if agent_input is not None:
            session.start(agent_input)
        try:
            async for _step in session.step_iter():
                if not session.has_pending_background:
                    await self._persist_snapshot(session)
            session.status = AgentStatus.DONE
            await self._write_terminal(session.task_id, AgentStatus.DONE, None)
        except asyncio.CancelledError:
            await session.kill()
            await self._persist_snapshot(session)
            session.status = AgentStatus.CANCELLED
            await self._write_terminal(session.task_id, AgentStatus.CANCELLED, None)
        except Exception as exc:
            await session.kill()
            err = _make_error(exc)
            session.status = AgentStatus.ERROR
            session.error = err
            err_json = msgspec.json.encode(err)
            await self._persist_snapshot(session)
            await self._write_terminal(session.task_id, AgentStatus.ERROR, err_json)
            logger.exception(
                "agent {agent_id} task {task_id} failed",
                agent_id=session.agent_id,
                task_id=session.task_id,
            )

    async def _persist_snapshot(self, session: Session) -> None:
        """Persist the current agent state as a snapshot checkpoint."""
        try:
            state = await session.snapshot()
        except RuntimeError:
            return
        turn = self._snapshot_turn.get(session.task_id, 0) + 1
        self._snapshot_turn[session.task_id] = turn
        await self.persistence.save_snapshot(
            task_id=session.task_id,
            turn=turn,
            state=state,
            status=session.status,
        )

    async def _write_terminal(
        self,
        task_id: str,
        status: AgentStatus,
        error_json: bytes | None,
    ) -> None:
        if status in (AgentStatus.ERROR, AgentStatus.CANCELLED, AgentStatus.DONE):
            await self.persistence.update_task_terminal(
                task_id=task_id,
                status=status,
                error_json=error_json,
            )
