"""AsyncSession — observable, interruptible, resumable agent sessions."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Coroutine, Protocol
from uuid import uuid4

from attrs import define, field
from loguru import logger

from yuuagents.agent import Agent
from yuuagents.context import AgentContext
from yuuagents.loop import run as run_agent
from yuuagents.types import AgentStatus


class Session(Protocol):
    """Observable async session wrapping a long-running task."""

    session_id: str
    status: str  # pending | running | done | error | interrupted

    async def start(self) -> None: ...
    async def interrupt(self) -> None: ...
    async def resume(self) -> None: ...
    def progress(self) -> list[str]: ...
    def result(self) -> str | None: ...


OnComplete = Callable[[str, str | None, str | None], Coroutine[Any, Any, None]]
"""(session_id, result_text_or_none, error_or_none) -> None"""


@define
class AgentSession:
    """Wraps an agent loop as a background asyncio task.

    Lifecycle: pending → running → done | error | interrupted
    """

    session_id: str
    agent: Agent
    ctx: AgentContext
    task_text: str
    on_complete: OnComplete | None = None
    _asyncio_task: asyncio.Task | None = field(default=None, init=False)
    _started_at: float = field(factory=time.monotonic, init=False)

    @property
    def status(self) -> str:
        st = self.agent.status
        if st == AgentStatus.CANCELLED:
            return "interrupted"
        return st.value

    async def start(self) -> None:
        self._started_at = time.monotonic()
        self._asyncio_task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        result_text: str | None = None
        error_text: str | None = None
        try:
            await run_agent(self.agent, self.task_text, self.ctx)
            result_text = self.result()
        except asyncio.CancelledError:
            self.agent.status = AgentStatus.CANCELLED
            error_text = "interrupted"
        except Exception as exc:
            self.agent.fail(exc)
            error_text = str(exc)
        finally:
            if self.on_complete is not None:
                try:
                    await self.on_complete(self.session_id, result_text, error_text)
                except Exception:
                    logger.exception("on_complete callback failed for {}", self.session_id)

    async def interrupt(self) -> None:
        self.agent.status = AgentStatus.CANCELLED
        if self._asyncio_task and not self._asyncio_task.done():
            self._asyncio_task.cancel()
            try:
                await self._asyncio_task
            except (asyncio.CancelledError, Exception):
                pass

    async def resume(self) -> None:
        if self.agent.status == AgentStatus.CANCELLED:
            self.agent.status = AgentStatus.RUNNING
        self._asyncio_task = asyncio.create_task(self._run_resume())

    async def _run_resume(self) -> None:
        result_text: str | None = None
        error_text: str | None = None
        try:
            await run_agent(self.agent, self.task_text, self.ctx, resume=True)
            result_text = self.result()
        except asyncio.CancelledError:
            self.agent.status = AgentStatus.CANCELLED
            error_text = "interrupted"
        except Exception as exc:
            self.agent.fail(exc)
            error_text = str(exc)
        finally:
            if self.on_complete is not None:
                try:
                    await self.on_complete(self.session_id, result_text, error_text)
                except Exception:
                    logger.exception("on_complete callback failed for {}", self.session_id)

    def progress(self) -> list[str]:
        """Extract recent assistant text from history."""
        out: list[str] = []
        for msg in self.agent.history:
            if not (isinstance(msg, tuple) and len(msg) == 2):
                continue
            role, items = msg
            if role != "assistant" or not isinstance(items, list):
                continue
            parts = [i for i in items if isinstance(i, str)]
            text = "".join(parts).strip()
            if text:
                out.append(text)
        return out

    def result(self) -> str | None:
        if self.agent.status not in (AgentStatus.DONE, AgentStatus.ERROR):
            return None
        msgs = self.progress()
        return msgs[-1] if msgs else None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started_at


@define
class SessionRegistry:
    """Global registry of active agent sessions."""

    _sessions: dict[str, AgentSession] = field(factory=dict)

    def create(
        self,
        agent: Agent,
        ctx: AgentContext,
        task_text: str,
        *,
        on_complete: OnComplete | None = None,
    ) -> AgentSession:
        session_id = f"ses-{uuid4().hex[:12]}"
        session = AgentSession(
            session_id=session_id,
            agent=agent,
            ctx=ctx,
            task_text=task_text,
            on_complete=on_complete,
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> AgentSession | None:
        return self._sessions.get(session_id)

    def list_active(self) -> list[AgentSession]:
        return [s for s in self._sessions.values() if s.status in ("pending", "running")]

    def remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    async def stop_all(self) -> None:
        for s in list(self._sessions.values()):
            await s.interrupt()
        self._sessions.clear()
