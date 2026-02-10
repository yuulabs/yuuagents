"""Task persistence — append-only checkpoints + deterministic replay."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Literal

import msgspec
import yuullm
from attrs import define, field
from sqlalchemy import (
    DateTime,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    select,
    update,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from yuuagents.types import AgentInfo, AgentStatus, ErrorInfo

Phase = Literal["llm", "tool"]

_json_encoder = msgspec.json.Encoder()


class ToolCallDTO(msgspec.Struct, frozen=True, kw_only=True):
    call_id: str
    name: str
    args_json: str


class LlmCheckpointPayload(msgspec.Struct, frozen=True, kw_only=True):
    history_append: Any | None = None
    tool_calls: list[ToolCallDTO] = msgspec.field(default_factory=list)
    status_after: str = ""


class ToolErrorDTO(msgspec.Struct, frozen=True, kw_only=True):
    type: str
    message: str
    interrupted: bool = False
    cancelled: bool = False


class ToolResultDTO(msgspec.Struct, frozen=True, kw_only=True):
    call_id: str
    ok: bool
    output_text: str = ""
    error: ToolErrorDTO | None = None


class ToolCheckpointPayload(msgspec.Struct, frozen=True, kw_only=True):
    results: list[ToolResultDTO] = msgspec.field(default_factory=list)
    status_after: str = ""


class _Base(DeclarativeBase):
    pass


class TaskRow(_Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)
    task: Mapped[str] = mapped_column(Text, nullable=False)
    system_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False, default="")
    tools_json: Mapped[bytes] = mapped_column(
        LargeBinary, nullable=False, default=b"[]"
    )
    docker_container: Mapped[str] = mapped_column(String, nullable=False, default="")

    status: Mapped[str] = mapped_column(String, nullable=False)
    head_turn: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    error_json: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)


class TaskCheckpointRow(_Base):
    __tablename__ = "task_checkpoints"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    turn: Mapped[int] = mapped_column(Integer, primary_key=True)
    phase: Mapped[str] = mapped_column(String, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)


Index(
    "ix_task_checkpoints_task_id_turn",
    TaskCheckpointRow.task_id,
    TaskCheckpointRow.turn,
)


@define(frozen=True)
class RestoredTask:
    task_id: str
    agent_id: str
    persona: str
    task: str
    system_prompt: str
    model: str
    tools: list[str]
    docker_container: str
    status: AgentStatus
    created_at: datetime
    head_turn: int
    history: list[Any]


@define
class TaskPersistence:
    db_url: str
    _engine: AsyncEngine | None = None
    _sessionmaker: async_sessionmaker[AsyncSession] | None = None

    async def start(self) -> None:
        assert self._engine is None
        self._engine = create_async_engine(self.db_url, future=True)
        self._sessionmaker = async_sessionmaker(self._engine, expire_on_commit=False)
        async with self._engine.begin() as conn:
            await conn.run_sync(_Base.metadata.create_all)

    async def stop(self) -> None:
        if self._engine is None:
            return
        await self._engine.dispose()
        self._engine = None
        self._sessionmaker = None

    def _session(self) -> async_sessionmaker[AsyncSession]:
        assert self._sessionmaker is not None
        return self._sessionmaker

    async def create_task(
        self,
        *,
        task_id: str,
        agent_id: str,
        persona: str,
        task: str,
        system_prompt: str,
        model: str,
        tools: list[str],
        docker_container: str,
        created_at: datetime,
    ) -> None:
        now = datetime.now(timezone.utc)
        stmt = sqlite_insert(TaskRow).values(
            task_id=task_id,
            agent_id=agent_id,
            persona=persona,
            task=task,
            system_prompt=system_prompt,
            model=model,
            tools_json=_json_encoder.encode(tools),
            docker_container=docker_container,
            status=AgentStatus.RUNNING.value,
            head_turn=0,
            created_at=created_at,
            updated_at=now,
            error_json=None,
        )
        stmt = stmt.on_conflict_do_nothing(index_elements=[TaskRow.task_id])
        async with self._session()() as session:
            async with session.begin():
                await session.execute(stmt)

    async def update_task_terminal(
        self,
        *,
        task_id: str,
        status: AgentStatus,
        error_json: bytes | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self._session()() as session:
            async with session.begin():
                await session.execute(
                    update(TaskRow)
                    .where(TaskRow.task_id == task_id)
                    .values(status=status.value, updated_at=now, error_json=error_json)
                )

    async def list_tasks(self) -> list[AgentInfo]:
        async with self._session()() as session:
            rows = (
                (
                    await session.execute(
                        select(TaskRow).order_by(TaskRow.updated_at.desc())
                    )
                )
                .scalars()
                .all()
            )
        infos: list[AgentInfo] = []
        for r in rows:
            err: ErrorInfo | None = None
            if r.error_json is not None:
                err = msgspec.json.decode(r.error_json, type=ErrorInfo)
            infos.append(
                AgentInfo(
                    task_id=r.task_id,
                    agent_id=r.agent_id,
                    persona=r.persona[:80],
                    task=r.task,
                    status=r.status,
                    created_at=r.created_at.isoformat(),
                    last_assistant_message="",
                    steps=r.head_turn,
                    total_tokens=0,
                    total_cost_usd=0.0,
                    error=err,
                )
            )
        return infos

    async def get_task_row(self, task_id: str) -> TaskRow | None:
        async with self._session()() as session:
            row = (
                await session.execute(select(TaskRow).where(TaskRow.task_id == task_id))
            ).scalar_one_or_none()
        return row

    async def load_history(self, task_id: str) -> list[Any]:
        row = await self.get_task_row(task_id)
        if row is None:
            raise KeyError(task_id)

        history: list[Any] = [
            yuullm.system(row.system_prompt),
            yuullm.user(row.task),
        ]

        async with self._session()() as session:
            cps = (
                (
                    await session.execute(
                        select(TaskCheckpointRow)
                        .where(TaskCheckpointRow.task_id == task_id)
                        .order_by(
                            TaskCheckpointRow.turn.asc(), TaskCheckpointRow.phase.asc()
                        )
                    )
                )
                .scalars()
                .all()
            )

        for cp in cps:
            if cp.phase == "llm":
                llm_payload = msgspec.json.decode(cp.payload, type=LlmCheckpointPayload)
                if llm_payload.history_append is not None:
                    history.append(llm_payload.history_append)
                continue

            tool_payload = msgspec.json.decode(cp.payload, type=ToolCheckpointPayload)
            for r in tool_payload.results:
                if r.ok:
                    content = r.output_text
                elif r.error is not None:
                    content = r.error.message
                else:
                    content = ""
                history.append(yuullm.tool(r.call_id, content))

        return history

    async def pending_input_prompt(self, task_id: str) -> str | None:
        row = await self.get_task_row(task_id)
        if row is None:
            raise KeyError(task_id)
        if row.status != AgentStatus.BLOCKED_ON_INPUT.value:
            return None
        if row.head_turn <= 0:
            return None

        turn = row.head_turn
        async with self._session()() as session:
            llm_cp = (
                await session.execute(
                    select(TaskCheckpointRow)
                    .where(
                        (TaskCheckpointRow.task_id == task_id)
                        & (TaskCheckpointRow.turn == turn)
                        & (TaskCheckpointRow.phase == "llm")
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()

        if llm_cp is None:
            return None

        llm_payload = msgspec.json.decode(llm_cp.payload, type=LlmCheckpointPayload)
        for c in llm_payload.tool_calls:
            if c.name != "user_input":
                continue
            args = json.loads(c.args_json) if c.args_json else {}
            prompt = str(args.get("prompt", "")).strip()
            return prompt or None
        return None

    async def load_unfinished(self) -> list[RestoredTask]:
        async with self._session()() as session:
            rows = (
                (
                    await session.execute(
                        select(TaskRow).where(
                            TaskRow.status.in_(
                                [
                                    AgentStatus.RUNNING.value,
                                    AgentStatus.BLOCKED_ON_INPUT.value,
                                ]
                            )
                        )
                    )
                )
                .scalars()
                .all()
            )

        restored: list[RestoredTask] = []
        for r in rows:
            tools = msgspec.json.decode(r.tools_json, type=list[str])
            history = await self.load_history(r.task_id)
            restored.append(
                RestoredTask(
                    task_id=r.task_id,
                    agent_id=r.agent_id,
                    persona=r.persona,
                    task=r.task,
                    system_prompt=r.system_prompt,
                    model=r.model,
                    tools=tools,
                    docker_container=r.docker_container,
                    status=AgentStatus(r.status),
                    created_at=r.created_at,
                    head_turn=r.head_turn,
                    history=history,
                )
            )
        return restored

    async def recover_pending_tools(self, task_id: str) -> bool:
        row = await self.get_task_row(task_id)
        if row is None:
            raise KeyError(task_id)
        if row.status != AgentStatus.BLOCKED_ON_INPUT.value:
            return False
        if row.head_turn <= 0:
            return False

        turn = row.head_turn
        async with self._session()() as session:
            cps = (
                (
                    await session.execute(
                        select(TaskCheckpointRow).where(
                            (TaskCheckpointRow.task_id == task_id)
                            & (TaskCheckpointRow.turn == turn)
                        )
                    )
                )
                .scalars()
                .all()
            )

        llm_cp = next((c for c in cps if c.phase == "llm"), None)
        tool_cp = next((c for c in cps if c.phase == "tool"), None)
        if llm_cp is None:
            return False

        llm_payload = msgspec.json.decode(llm_cp.payload, type=LlmCheckpointPayload)
        if not llm_payload.tool_calls:
            return False

        requested = {c.call_id for c in llm_payload.tool_calls}
        done: set[str] = set()
        if tool_cp is not None:
            tool_payload = msgspec.json.decode(
                tool_cp.payload, type=ToolCheckpointPayload
            )
            done = {r.call_id for r in tool_payload.results}

        pending = sorted(requested - done)
        if not pending:
            return False

        payload = ToolCheckpointPayload(
            results=[
                ToolResultDTO(
                    call_id=cid,
                    ok=False,
                    error=ToolErrorDTO(
                        type="interrupted",
                        message="interrupted",
                        interrupted=True,
                    ),
                )
                for cid in pending
            ],
            status_after=AgentStatus.RUNNING.value,
        )
        now = datetime.now(timezone.utc)
        insert_cp = sqlite_insert(TaskCheckpointRow).values(
            task_id=task_id,
            turn=turn,
            phase="tool",
            ts=now,
            payload=msgspec.json.encode(payload),
        )
        insert_cp = insert_cp.on_conflict_do_nothing(
            index_elements=[
                TaskCheckpointRow.task_id,
                TaskCheckpointRow.turn,
                TaskCheckpointRow.phase,
            ]
        )

        async with self._session()() as session:
            async with session.begin():
                res: Any = await session.execute(insert_cp)
                if res.rowcount and res.rowcount > 0:  # type:ignore
                    await session.execute(
                        update(TaskRow)
                        .where(TaskRow.task_id == task_id)
                        .values(status=AgentStatus.RUNNING.value, updated_at=now)
                    )
                    return True
        return False


@define(frozen=True)
class _BufferedCheckpoint:
    task_id: str
    turn: int
    phase: Phase
    ts: datetime
    payload: bytes
    status_after: AgentStatus


@define
class TaskWriter:
    persistence: TaskPersistence
    _queue: asyncio.Queue[object] = field(factory=asyncio.Queue)
    _task: asyncio.Task[None] | None = None
    _stopping: bool = False

    async def start(self) -> None:
        assert self._task is None
        self._task = asyncio.create_task(self._run())

    async def flush(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stopping = True
        await self._queue.put(None)
        await self._task
        self._task = None
        self._stopping = False

    async def append_checkpoint(self, cp: _BufferedCheckpoint) -> None:
        await self._queue.put(cp)

    async def _run(self) -> None:
        buffer: list[_BufferedCheckpoint] = []
        bytes_total = 0
        last_flush = asyncio.get_running_loop().time()

        async def _flush() -> None:
            nonlocal buffer, bytes_total, last_flush
            if not buffer:
                return

            task_updates: dict[str, tuple[int, str, datetime]] = {}
            values = []
            for cp in buffer:
                values.append(
                    {
                        "task_id": cp.task_id,
                        "turn": cp.turn,
                        "phase": cp.phase,
                        "ts": cp.ts,
                        "payload": cp.payload,
                    }
                )
                prev = task_updates.get(cp.task_id)
                head = cp.turn if prev is None else max(prev[0], cp.turn)
                task_updates[cp.task_id] = (head, cp.status_after.value, cp.ts)

            insert_stmt = sqlite_insert(TaskCheckpointRow).values(values)
            insert_stmt = insert_stmt.on_conflict_do_nothing(
                index_elements=[
                    TaskCheckpointRow.task_id,
                    TaskCheckpointRow.turn,
                    TaskCheckpointRow.phase,
                ]
            )

            async with self.persistence._session()() as session:  # noqa: SLF001
                async with session.begin():
                    await session.execute(insert_stmt)
                    for task_id, (head_turn, status, ts) in task_updates.items():
                        await session.execute(
                            update(TaskRow)
                            .where(TaskRow.task_id == task_id)
                            .values(head_turn=head_turn, updated_at=ts)
                        )
                        await session.execute(
                            update(TaskRow)
                            .where(TaskRow.task_id == task_id)
                            .values(status=status)
                        )

            buffer = []
            bytes_total = 0
            last_flush = asyncio.get_running_loop().time()

        while True:
            timeout = 0.3
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            except asyncio.TimeoutError:
                item = "timeout"

            if item is None:
                await _flush()
                if self._stopping:
                    return
                continue

            if item == "timeout":
                await _flush()
                continue

            assert isinstance(item, _BufferedCheckpoint)
            buffer.append(item)
            bytes_total += len(item.payload)

            elapsed = asyncio.get_running_loop().time() - last_flush
            if len(buffer) >= 200 or bytes_total >= 1024 * 1024 or elapsed >= 0.3:
                await _flush()


@define(frozen=True)
class TaskRecorder:
    task_id: str
    writer: TaskWriter

    async def record_llm(
        self,
        *,
        turn: int,
        history_append: Any | None,
        tool_calls: list[ToolCallDTO],
    ) -> None:
        status_after = AgentStatus.BLOCKED_ON_INPUT if tool_calls else AgentStatus.DONE
        payload = LlmCheckpointPayload(
            history_append=history_append,
            tool_calls=tool_calls,
            status_after=status_after.value,
        )
        cp = _BufferedCheckpoint(
            task_id=self.task_id,
            turn=turn,
            phase="llm",
            ts=datetime.now(timezone.utc),
            payload=msgspec.json.encode(payload),
            status_after=status_after,
        )
        await self.writer.append_checkpoint(cp)

    async def record_user(
        self,
        *,
        turn: int,
        message: Any,
    ) -> None:
        payload = LlmCheckpointPayload(
            history_append=message,
            tool_calls=[],
            status_after=AgentStatus.RUNNING.value,
        )
        cp = _BufferedCheckpoint(
            task_id=self.task_id,
            turn=turn,
            phase="llm",
            ts=datetime.now(timezone.utc),
            payload=msgspec.json.encode(payload),
            status_after=AgentStatus.RUNNING,
        )
        await self.writer.append_checkpoint(cp)

    async def record_tool(
        self,
        *,
        turn: int,
        results: list[ToolResultDTO],
    ) -> None:
        payload = ToolCheckpointPayload(
            results=results,
            status_after=AgentStatus.RUNNING.value,
        )
        cp = _BufferedCheckpoint(
            task_id=self.task_id,
            turn=turn,
            phase="tool",
            ts=datetime.now(timezone.utc),
            payload=msgspec.json.encode(payload),
            status_after=AgentStatus.RUNNING,
        )
        await self.writer.append_checkpoint(cp)
