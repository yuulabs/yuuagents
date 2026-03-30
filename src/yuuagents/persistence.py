"""Task persistence — snapshot-based checkpoints with structured startup input."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal

import msgspec
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

from yuuagents.input import (
    AgentInput,
    agent_input_from_jsonable,
    agent_input_preview,
    agent_input_to_jsonable,
    message_from_jsonable,
)
from yuuagents.types import AgentInfo, AgentStatus, ErrorInfo

Phase = Literal["snapshot"]

_json_encoder = msgspec.json.Encoder()


class _Base(DeclarativeBase):
    pass


class TaskRow(_Base):
    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    persona: Mapped[str] = mapped_column(Text, nullable=False)
    input_kind: Mapped[str] = mapped_column(String, nullable=False)
    input_preview: Mapped[str] = mapped_column(Text, nullable=False, default="")
    input_json: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
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
    input: AgentInput
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
        input: AgentInput,
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
            input_kind=input.kind,
            input_preview=agent_input_preview(input, max_chars=400),
            input_json=_json_encoder.encode(agent_input_to_jsonable(input)),
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
                    input_kind=r.input_kind,
                    input_preview=r.input_preview,
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

        async with self._session()() as session:
            snapshot = (
                await session.execute(
                    select(TaskCheckpointRow)
                    .where(
                        (TaskCheckpointRow.task_id == task_id)
                        & (TaskCheckpointRow.phase == "snapshot")
                    )
                    .order_by(TaskCheckpointRow.turn.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
        if snapshot is None:
            return []
        payload = msgspec.json.decode(snapshot.payload)
        if not isinstance(payload, dict):
            return []
        raw_messages = payload.get("messages", [])
        if not isinstance(raw_messages, list):
            return []
        return [message_from_jsonable(message) for message in raw_messages]

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
                    input=agent_input_from_jsonable(msgspec.json.decode(r.input_json)),
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
