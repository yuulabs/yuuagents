"""StepHandle / Fork / StepResult — the stream-tree execution model.

A step is one LLM call + tool execution round, represented as a tree:

    stem: LLM output stream (text + tool_call declarations)
    +-- fork[0]: tool_call_0 execution
    +-- fork[1]: tool_call_1 execution
    +-- fork[n]: ...

Life cycle:
    step() -> STREAMING -> FORKING -> DONE
"""

from __future__ import annotations

import asyncio
import enum
import uuid
from typing import Any

import msgspec
from attrs import define, field


# ---------------------------------------------------------------------------
# OutputBuffer (used by tools for streaming capture)
# ---------------------------------------------------------------------------


@define
class OutputBuffer:
    """Accumulates streaming output from a subprocess."""

    _chunks: list[bytes] = field(factory=list)

    def write(self, data: bytes) -> None:
        self._chunks.append(data)

    def tail(self, n_bytes: int = 4096) -> str:
        joined = b"".join(self._chunks)
        segment = joined[-n_bytes:] if len(joined) > n_bytes else joined
        return segment.decode("utf-8", errors="replace")

    def full(self) -> str:
        return b"".join(self._chunks).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StepStatus(str, enum.Enum):
    STREAMING = "streaming"
    FORKING = "forking"
    DONE = "done"


class ForkStatus(str, enum.Enum):
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class AgentLoopStatus(str, enum.Enum):
    """Whether the host should keep stepping."""

    RUNNING = "running"
    DONE = "done"


# ---------------------------------------------------------------------------
# ForkResult / StepResult
# ---------------------------------------------------------------------------


class ToolResult(msgspec.Struct, frozen=True):
    """Result of a single tool invocation."""

    tool_call_id: str
    output: Any
    error: str | None = None


class ForkResult(msgspec.Struct, frozen=True, kw_only=True):
    output: Any = ""
    error: str | None = None


class StepResult(msgspec.Struct, frozen=True, kw_only=True):
    status: AgentLoopStatus
    text: str
    tool_names: list[str]


# ---------------------------------------------------------------------------
# Fork
# ---------------------------------------------------------------------------


@define
class Fork:
    """One tool-call execution within a step."""

    id: str
    name: str
    tool_call_id: str
    _task: asyncio.Task[Any] | None = None
    _result: ForkResult | None = None
    _status: ForkStatus = ForkStatus.RUNNING
    _content: str = ""
    _done_event: asyncio.Event = field(factory=asyncio.Event)

    @property
    def status(self) -> ForkStatus:
        return self._status

    @property
    def content(self) -> str:
        return self._content

    @property
    def result(self) -> ForkResult | None:
        return self._result

    async def join(self) -> ForkResult:
        """Wait for this fork to finish and return its result."""
        await self._done_event.wait()
        assert self._result is not None
        return self._result

    def cancel(self) -> None:
        """Cancel this fork."""
        if self._status != ForkStatus.RUNNING:
            return
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._resolve(ForkResult(error="cancelled"), ForkStatus.ERROR)

    def _resolve(self, result: ForkResult, status: ForkStatus) -> None:
        if self._status != ForkStatus.RUNNING:
            return
        self._result = result
        self._status = status
        self._done_event.set()

    def _complete(self, output: Any) -> None:
        self._resolve(ForkResult(output=output), ForkStatus.DONE)

    def _fail(self, error: str) -> None:
        self._resolve(ForkResult(error=error), ForkStatus.ERROR)


# ---------------------------------------------------------------------------
# StepHandle
# ---------------------------------------------------------------------------


@define
class StepHandle:
    """Handle to an in-progress step (one LLM call + tool round)."""

    _stem_text: str = ""
    _stem_done: asyncio.Event = field(factory=asyncio.Event)
    _forks: list[Fork] = field(factory=list)
    _status: StepStatus = StepStatus.STREAMING
    _step_done: asyncio.Event = field(factory=asyncio.Event)
    _result: StepResult | None = None

    # -- Observation --

    @property
    def stem(self) -> str:
        return self._stem_text

    @property
    def forks(self) -> list[Fork]:
        return list(self._forks)

    @property
    def status(self) -> StepStatus:
        return self._status

    @property
    def result(self) -> StepResult | None:
        return self._result

    # -- Host operations --

    async def join(self) -> StepResult:
        """Wait for the entire step to complete."""
        await self._step_done.wait()
        assert self._result is not None
        return self._result

    async def join_stem(self) -> None:
        """Wait for the LLM output to finish (stem only)."""
        await self._stem_done.wait()

    # -- Internal (called by the step implementation) --

    def _append_stem(self, text: str) -> None:
        self._stem_text += text

    def _finish_stem(self) -> None:
        self._stem_done.set()
        if not self._forks:
            self._status = StepStatus.DONE
        else:
            self._status = StepStatus.FORKING

    def _add_fork(self, fork: Fork) -> None:
        self._forks.append(fork)

    def _mark_done(self, result: StepResult) -> None:
        self._result = result
        self._status = StepStatus.DONE
        self._step_done.set()


def _new_fork_id() -> str:
    return uuid.uuid4().hex[:8]
