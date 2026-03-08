"""RunningToolRegistry — track tools that outlive their soft timeout."""

from __future__ import annotations

import asyncio
import time
import uuid

from attrs import define, field

from yuutrace.context import ToolResult


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


@define
class RunningToolEntry:
    handle: str
    name: str
    task: asyncio.Task[ToolResult]
    buffer: OutputBuffer
    started: float  # time.monotonic()
    tool_call_id: str


@define
class RunningToolRegistry:
    """Holds references to tools still running after soft timeout."""

    _entries: dict[str, RunningToolEntry] = field(factory=dict)

    def register(
        self,
        name: str,
        task: asyncio.Task[ToolResult],
        buffer: OutputBuffer,
        tool_call_id: str,
    ) -> str:
        handle = uuid.uuid4().hex[:8]
        self._entries[handle] = RunningToolEntry(
            handle=handle,
            name=name,
            task=task,
            buffer=buffer,
            started=time.monotonic(),
            tool_call_id=tool_call_id,
        )
        return handle

    async def check(self, handle: str, wait: float = 120) -> str:
        entry = self._entries.get(handle)
        if entry is None:
            return f"[ERROR] unknown handle {handle!r}"
        if entry.task.done():
            return self._format_done(entry)
        try:
            await asyncio.wait_for(asyncio.shield(entry.task), timeout=wait)
        except (TimeoutError, asyncio.TimeoutError):
            elapsed = time.monotonic() - entry.started
            tail = entry.buffer.tail()
            return (
                f"Still running ({elapsed:.0f}s). handle={handle}\n"
                f"Tail output:\n{tail}"
            )
        return self._format_done(entry)

    def cancel(self, handle: str) -> str:
        entry = self._entries.get(handle)
        if entry is None:
            return f"[ERROR] unknown handle {handle!r}"
        if entry.task.done():
            return "Tool already finished."
        entry.task.cancel()
        self._entries.pop(handle, None)
        return "Cancelled."

    def collect_finished(self) -> list[RunningToolEntry]:
        """Remove and return entries whose tasks have completed."""
        finished = [e for e in self._entries.values() if e.task.done()]
        for e in finished:
            self._entries.pop(e.handle, None)
        return finished

    def _format_done(self, entry: RunningToolEntry) -> str:
        self._entries.pop(entry.handle, None)
        try:
            result = entry.task.result()
            if result.error:
                return f"[ERROR] {result.error}"
            return str(result.output)
        except asyncio.CancelledError:
            return "Tool was cancelled."
        except Exception as exc:
            return f"[ERROR] {type(exc).__name__}: {exc}"
