"""Flow / Ping / FlowManager — unified runtime abstraction.

A Flow represents any running unit: root agent, tool, or subagent.
Flows form a tree; children notify parents via Ping messages.
FlowManager owns the tree and provides creation, completion, and query.
"""

from __future__ import annotations

import asyncio
import enum
import time
import uuid
from typing import TYPE_CHECKING

import msgspec
from attrs import define, field

from yuutrace.context import ToolResult


# ---------------------------------------------------------------------------
# OutputBuffer (moved from running_tools.py)
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


class FlowKind(enum.Enum):
    AGENT = "agent"
    TOOL = "tool"


class FlowStatus(enum.Enum):
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


class PingKind(enum.Enum):
    CHILD_COMPLETED = "child_completed"
    CHILD_FAILED = "child_failed"
    TOOL_OUTPUT = "tool_output"
    USER_MESSAGE = "user_message"
    CANCEL = "cancel"
    SYSTEM_NOTE = "system_note"


# ---------------------------------------------------------------------------
# Ping
# ---------------------------------------------------------------------------


class Ping(msgspec.Struct, frozen=True, kw_only=True):
    """Event passed between flows."""

    kind: PingKind
    source_flow_id: str
    payload: str = ""


# ---------------------------------------------------------------------------
# Flow
# ---------------------------------------------------------------------------


@define
class Flow:
    """A single running unit in the flow tree."""

    flow_id: str
    kind: FlowKind
    name: str
    parent_flow_id: str | None = None
    status: FlowStatus = FlowStatus.RUNNING
    task: asyncio.Task[ToolResult] | None = None
    result: ToolResult | None = None
    error: str | None = None
    tool_call_id: str = ""
    started: float = field(factory=time.monotonic)
    _children: list[str] = field(factory=list)
    _ping_queue: asyncio.Queue[Ping] = field(factory=asyncio.Queue)
    _output_buffer: OutputBuffer = field(factory=OutputBuffer)

    def ping(self, p: Ping) -> None:
        """Enqueue a ping for this flow."""
        self._ping_queue.put_nowait(p)

    async def recv(self, timeout: float | None = None) -> Ping | None:
        """Wait for the next ping. Returns None on timeout."""
        try:
            if timeout is None:
                return await self._ping_queue.get()
            return await asyncio.wait_for(self._ping_queue.get(), timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            return None


# ---------------------------------------------------------------------------
# FlowManager
# ---------------------------------------------------------------------------


@define
class FlowManager:
    """Owns the flow tree for one agent run."""

    _flows: dict[str, Flow] = field(factory=dict)

    def create(
        self,
        kind: FlowKind,
        name: str,
        *,
        parent_flow_id: str | None = None,
        task: asyncio.Task[ToolResult] | None = None,
        tool_call_id: str = "",
    ) -> Flow:
        flow_id = uuid.uuid4().hex[:8]
        flow = Flow(
            flow_id=flow_id,
            kind=kind,
            name=name,
            parent_flow_id=parent_flow_id,
            task=task,
            tool_call_id=tool_call_id,
        )
        self._flows[flow_id] = flow
        if parent_flow_id and parent_flow_id in self._flows:
            self._flows[parent_flow_id]._children.append(flow_id)
        return flow

    def get(self, flow_id: str) -> Flow | None:
        return self._flows.get(flow_id)

    def complete(self, flow_id: str, result: ToolResult) -> None:
        """Mark a flow as done and ping its parent."""
        flow = self._flows.get(flow_id)
        if flow is None:
            return
        flow.status = FlowStatus.DONE
        flow.result = result
        if flow.parent_flow_id:
            parent = self._flows.get(flow.parent_flow_id)
            if parent is not None:
                parent.ping(Ping(
                    kind=PingKind.CHILD_COMPLETED,
                    source_flow_id=flow_id,
                    payload=str(result.output) if result.error is None else str(result.error),
                ))

    def fail(self, flow_id: str, error: str) -> None:
        """Mark a flow as errored and ping its parent."""
        flow = self._flows.get(flow_id)
        if flow is None:
            return
        flow.status = FlowStatus.ERROR
        flow.error = error
        if flow.parent_flow_id:
            parent = self._flows.get(flow.parent_flow_id)
            if parent is not None:
                parent.ping(Ping(
                    kind=PingKind.CHILD_FAILED,
                    source_flow_id=flow_id,
                    payload=error,
                ))

    def cancel(self, flow_id: str) -> str:
        """Cancel a flow: cancel its asyncio task, mark cancelled, ping parent."""
        flow = self._flows.get(flow_id)
        if flow is None:
            return f"[ERROR] unknown flow {flow_id!r}"
        if flow.status in (FlowStatus.DONE, FlowStatus.ERROR, FlowStatus.CANCELLED):
            return f"Flow already {flow.status.value}."
        if flow.task and not flow.task.done():
            flow.task.cancel()
        flow.status = FlowStatus.CANCELLED
        if flow.parent_flow_id:
            parent = self._flows.get(flow.parent_flow_id)
            if parent is not None:
                parent.ping(Ping(
                    kind=PingKind.CHILD_FAILED,
                    source_flow_id=flow_id,
                    payload="cancelled",
                ))
        return "Cancelled."

    def has_running_children(self, flow_id: str) -> bool:
        flow = self._flows.get(flow_id)
        if flow is None:
            return False
        for child_id in flow._children:
            child = self._flows.get(child_id)
            if child and child.status in (FlowStatus.RUNNING, FlowStatus.WAITING_INPUT):
                return True
        return False

    def running_children_count(self, flow_id: str) -> int:
        flow = self._flows.get(flow_id)
        if flow is None:
            return 0
        return sum(
            1
            for cid in flow._children
            if (c := self._flows.get(cid)) and c.status in (FlowStatus.RUNNING, FlowStatus.WAITING_INPUT)
        )

    def collect_completed_children(self, flow_id: str) -> list[Flow]:
        """Return children that finished since last collection, remove them."""
        flow = self._flows.get(flow_id)
        if flow is None:
            return []
        completed: list[Flow] = []
        remaining: list[str] = []
        for child_id in flow._children:
            child = self._flows.get(child_id)
            if child is None:
                continue
            if child.status in (FlowStatus.DONE, FlowStatus.ERROR, FlowStatus.CANCELLED):
                completed.append(child)
                self._flows.pop(child_id, None)
            else:
                remaining.append(child_id)
        flow._children = remaining
        return completed


# ---------------------------------------------------------------------------
# Ping formatting
# ---------------------------------------------------------------------------


def format_ping(ping: Ping, flow_manager: FlowManager) -> str:
    """Render a Ping as a human-readable line for LLM history."""
    match ping.kind:
        case PingKind.CHILD_COMPLETED:
            child = flow_manager.get(ping.source_flow_id)
            name = child.name if child else "unknown"
            elapsed = time.monotonic() - child.started if child else 0
            return (f"[完成] {name} (handle={ping.source_flow_id}, "
                    f"耗时 {elapsed:.0f}s): {ping.payload[:200]}")
        case PingKind.CHILD_FAILED:
            child = flow_manager.get(ping.source_flow_id)
            name = child.name if child else "unknown"
            elapsed = time.monotonic() - child.started if child else 0
            return (f"[失败] {name} (handle={ping.source_flow_id}, "
                    f"耗时 {elapsed:.0f}s): {ping.payload[:200]}")
        case PingKind.USER_MESSAGE:
            return f"[用户消息] {ping.payload}"
        case PingKind.SYSTEM_NOTE:
            return f"[系统] {ping.payload}"
        case _:
            return f"[{ping.kind.value}] {ping.payload}"
