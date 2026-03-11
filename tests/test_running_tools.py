"""E2E tests for Flow / Ping / FlowManager infrastructure."""

from __future__ import annotations

import asyncio

import pytest

from yuuagents.flow import (
    FlowKind,
    FlowManager,
    FlowStatus,
    OutputBuffer,
    Ping,
    PingKind,
)
from yuutrace.context import ToolResult


# ---------------------------------------------------------------------------
# OutputBuffer
# ---------------------------------------------------------------------------

def test_output_buffer_tail():
    buf = OutputBuffer()
    buf.write(b"hello ")
    buf.write(b"world")
    assert buf.full() == "hello world"
    assert buf.tail(5) == "world"
    assert buf.tail(100) == "hello world"


# ---------------------------------------------------------------------------
# FlowManager: create / get / complete / fail / cancel
# ---------------------------------------------------------------------------


def test_flow_manager_create_and_get():
    fm = FlowManager()
    flow = fm.create(FlowKind.AGENT, "root")
    assert flow.kind == FlowKind.AGENT
    assert flow.name == "root"
    assert flow.status == FlowStatus.RUNNING
    assert fm.get(flow.flow_id) is flow
    assert fm.get("nonexistent") is None


def test_flow_manager_parent_child():
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")
    child = fm.create(FlowKind.TOOL, "bash", parent_flow_id=root.flow_id)
    assert child.parent_flow_id == root.flow_id
    assert child.flow_id in root._children


@pytest.mark.asyncio
async def test_complete_pings_parent():
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")
    child = fm.create(FlowKind.TOOL, "bash", parent_flow_id=root.flow_id)

    result = ToolResult(tool_call_id="tc1", output="done!")
    fm.complete(child.flow_id, result)

    assert child.status == FlowStatus.DONE
    assert child.result is result

    # Parent should have received a ping
    ping = await root.recv(timeout=0.1)
    assert ping is not None
    assert ping.kind == PingKind.CHILD_COMPLETED
    assert ping.source_flow_id == child.flow_id
    assert "done!" in ping.payload


@pytest.mark.asyncio
async def test_fail_pings_parent():
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")
    child = fm.create(FlowKind.TOOL, "bash", parent_flow_id=root.flow_id)

    fm.fail(child.flow_id, "something broke")

    assert child.status == FlowStatus.ERROR
    assert child.error == "something broke"

    ping = await root.recv(timeout=0.1)
    assert ping is not None
    assert ping.kind == PingKind.CHILD_FAILED
    assert "something broke" in ping.payload


@pytest.mark.asyncio
async def test_cancel_cancels_task_and_pings_parent():
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")

    async def forever() -> ToolResult:
        await asyncio.sleep(9999)
        return ToolResult(tool_call_id="tc1", output="nope")

    task = asyncio.create_task(forever())
    child = fm.create(
        FlowKind.TOOL, "forever",
        parent_flow_id=root.flow_id, task=task, tool_call_id="tc1",
    )

    msg = fm.cancel(child.flow_id)
    assert "Cancelled" in msg
    assert child.status == FlowStatus.CANCELLED

    await asyncio.sleep(0.05)
    assert task.cancelled()

    ping = await root.recv(timeout=0.1)
    assert ping is not None
    assert ping.kind == PingKind.CHILD_FAILED
    assert "cancelled" in ping.payload


# ---------------------------------------------------------------------------
# has_running_children / collect_completed_children
# ---------------------------------------------------------------------------


def test_has_running_children():
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")
    assert not fm.has_running_children(root.flow_id)

    child = fm.create(FlowKind.TOOL, "bash", parent_flow_id=root.flow_id)
    assert fm.has_running_children(root.flow_id)

    child.status = FlowStatus.DONE
    assert not fm.has_running_children(root.flow_id)


def test_collect_completed_children():
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")
    c1 = fm.create(FlowKind.TOOL, "t1", parent_flow_id=root.flow_id)
    c2 = fm.create(FlowKind.TOOL, "t2", parent_flow_id=root.flow_id)

    c1.status = FlowStatus.DONE
    completed = fm.collect_completed_children(root.flow_id)
    assert len(completed) == 1
    assert completed[0].flow_id == c1.flow_id

    # c1 removed from tree, c2 still there
    assert fm.get(c1.flow_id) is None
    assert fm.get(c2.flow_id) is c2
    assert len(root._children) == 1


# ---------------------------------------------------------------------------
# Flow ping / recv
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_flow_ping_recv():
    fm = FlowManager()
    flow = fm.create(FlowKind.AGENT, "root")

    flow.ping(Ping(kind=PingKind.SYSTEM_NOTE, source_flow_id="sys", payload="hello"))
    ping = await flow.recv(timeout=0.1)
    assert ping is not None
    assert ping.kind == PingKind.SYSTEM_NOTE
    assert ping.payload == "hello"


@pytest.mark.asyncio
async def test_flow_recv_timeout_returns_none():
    fm = FlowManager()
    flow = fm.create(FlowKind.AGENT, "root")
    result = await flow.recv(timeout=0.05)
    assert result is None


# ---------------------------------------------------------------------------
# gather() with soft_timeout + on_pending (replaces old registry test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gather_soft_timeout_creates_flow():
    """gather() with soft_timeout calls on_pending which creates child flows."""
    from yuutrace.context import ToolsContext
    from opentelemetry import trace

    tracer = trace.get_tracer("test")
    fm = FlowManager()
    root = fm.create(FlowKind.AGENT, "root")
    created_flows: list[str] = []

    def on_pending(
        name: str,
        task: asyncio.Task,
        buffer: OutputBuffer,
        tool_call_id: str,
    ) -> str:
        child = fm.create(
            FlowKind.TOOL, name,
            parent_flow_id=root.flow_id,
            task=task, tool_call_id=tool_call_id,
        )
        created_flows.append(child.flow_id)
        return child.flow_id

    async def slow_fn() -> str:
        await asyncio.sleep(10)
        return "done"

    async def fast_fn() -> str:
        return "fast result"

    with tracer.start_as_current_span("test") as span:
        ctx = ToolsContext(span, tracer)
        results = await ctx.gather(
            [
                {"tool_call_id": "c1", "name": "slow_fn", "tool": slow_fn},
                {"tool_call_id": "c2", "name": "fast_fn", "tool": fast_fn},
            ],
            soft_timeout=0.2,
            on_pending=on_pending,
        )

    assert len(results) == 2
    # fast_fn should have completed
    r_fast = next(r for r in results if r.tool_call_id == "c2")
    assert r_fast.output == "fast result"
    # slow_fn should be a placeholder
    r_slow = next(r for r in results if r.tool_call_id == "c1")
    assert "still running" in str(r_slow.output)

    # A child flow should have been created
    assert len(created_flows) == 1
    child = fm.get(created_flows[0])
    assert child is not None
    assert child.name == "slow_fn"

    # Clean up
    fm.cancel(created_flows[0])
    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# _merge_user_pings
# ---------------------------------------------------------------------------


def test_merge_user_pings_combines_consecutive_user_messages():
    from yuuagents.loop import _merge_user_pings

    pings = [
        Ping(kind=PingKind.USER_MESSAGE, source_flow_id="a", payload="hello"),
        Ping(kind=PingKind.USER_MESSAGE, source_flow_id="b", payload="world"),
    ]
    result = _merge_user_pings(pings)
    assert len(result) == 1
    assert result[0].kind == PingKind.USER_MESSAGE
    assert result[0].payload == "hello\nworld"
    # source_flow_id comes from the first ping
    assert result[0].source_flow_id == "a"


def test_merge_user_pings_combines_system_note_with_user():
    from yuuagents.loop import _merge_user_pings

    pings = [
        Ping(kind=PingKind.USER_MESSAGE, source_flow_id="a", payload="msg"),
        Ping(kind=PingKind.SYSTEM_NOTE, source_flow_id="sys", payload="note"),
    ]
    result = _merge_user_pings(pings)
    assert len(result) == 1
    assert result[0].payload == "msg\nnote"


def test_merge_user_pings_preserves_non_user_pings():
    from yuuagents.loop import _merge_user_pings

    pings = [
        Ping(kind=PingKind.USER_MESSAGE, source_flow_id="a", payload="hello"),
        Ping(kind=PingKind.CHILD_COMPLETED, source_flow_id="c1", payload="done"),
        Ping(kind=PingKind.USER_MESSAGE, source_flow_id="b", payload="world"),
    ]
    result = _merge_user_pings(pings)
    assert len(result) == 3
    assert result[0].kind == PingKind.USER_MESSAGE
    assert result[0].payload == "hello"
    assert result[1].kind == PingKind.CHILD_COMPLETED
    assert result[2].kind == PingKind.USER_MESSAGE
    assert result[2].payload == "world"


def test_merge_user_pings_empty():
    from yuuagents.loop import _merge_user_pings

    assert _merge_user_pings([]) == []


# ---------------------------------------------------------------------------
# Loop-level ping delivery: external ping reaches agent history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_ping_reaches_agent_history():
    """External USER_MESSAGE ping via root_flow appears in agent.history."""
    import yuullm
    from unittest.mock import AsyncMock, MagicMock
    from yuuagents.loop import _drain_pings, _merge_user_pings, _apply_ping

    fm = FlowManager()
    root_flow = fm.create(FlowKind.AGENT, name="test-agent")

    # Simulate external ping (what dispatcher would do)
    root_flow.ping(Ping(
        kind=PingKind.USER_MESSAGE,
        source_flow_id="dispatcher",
        payload="<msg>user追加消息</msg>",
    ))

    # Simulate what the loop does at the top of each step
    pings = _drain_pings(root_flow)
    assert len(pings) == 1

    merged = _merge_user_pings(pings)
    assert len(merged) == 1
    assert merged[0].kind == PingKind.USER_MESSAGE

    # Apply to a mock agent
    agent = MagicMock()
    agent.history = []
    _apply_ping(agent, merged[0], fm)

    # Should have appended a user message to history
    assert len(agent.history) == 1
    role, items = agent.history[0]
    assert role == "user"
    assert "<msg>user追加消息</msg>" in str(items)
