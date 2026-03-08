"""E2E tests for soft-timeout / running tools infrastructure."""

from __future__ import annotations

import asyncio

import pytest

from yuuagents.running_tools import OutputBuffer, RunningToolRegistry
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
# Registry: check blocks until tool finishes
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_check_returns_immediately_on_done():
    registry = RunningToolRegistry()
    buf = OutputBuffer()

    async def quick_tool() -> ToolResult:
        return ToolResult(tool_call_id="tc1", output="done!")

    task = asyncio.create_task(quick_tool())
    await task  # let it finish
    handle = registry.register("test_tool", task, buf, "tc1")
    result = await registry.check(handle, wait=1)
    assert "done!" in result


@pytest.mark.asyncio
async def test_check_blocks_then_returns():
    """Tool finishes after 0.5s, check(wait=5) should return in ~0.5s, not 5s."""
    registry = RunningToolRegistry()
    buf = OutputBuffer()

    async def slow_tool() -> ToolResult:
        await asyncio.sleep(0.5)
        return ToolResult(tool_call_id="tc2", output="finished")

    task = asyncio.create_task(slow_tool())
    handle = registry.register("slow", task, buf, "tc2")
    result = await registry.check(handle, wait=5)
    assert "finished" in result


@pytest.mark.asyncio
async def test_check_returns_tail_on_timeout():
    """Tool doesn't finish within wait → returns tail output."""
    registry = RunningToolRegistry()
    buf = OutputBuffer()
    buf.write(b"partial output line 1\n")

    async def very_slow() -> ToolResult:
        await asyncio.sleep(100)
        return ToolResult(tool_call_id="tc3", output="never")

    task = asyncio.create_task(very_slow())
    handle = registry.register("very_slow", task, buf, "tc3")
    result = await registry.check(handle, wait=0.3)
    assert "Still running" in result
    assert "partial output" in result
    # Clean up
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# Cancel
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cancel():
    registry = RunningToolRegistry()
    buf = OutputBuffer()

    async def forever() -> ToolResult:
        await asyncio.sleep(9999)
        return ToolResult(tool_call_id="tc4", output="nope")

    task = asyncio.create_task(forever())
    handle = registry.register("forever", task, buf, "tc4")
    msg = registry.cancel(handle)
    assert "Cancelled" in msg
    # Task should be cancelled
    await asyncio.sleep(0.05)
    assert task.cancelled()


# ---------------------------------------------------------------------------
# collect_finished
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_collect_finished():
    registry = RunningToolRegistry()
    buf = OutputBuffer()

    async def fast() -> ToolResult:
        return ToolResult(tool_call_id="tc5", output="ok")

    task = asyncio.create_task(fast())
    handle = registry.register("fast", task, buf, "tc5")
    await asyncio.sleep(0.05)  # let it finish

    finished = registry.collect_finished()
    assert len(finished) == 1
    assert finished[0].handle == handle
    # Should be removed now
    assert registry.collect_finished() == []


# ---------------------------------------------------------------------------
# gather() with soft_timeout
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_gather_soft_timeout():
    """gather() with soft_timeout registers pending tools and returns placeholder."""
    from yuutrace.context import ToolsContext
    from opentelemetry import trace

    tracer = trace.get_tracer("test")
    registry = RunningToolRegistry()

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
            registry=registry,
        )

    assert len(results) == 2
    # fast_fn should have completed
    r_fast = next(r for r in results if r.tool_call_id == "c2")
    assert r_fast.output == "fast result"
    # slow_fn should be a placeholder
    r_slow = next(r for r in results if r.tool_call_id == "c1")
    assert "still running" in str(r_slow.output)
    assert "handle=" in str(r_slow.output)

    # Clean up pending task
    finished = registry.collect_finished()
    for entry in registry._entries.values():
        entry.task.cancel()
    for entry in list(registry._entries.values()):
        try:
            await entry.task
        except asyncio.CancelledError:
            pass
