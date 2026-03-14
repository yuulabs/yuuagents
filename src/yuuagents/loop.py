"""Single-step execution — LLM call + tool round -> StepHandle.

The host drives the loop through Session.step(); this module only provides
internal helpers plus a convenience run() wrapper.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import yuullm
import yuutrace as ytrace

from yuuagents.persistence import TaskRecorder, ToolCallDTO, ToolErrorDTO, ToolResultDTO
from yuuagents.step import (
    AgentLoopStatus,
    Fork,
    ForkResult,
    ForkStatus,
    StepHandle,
    StepResult,
    ToolResult,
    _new_fork_id,
)
from yuuagents.types import AgentStatus

if TYPE_CHECKING:
    from yuuagents.runtime_session import Session


def _write_output_buffer(session: Session, text: str) -> None:
    ctx = session.context
    buf = ctx.current_output_buffer or ctx.output_buffer
    if buf is None or not text:
        return
    buf.write(text.encode("utf-8", errors="replace"))


def _trace_llm_gen_items(items: list[Any]) -> list[Any]:
    out: list[Any] = []
    text_buf: list[str] = []
    tool_buf: list[dict[str, Any]] = []

    def flush_text() -> None:
        if not text_buf:
            return
        out.append({"type": "text", "text": "".join(text_buf)})
        text_buf.clear()

    def flush_tools() -> None:
        if not tool_buf:
            return
        out.append({"type": "tool_calls", "tool_calls": list(tool_buf)})
        tool_buf.clear()

    for item in items:
        match item:
            case yuullm.Response(item=i):
                flush_tools()
                if isinstance(i, str):
                    text_buf.append(i)
                elif isinstance(i, dict):
                    flush_text()
                    out.append(i)
                else:
                    flush_text()
                    out.append({"type": "text", "text": str(i)})
            case yuullm.ToolCall() as tc:
                flush_text()
                if tc.arguments:
                    try:
                        args: Any = json.loads(tc.arguments)
                    except Exception:
                        args = tc.arguments
                else:
                    args = {}
                tool_buf.append({"id": tc.id, "function": tc.name, "arguments": args})
            case yuullm.Reasoning():
                pass
            case _:
                flush_tools()
                flush_text()
                out.append({"type": "text", "text": str(item)})

    flush_tools()
    flush_text()
    return out


# ---------------------------------------------------------------------------
# Fork monitor — watches a fork's asyncio task and resolves it
# ---------------------------------------------------------------------------


async def _monitor_fork(fork: Fork) -> None:
    """Watch a fork's task; on completion, resolve the fork."""
    if fork._task is None:
        return
    try:
        result = await fork._task
        # result is a ToolResult
        if result.error:
            fork._fail(str(result.error))
        else:
            fork._complete(result.output)
    except asyncio.CancelledError:
        fork._resolve(ForkResult(error="cancelled"), ForkStatus.ERROR)
    except Exception as exc:
        fork._fail(f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# step() — the only public entry point
# ---------------------------------------------------------------------------


async def _step_impl(
    session: Session,
    *,
    recorder: TaskRecorder | None = None,
    chat: ytrace.ConversationContext | None = None,
) -> StepHandle:
    """Execute one LLM call + tool round. Returns a StepHandle.
    """
    handle = StepHandle()
    session.status = AgentStatus.RUNNING

    # 1. Call LLM
    gen = chat.start_llm_gen() if chat else None
    try:
        stream, store = await session.config.llm.stream(
            session.llm_history(),
            tools=session.config.tools.specs(),
        )

        items: list[Any] = []
        async for item in stream:
            items.append(item)
            match item:
                case yuullm.Response(item=i) if isinstance(i, str):
                    handle._append_stem(i)
                    _write_output_buffer(session, i)
                case yuullm.ToolCall() as tc:
                    _write_output_buffer(session, f"\n[calling {tc.name}]\n")

        if gen is not None:
            gen.log(_trace_llm_gen_items(items))

        # Record usage & cost
        usage = store.get("usage")
        if usage is not None:
            ytrace.record_llm_usage(
                ytrace.LlmUsageDelta(
                    provider=usage.provider,
                    model=usage.model,
                    request_id=usage.request_id,
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    total_tokens=usage.total_tokens,
                ),
            )
            session.last_input_tokens = usage.input_tokens
            session.total_tokens += usage.input_tokens + usage.output_tokens

        cost = store.get("cost")
        if cost is not None:
            ytrace.record_cost(
                category="llm",
                currency="USD",
                amount=cost.total_cost,
                source=cost.source,
                llm_provider=usage.provider if usage else "",
                llm_model=usage.model if usage else "",
                llm_request_id=usage.request_id if usage else None,
            )
            session.total_cost_usd += cost.total_cost
    finally:
        if gen is not None:
            gen.end()

    session.steps += 1

    # 2. Separate tool calls from text
    tool_calls: list[yuullm.ToolCall] = []
    text_parts: list[str] = []
    for item in items:
        match item:
            case yuullm.ToolCall():
                tool_calls.append(item)
            case yuullm.Response(item=i):
                if isinstance(i, str):
                    text_parts.append(i)
            case yuullm.Reasoning():
                pass

    # Build assistant message
    assistant_items: list[yuullm.Item] = []
    full_text = "".join(text_parts)
    if full_text:
        assistant_items.append(full_text)
    for tc in tool_calls:
        assistant_items.append(
            {
                "type": "tool_call",
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments,
            }
        )

    if assistant_items:
        assistant_msg = yuullm.assistant(*assistant_items)
        session.history.append(assistant_msg)
    else:
        assistant_msg = None

    if recorder is not None:
        await recorder.record_llm(
            turn=session.steps,
            history_append=assistant_msg,
            tool_calls=[
                ToolCallDTO(call_id=tc.id, name=tc.name, args_json=tc.arguments or "")
                for tc in tool_calls
            ],
        )

    handle._finish_stem()

    # 3. No tool calls -> done
    if not tool_calls:
        session.status = AgentStatus.DONE
        handle._mark_done(
            StepResult(
                status=AgentLoopStatus.DONE,
                text=full_text,
                tool_names=[],
            )
        )
        return handle

    # 4. Execute tools — create forks
    calls = []
    unknown_results: list[ToolResult] = []
    for tc in tool_calls:
        try:
            tool_obj = session.config.tools[tc.name]
        except KeyError as exc:
            unknown_results.append(
                ToolResult(tool_call_id=tc.id, output="", error=str(exc))
            )
            continue
        bound = tool_obj.bind(session.context)
        params = json.loads(tc.arguments) if tc.arguments else {}
        calls.append((tc, bound, params))

    # Handle unknown tool results immediately
    for r in unknown_results:
        session.history.append(yuullm.tool(r.tool_call_id, str(r.error)))

    # Execute all known tools concurrently as forks
    for tc, bound, params in calls:
        fork_id = _new_fork_id()
        task = asyncio.create_task(
            _run_tool(bound, params, tc.id),
            name=f"fork-{tc.name}-{fork_id}",
        )
        fork = Fork(
            id=fork_id,
            name=tc.name,
            tool_call_id=tc.id,
            task=task,
        )
        handle._add_fork(fork)
        asyncio.create_task(_monitor_fork(fork))

    # Spawn background finalizer — waits for forks, appends results, marks done
    asyncio.create_task(_finalize_step(handle, session, full_text, recorder))

    return handle


async def _finalize_step(
    handle: StepHandle,
    session: Session,
    full_text: str,
    recorder: TaskRecorder | None,
) -> None:
    """Background task: wait for all forks, append results to history, mark done."""
    await asyncio.gather(*[f.join() for f in handle.forks])

    tool_names: list[str] = []
    for fork in handle.forks:
        tool_names.append(fork.name)
        if fork.result is not None:
            if fork.result.error:
                session.history.append(
                    yuullm.tool(fork.tool_call_id, str(fork.result.error))
                )
            elif isinstance(fork.result.output, list):
                session.history.append(
                    yuullm.tool(fork.tool_call_id, fork.result.output)
                )
            else:
                session.history.append(
                    yuullm.tool(fork.tool_call_id, str(fork.result.output))
                )

    if recorder is not None:
        await recorder.record_tool(
            turn=session.steps,
            results=[
                ToolResultDTO(
                    call_id=fork.tool_call_id,
                    ok=fork.result is not None and fork.result.error is None,
                    output_text=str(fork.result.output)
                    if fork.result and fork.result.error is None
                    else "",
                    error=(
                        None
                        if fork.result is None or fork.result.error is None
                        else ToolErrorDTO(
                            type="ToolError", message=str(fork.result.error)
                        )
                    ),
                )
                for fork in handle.forks
            ],
        )

    session.status = AgentStatus.RUNNING
    handle._mark_done(
        StepResult(
            status=AgentLoopStatus.RUNNING,
            text=full_text,
            tool_names=tool_names,
        )
    )


async def run(
    session: Session,
    *,
    task: str | None = None,
    recorder: TaskRecorder | None = None,
) -> None:
    """Drive a session until completion."""
    if task is not None:
        session.init([task])
    while True:
        handle = await session.step(recorder=recorder)
        result = await handle.join()
        if result.status == AgentLoopStatus.DONE:
            return
        if session.config.max_steps and session.steps >= session.config.max_steps:
            session.status = AgentStatus.DONE
            return


async def _run_tool(
    bound: Any,
    params: dict[str, Any],
    tool_call_id: str,
) -> ToolResult:
    """Run a single tool and return a ToolResult."""
    try:
        output = await bound.run(**params)
        return ToolResult(tool_call_id=tool_call_id, output=output)
    except Exception as exc:
        return ToolResult(
            tool_call_id=tool_call_id,
            output="",
            error=f"{type(exc).__name__}: {exc}",
        )
