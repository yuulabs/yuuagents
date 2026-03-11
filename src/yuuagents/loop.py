"""Step loop — LLM → tool calls → trace."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from uuid import UUID

import yuullm
import yuutrace as ytrace

from yuuagents.agent import Agent
from yuuagents.context import AgentContext
from yuuagents.flow import (
    Flow,
    FlowKind,
    FlowManager,
    FlowStatus,
    OutputBuffer,
    Ping,
    PingKind,
    format_ping,
)
from yuuagents.persistence import TaskRecorder, ToolCallDTO, ToolErrorDTO, ToolResultDTO
from yuuagents.types import AgentStatus


def _write_output_buffer(ctx: AgentContext, text: str) -> None:
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


def _has_im_send(results: list[ytrace.ToolResult], tool_calls: list[Any]) -> bool:
    """Check if any tool call in this step was an im send."""
    for tc in tool_calls:
        if hasattr(tc, "name") and "im" in tc.name and "send" in tc.name:
            return True
        # Also check execute_skill_cli calls that contain "im send"
        if hasattr(tc, "arguments") and tc.arguments:
            try:
                args = json.loads(tc.arguments)
                cmd = args.get("command", "")
                if isinstance(cmd, str) and "im send" in cmd:
                    return True
            except Exception:
                pass
    return False


# ---------------------------------------------------------------------------
# Ping helpers
# ---------------------------------------------------------------------------


def _drain_pings(flow: Any) -> list[Ping]:
    """Non-blocking drain of all pending pings from a flow's queue."""
    pings: list[Ping] = []
    while True:
        try:
            pings.append(flow._ping_queue.get_nowait())
        except asyncio.QueueEmpty:
            break
    return pings


_MERGEABLE_KINDS = {PingKind.USER_MESSAGE, PingKind.SYSTEM_NOTE}


def _merge_user_pings(pings: list[Ping]) -> list[Ping]:
    """Merge consecutive USER_MESSAGE/SYSTEM_NOTE pings into a single USER_MESSAGE ping."""
    result: list[Ping] = []
    i = 0
    while i < len(pings):
        p = pings[i]
        if p.kind in _MERGEABLE_KINDS:
            parts = [p.payload]
            first_source = p.source_flow_id
            j = i + 1
            while j < len(pings) and pings[j].kind in _MERGEABLE_KINDS:
                parts.append(pings[j].payload)
                j += 1
            result.append(Ping(
                kind=PingKind.USER_MESSAGE,
                source_flow_id=first_source,
                payload="\n".join(parts),
            ))
            i = j
        else:
            result.append(p)
            i += 1
    return result


def _apply_ping(agent: Any, ping: Ping, flow_manager: FlowManager) -> None:
    """Append a ping to the agent's history as a user message."""
    agent.history.append(yuullm.user(ping.payload))


# ---------------------------------------------------------------------------
# Monitor coroutine for tool flows
# ---------------------------------------------------------------------------


async def _monitor_tool_flow(flow_manager: FlowManager, flow_id: str) -> None:
    """Watch a tool flow's asyncio task; on completion, call flow_manager.complete/fail."""
    flow = flow_manager.get(flow_id)
    if flow is None or flow.task is None:
        return
    try:
        result = await flow.task
        flow_manager.complete(flow_id, result)
    except asyncio.CancelledError:
        flow_manager.cancel(flow_id)
    except Exception as exc:
        flow_manager.fail(flow_id, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


async def run(
    agent: Agent,
    task: str,
    ctx: AgentContext,
    *,
    recorder: TaskRecorder | None = None,
    resume: bool = False,
    flow_manager: FlowManager | None = None,
    root_flow: Flow | None = None,
) -> None:
    """Run the agent loop until completion or error."""
    if not resume:
        agent.setup(task)

    # Set up flow manager
    if flow_manager is None:
        flow_manager = FlowManager()
    ctx.flow_manager = flow_manager

    if root_flow is None:
        root_flow = flow_manager.create(FlowKind.AGENT, name=agent.agent_id)
    ctx.root_flow = root_flow
    ctx.current_flow_id = root_flow.flow_id

    with ytrace.conversation(
        id=UUID(agent.task_id),
        agent=agent.agent_id,
        model=agent.llm.default_model,
    ) as chat:
        chat.system(persona=agent.full_system_prompt, tools=agent.tools.specs())
        chat.user(task if not resume else agent.task)

        last_user_msg_time = time.monotonic()
        silence_interval_first = agent.silence_timeout or 0
        silence_interval_subsequent = max(silence_interval_first * 2.5, 300)

        while not agent.done():
            if agent.max_steps and agent.steps >= agent.max_steps:
                agent.status = AgentStatus.DONE
                break

            # Silence detection → inject user message directly
            if silence_interval_first > 0:
                elapsed_silent = time.monotonic() - last_user_msg_time
                threshold = (
                    silence_interval_first
                    if elapsed_silent < silence_interval_subsequent
                    else silence_interval_subsequent
                )
                if elapsed_silent > threshold:
                    silence_msg = (
                        f"[system] 你已经工作了 {elapsed_silent:.0f}s 没有给用户发送任何消息。"
                        "请通过 im send 告知用户当前进度和预计剩余时间。"
                    )
                    agent.history.append(yuullm.user(silence_msg))
                    chat.user(silence_msg)
                    last_user_msg_time = time.monotonic()

            try:
                tool_calls = await _step(
                    agent, chat, ctx, flow_manager=flow_manager,
                    root_flow=root_flow, recorder=recorder,
                )
                # Track im send for silence detection
                if tool_calls and _has_im_send([], tool_calls):
                    last_user_msg_time = time.monotonic()

                # After a text-only step, drain any pings that arrived while the
                # LLM was generating.  This handles the race where a USER_MESSAGE
                # ping lands in the queue just as the agent finishes its last
                # tool-call round and the LLM decides to emit text only.
                # Without this drain the ping would be discarded when the loop exits.
                if not tool_calls and agent.done():
                    pending_pings = _drain_pings(root_flow)
                    if pending_pings:
                        lines = [format_ping(p, flow_manager) for p in pending_pings]
                        summary = "[system] 后台通知：\n" + "\n".join(lines)
                        agent.history.append(yuullm.user(summary))
                        chat.user(summary)
                        agent.status = AgentStatus.RUNNING  # keep the loop alive

                # LLM emitted text only but children are still running —
                # block until a child completes or a user message arrives.
                if (
                    not tool_calls
                    and not agent.done()
                    and flow_manager.has_running_children(root_flow.flow_id)
                ):
                    ping = await root_flow.recv(timeout=300.0)
                    if ping is not None:
                        ping_msg = "[system] 后台通知：\n" + format_ping(ping, flow_manager)
                        agent.history.append(yuullm.user(ping_msg))
                        chat.user(ping_msg)
                    else:
                        # Timed out waiting — give up
                        agent.status = AgentStatus.DONE

            except BaseException as exc:
                agent.fail(exc)
                raise


async def _step(
    agent: Agent,
    chat: ytrace.ConversationContext,
    ctx: AgentContext,
    *,
    flow_manager: FlowManager,
    root_flow: Any,
    recorder: TaskRecorder | None = None,
) -> list[yuullm.ToolCall]:
    """Execute one LLM call + tool round. Returns tool_calls from this step."""
    # 1. Call LLM
    with chat.llm_gen() as gen:
        stream, store = await agent.llm.stream(
            agent.history,
            tools=agent.tools.specs(),
        )

        items: list[Any] = []
        async for item in stream:
            items.append(item)
            match item:
                case yuullm.Response(item=i) if isinstance(i, str):
                    _write_output_buffer(ctx, i)
                case yuullm.ToolCall() as tc:
                    _write_output_buffer(ctx, f"\n[calling {tc.name}]\n")
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
            agent.total_tokens += usage.input_tokens + usage.output_tokens

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
            agent.total_cost_usd += cost.total_cost

    agent.steps += 1

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
                pass  # reasoning is logged but not appended

    # Build assistant message items
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
        agent.history.append(assistant_msg)
    else:
        assistant_msg = None

    if recorder is not None:
        await recorder.record_llm(
            turn=agent.steps,
            history_append=assistant_msg,
            tool_calls=[
                ToolCallDTO(call_id=tc.id, name=tc.name, args_json=tc.arguments or "")
                for tc in tool_calls
            ],
        )

    # 3. If no tool calls, check whether we can truly stop.
    #    Running children (e.g. soft-timed-out tools / delegates) keep
    #    the loop alive — the agent must wait for them to finish.
    if not tool_calls:
        if flow_manager.has_running_children(root_flow.flow_id):
            agent.status = AgentStatus.RUNNING
        else:
            agent.status = AgentStatus.DONE
            agent.state.pending_input_prompt = ""
        return tool_calls

    # 4. Execute tools — on_pending creates child flows for timed-out tools
    def _on_pending(
        name: str,
        task: asyncio.Task,  # type: ignore[type-arg]
        buffer: OutputBuffer,
        tool_call_id: str,
    ) -> str:
        child = flow_manager.create(
            FlowKind.TOOL,
            name=name,
            parent_flow_id=root_flow.flow_id,
            task=task,
            tool_call_id=tool_call_id,
        )
        child._output_buffer = buffer
        # Launch monitor coroutine that will ping parent on completion
        asyncio.create_task(_monitor_tool_flow(flow_manager, child.flow_id))
        return child.flow_id

    with chat.tools() as tools_ctx:
        calls = []
        buffers: dict[str, OutputBuffer] = {}
        for tc in tool_calls:
            tool_obj = agent.tools[tc.name]
            # Create per-call output buffer for streaming capture
            buf = OutputBuffer()
            ctx.current_output_buffer = buf
            bound = tool_obj.bind(ctx)
            ctx.current_output_buffer = None
            params = json.loads(tc.arguments) if tc.arguments else {}
            buffers[tc.id] = buf
            calls.append(
                {
                    "tool_call_id": tc.id,
                    "name": tc.name,
                    "tool": bound.run,
                    "params": params,
                }
            )
        # Exempt sleep from soft_timeout — it manages its own waiting
        effective_soft_timeout = agent.soft_timeout
        if all(tc.name == "sleep" for tc in tool_calls):
            effective_soft_timeout = None

        results = await tools_ctx.gather(
            calls,
            soft_timeout=effective_soft_timeout,
            on_pending=_on_pending,
            buffers=buffers,
        )

    # 5. Append tool results to history
    for r in results:
        if r.error:
            content = str(r.error)
        elif isinstance(r.output, list):
            content = r.output  # multimodal content blocks — passthrough
        else:
            content = str(r.output)
        agent.history.append(yuullm.tool(r.tool_call_id, content))
    if agent.status not in (AgentStatus.DONE, AgentStatus.ERROR, AgentStatus.CANCELLED):
        agent.status = AgentStatus.RUNNING

    if recorder is not None:
        await recorder.record_tool(
            turn=agent.steps,
            results=[
                ToolResultDTO(
                    call_id=r.tool_call_id,
                    ok=r.error is None,
                    output_text=str(r.output) if r.error is None else "",
                    error=(
                        None
                        if r.error is None
                        else ToolErrorDTO(
                            type=type(r.error).__name__,
                            message=str(r.error),
                        )
                    ),
                )
                for r in results
            ],
        )

    # 5b. Path 2: if sleep was NOT called, drain pending pings as user message
    if not any(tc.name == "sleep" for tc in tool_calls):
        pending_pings = _drain_pings(root_flow)
        if pending_pings:
            lines = [format_ping(p, flow_manager) for p in pending_pings]
            summary = "[system] 后台通知：\n" + "\n".join(lines)
            agent.history.append(yuullm.user(summary))
            chat.user(summary)

    return tool_calls
