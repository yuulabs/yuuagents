"""Step loop — LLM → tool calls → trace."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import UUID

import yuullm
import yuutrace as ytrace

from yuuagents.agent import Agent
from yuuagents.context import AgentContext
from yuuagents.persistence import TaskRecorder, ToolCallDTO, ToolErrorDTO, ToolResultDTO
from yuuagents.running_tools import OutputBuffer, RunningToolRegistry
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


async def run(
    agent: Agent,
    task: str,
    ctx: AgentContext,
    *,
    recorder: TaskRecorder | None = None,
    resume: bool = False,
) -> None:
    """Run the agent loop until completion or error."""
    if not resume:
        agent.setup(task)

    # Set up running tools registry
    registry = RunningToolRegistry()
    ctx.running_tools = registry

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

            # Inject finished background tools as synthetic history
            finished = registry.collect_finished()
            for entry in finished:
                try:
                    result = entry.task.result()
                    content = str(result.output) if result.error is None else str(result.error)
                except Exception as exc:
                    content = f"[ERROR] {type(exc).__name__}: {exc}"
                # Synthetic assistant tool_call + tool_result pair
                agent.history.append(yuullm.assistant(
                    {"type": "tool_call", "id": entry.tool_call_id,
                     "name": "check_running_tool",
                     "arguments": json.dumps({"handle": entry.handle})}
                ))
                agent.history.append(yuullm.tool(
                    entry.tool_call_id,
                    f"Tool completed. Output: {content}",
                ))

            # Silence detection ping
            if silence_interval_first > 0:
                elapsed_silent = time.monotonic() - last_user_msg_time
                threshold = (
                    silence_interval_first
                    if elapsed_silent < silence_interval_subsequent
                    else silence_interval_subsequent
                )
                if elapsed_silent > threshold:
                    total_elapsed = time.monotonic() - agent.created_at.timestamp()
                    agent.history.append(yuullm.user(
                        f"[system] 你已经工作了 {elapsed_silent:.0f}s 没有给用户发送任何消息。"
                        "请通过 im send 告知用户当前进度和预计剩余时间。"
                    ))
                    last_user_msg_time = time.monotonic()

            try:
                tool_calls = await _step(
                    agent, chat, ctx, recorder=recorder,
                )
                # Track im send for silence detection
                if tool_calls and _has_im_send([], tool_calls):
                    last_user_msg_time = time.monotonic()
            except BaseException as exc:
                agent.fail(exc)
                raise


async def _step(
    agent: Agent,
    chat: ytrace.ConversationContext,
    ctx: AgentContext,
    *,
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

    # 3. If no tool calls, we're done
    if not tool_calls:
        agent.status = AgentStatus.DONE
        agent.state.pending_input_prompt = ""
        return tool_calls

    # 4. Execute tools
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
        results = await tools_ctx.gather(
            calls,
            soft_timeout=agent.soft_timeout,
            registry=ctx.running_tools,
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

    return tool_calls
