"""Step loop — LLM → tool calls → trace."""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

import yuullm
import yuutrace as ytrace

from yuuagents.agent import Agent
from yuuagents.context import AgentContext
from yuuagents.persistence import TaskRecorder, ToolCallDTO, ToolErrorDTO, ToolResultDTO
from yuuagents.types import AgentStatus


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

    ytrace.init(service_name="yuuagents")
    with ytrace.conversation(
        id=UUID(agent.task_id),
        agent=agent.agent_id,
        model=agent.llm.default_model,
    ) as chat:
        chat.system(persona=agent.full_system_prompt, tools=agent.tools.specs())
        chat.user(task if not resume else agent.task)

        while not agent.done():
            try:
                await _step(agent, chat, ctx, recorder=recorder)
            except Exception as exc:
                agent.fail(exc)
                raise


async def _step(
    agent: Agent,
    chat: ytrace.ConversationContext,
    ctx: AgentContext,
    *,
    recorder: TaskRecorder | None = None,
) -> None:
    """Execute one LLM call + tool round."""
    # 1. Call LLM
    with chat.llm_gen() as gen:
        stream, store = await agent.llm.stream(
            agent.history,
            tools=agent.tools.specs(),
        )

        items: list[Any] = []
        async for item in stream:
            items.append(item)
        gen.log(items)

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
        return

    # 4. Execute tools
    with chat.tools() as tools_ctx:
        calls = []
        for tc in tool_calls:
            tool_obj = agent.tools[tc.name]
            bound = tool_obj.bind(ctx)
            params = json.loads(tc.arguments) if tc.arguments else {}
            calls.append(
                {
                    "tool_call_id": tc.id,
                    "tool": bound.run,
                    "params": params,
                }
            )
        results = await tools_ctx.gather(calls)

    # 5. Append tool results to history
    for r in results:
        content = str(r.error) if r.error else str(r.output)
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
