"""Flow — observable, addressable, interruptible execution unit.

Everything that runs — LLM, tool, bash, sub-agent — is a Flow.
A Flow has three capabilities:
  stem    — typed, append-only event log (observe)
  mailbox — async queue (communicate)
  cancel  — task cancellation (interrupt)

Agent composes a Flow and adds LLM-specific behaviour (defer, messages).
No inheritance between Flow and Agent.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

import msgspec
import yuullm
import yuutrace as ytrace
import yuutools
from attrs import define, field

S = TypeVar("S")
M = TypeVar("M")
Ctx = TypeVar("Ctx")


def _usage_total_tokens(usage: yuullm.Usage) -> int:
    total = usage.total_tokens
    if total is not None:
        return total
    return (
        usage.input_tokens
        + usage.output_tokens
        + usage.cache_read_tokens
        + usage.cache_write_tokens
    )


def _merge_usage(total: yuullm.Usage | None, delta: yuullm.Usage) -> yuullm.Usage:
    if total is None:
        return delta
    return yuullm.Usage(
        provider=delta.provider or total.provider,
        model=delta.model or total.model,
        request_id=delta.request_id,
        input_tokens=total.input_tokens + delta.input_tokens,
        output_tokens=total.output_tokens + delta.output_tokens,
        cache_read_tokens=total.cache_read_tokens + delta.cache_read_tokens,
        cache_write_tokens=total.cache_write_tokens + delta.cache_write_tokens,
        total_tokens=_usage_total_tokens(total) + _usage_total_tokens(delta),
    )


# ---------------------------------------------------------------------------
# Agent stem events — what an Agent's flow.stem contains
# ---------------------------------------------------------------------------

class UserMessage(msgspec.Struct, frozen=True):
    content: str


class ToolResult(msgspec.Struct, frozen=True):
    call_id: str
    name: str
    output: str


AgentEvent = (
    UserMessage
    | yuullm.Reasoning
    | yuullm.Response
    | yuullm.ToolCall
    | ToolResult
)


def render_agent_event(event: AgentEvent) -> str:
    """Render an agent stem event to human-readable text."""
    match event:
        case UserMessage(content=text):
            return f"[user] {text}"
        case yuullm.Reasoning(item=text):
            return f"[thinking] {text}"
        case yuullm.Response(item=text):
            return str(text)
        case yuullm.ToolCall() as tc:
            return f"[call {tc.name}({tc.arguments or ''})]"
        case ToolResult(name=name, output=out):
            return f"[result {name}] {out}"
        case _:
            return str(event)


# ---------------------------------------------------------------------------
# FlowTree — serializable snapshot
# ---------------------------------------------------------------------------


class FlowTree(msgspec.Struct, frozen=True):
    flow_id: str
    kind: str
    info: dict[str, Any]
    stem: tuple[Any, ...]
    children: tuple["FlowTree", ...]


# ---------------------------------------------------------------------------
# Flow — the universal execution container
# ---------------------------------------------------------------------------


@define
class Flow(Generic[S, M]):
    """Observable, addressable, interruptible execution unit."""

    id: str = field(factory=lambda: uuid4().hex)
    kind: str = "flow"
    info: dict[str, Any] = field(factory=dict)
    stem: list[S] = field(factory=list)
    mailbox: asyncio.Queue[M] = field(factory=asyncio.Queue)
    children: list[Flow[Any, Any]] = field(factory=list, init=False)
    _task: asyncio.Task[Any] | None = field(default=None, init=False)
    _defer_requested: asyncio.Event = field(factory=asyncio.Event, init=False)
    _defer_partial: str | None = field(default=None, init=False)

    def emit(self, item: S) -> None:
        self.stem.append(item)

    def send(self, msg: M) -> None:
        self.mailbox.put_nowait(msg)

    def start(self, coro: Coroutine[Any, Any, Any]) -> None:
        self._task = asyncio.create_task(coro, name=f"{self.kind}-{self.id}")

    def spawn(self, kind: str) -> Flow[Any, Any]:
        child: Flow[Any, Any] = Flow(kind=kind)
        self.children.append(child)
        return child

    def find(self, flow_id: str) -> Flow[Any, Any] | None:
        if self.id == flow_id:
            return self
        for child in self.children:
            hit = child.find(flow_id)
            if hit is not None:
                return hit
        return None

    def request_defer(self, partial: str = "") -> None:
        """Request the parent agent to defer this flow to background.

        *partial* is included in the tool result the LLM sees.
        The flow's task keeps running; the agent will pick up the
        final result via ``_bg_finish``.
        """
        self._defer_partial = partial or None
        self._defer_requested.set()

    def cancel(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        for child in self.children:
            child.cancel()

    async def wait(self) -> None:
        if self._task is not None:
            await self._task

    def tail(self, limit: int = 200) -> tuple[S, ...]:
        if limit <= 0:
            return ()
        return tuple(self.stem[-limit:])

    def render(self, fn: Callable[[S], str], limit: int = 200) -> str:
        """Render recent stem events to text using *fn*."""
        return "\n".join(fn(e) for e in self.tail(limit))

    def inspect(self) -> FlowTree:
        return FlowTree(
            self.id,
            self.kind,
            dict(self.info),
            tuple(self.stem),
            tuple(child.inspect() for child in self.children),
        )


# ---------------------------------------------------------------------------
# Agent — LLM execution flow with config, defer, mailbox proxy
# ---------------------------------------------------------------------------


@define
class Agent(Generic[Ctx]):
    """An LLM agent: observable, interruptible, addressable."""

    client: yuullm.YLLMClient
    manager: yuutools.ToolManager[Ctx]
    ctx: Ctx
    system: str = ""
    model: str | None = None
    agent_name: str = "agent"
    conversation_id: UUID | None = None
    initial_messages: list[yuullm.Message] = field(factory=list)

    flow: Flow[AgentEvent, yuullm.Message] = field(init=False)
    messages: list[yuullm.Message] = field(factory=list, init=False)
    total_tokens: int = field(default=0, init=False)
    last_usage: yuullm.Usage | None = field(default=None, init=False)
    total_usage: yuullm.Usage | None = field(default=None, init=False)
    last_cost_usd: float = field(default=0.0, init=False)
    total_cost_usd: float = field(default=0.0, init=False)
    _defer: asyncio.Event = field(factory=asyncio.Event, init=False)
    _chat: ytrace.ConversationContext | None = field(default=None, init=False)

    def start(self) -> None:
        self.flow = Flow(kind="agent")
        self.flow.start(self._run())

    def send(self, msg: yuullm.Message | str, *, defer_tools: bool = False) -> None:
        if isinstance(msg, str):
            msg = yuullm.user(msg)
        self.flow.send(msg)
        if defer_tools:
            self._defer.set()

    def inspect(self) -> FlowTree:
        return self.flow.inspect()

    def render(self, limit: int = 200) -> str:
        return self.flow.render(render_agent_event, limit)

    async def wait(self) -> None:
        await self.flow.wait()

    # -- LLM loop --

    def _conv_id(self) -> UUID:
        return self.conversation_id or UUID(self.flow.id)

    @property
    def conversation_id_value(self) -> UUID:
        return self._conv_id()

    async def _run(self) -> None:
        if self.initial_messages:
            self.messages.extend(self.initial_messages)
        elif self.system:
            self.messages.append(yuullm.system(self.system))

        first: yuullm.Message = await self.flow.mailbox.get()
        self.flow.emit(UserMessage(_msg_text(first)))
        self.messages.append(first)

        if ytrace.is_initialized():
            with ytrace.conversation(
                id=self._conv_id(),
                agent=self.agent_name,
                model=self.model or "",
            ) as chat:
                self._chat = chat
                if self.system:
                    chat.system(self.system, tools=self.manager.specs() or None)
                chat.user(_msg_text(first))
                try:
                    await self._loop()
                finally:
                    self._chat = None
        else:
            await self._loop()

    async def _loop(self) -> None:
        while True:
            tool_calls = await self._call_llm()
            if not tool_calls:
                return
            await self._run_tools(tool_calls)
            self._drain_mailbox()

    async def _call_llm(self) -> list[yuullm.ToolCall]:
        return await self._stream_llm(self._chat)

    async def _stream_llm(
        self, chat: ytrace.ConversationContext | None,
    ) -> list[yuullm.ToolCall]:
        if chat is None:
            return await self._stream_llm_step(None)
        with chat.llm_gen() as gen:
            return await self._stream_llm_step(gen)

    async def _stream_llm_step(
        self,
        gen: ytrace.LlmGenContext | None,
    ) -> list[yuullm.ToolCall]:
        stream, store = await self.client.stream(
            self.messages, model=self.model, tools=self.manager.specs() or None,
        )
        assistant_items: list[yuullm.Item] = []
        reasoning_items: list[str] = []
        tool_calls: list[yuullm.ToolCall] = []

        async for item in stream:
            if isinstance(item, yuullm.Tick):
                continue
            self.flow.emit(item)
            match item:
                case yuullm.Reasoning(item=reasoning):
                    reasoning_items.append(reasoning)
                case yuullm.Response(item=resp):
                    assistant_items.append(resp)
                case yuullm.ToolCall() as tc:
                    tool_calls.append(tc)
                    assistant_items.append({
                        "type": "tool_call",
                        "id": tc.id,
                        "name": tc.name,
                        "arguments": tc.arguments,
                    })

        # Accumulate token counts
        usage = store.get("usage")
        if usage is not None:
            self.last_usage = usage
            self.total_usage = _merge_usage(self.total_usage, usage)
            self.total_tokens = _usage_total_tokens(self.total_usage)
        cost = store.get("cost")
        if cost is not None:
            self.last_cost_usd = float(getattr(cost, "total_cost", 0.0) or 0.0)
            self.total_cost_usd += self.last_cost_usd
        else:
            self.last_cost_usd = 0.0

        # When gen is active via chat.llm_gen(), usage/cost events attach to the
        # llm_gen span instead of the parent conversation span.
        if gen is not None:
            gen.log(_trace_items_for_log(reasoning_items, assistant_items))
            if usage is not None:
                if cost is not None:
                    ytrace.record_llm_cost(usage, cost)
                else:
                    ytrace.record_llm_usage(usage)

        if assistant_items:
            normalized_items = _normalize_assistant_items(
                assistant_items, group_tool_calls=False,
            )
            self.messages.append(yuullm.assistant(*normalized_items))
        return tool_calls

    async def _exec_tool(
        self,
        tc: yuullm.ToolCall,
        child: Flow[Any, Any],
        span: ytrace.ToolSpan | None,
    ) -> str:
        kwargs = json.loads(tc.arguments) if tc.arguments else {}
        ctx = self.ctx
        evolve = getattr(ctx, "evolve", None)
        if callable(evolve):
            ctx = evolve(current_run_id=child.id, current_flow=child)
        bound = self.manager[tc.name].bind(ctx)
        try:
            result = await bound.run(**kwargs)
        except Exception as exc:
            if span is not None:
                span.fail(f"{type(exc).__name__}: {exc}")
                span.end()
            raise
        child.emit(result)
        output = str(result)
        if span is not None:
            span.ok(output)
            span.end()
        return output

    async def _run_tools(self, tool_calls: list[yuullm.ToolCall]) -> None:
        chat = self._chat

        # Open a tools span for the batch if tracing
        tools_ctx: ytrace.ToolsContext | None = None
        if chat is not None:
            tools_ctx = chat.start_tools()

        tasks: dict[str, tuple[yuullm.ToolCall, Flow[Any, Any], asyncio.Task[str], ytrace.ToolSpan | None]] = {}
        for tc in tool_calls:
            child = self.flow.spawn("tool")
            child.info["tool_name"] = tc.name
            kwargs = json.loads(tc.arguments) if tc.arguments else {}
            span: ytrace.ToolSpan | None = None
            if tools_ctx is not None:
                span = tools_ctx.start_tool(name=tc.name, call_id=tc.id, input=kwargs)
            child.start(self._exec_tool(tc, child, span))
            task = child._task
            assert task is not None
            tasks[tc.id] = (tc, child, task, span)

        # wait all OR defer signal (external or per-tool self-defer)
        pending = {t for _, _, t, _ in tasks.values()}
        defer_task = asyncio.create_task(self._defer.wait())

        # Track per-tool self-defer requests.
        child_defer_tasks: dict[asyncio.Task[None], str] = {}
        for call_id, (tc, child, task, span) in tasks.items():
            cdt = asyncio.create_task(child._defer_requested.wait())
            child_defer_tasks[cdt] = call_id

        self_deferred: set[str] = set()  # call_ids that self-deferred

        while pending:
            signals = {defer_task} | set(child_defer_tasks)
            done, pending = await asyncio.wait(
                pending | signals, return_when=asyncio.FIRST_COMPLETED,
            )
            pending -= signals

            # Handle per-tool self-defer: remove the tool from pending.
            for cdt in done & set(child_defer_tasks):
                cid = child_defer_tasks.pop(cdt)
                self_deferred.add(cid)
                _, _, t, _ = tasks[cid]
                pending.discard(t)

            if defer_task in done:
                break

        if not defer_task.done():
            defer_task.cancel()
        self._defer.clear()
        for cdt in child_defer_tasks:
            cdt.cancel()

        # collect results
        for call_id, (tc, child, task, span) in tasks.items():
            if task.done() and call_id not in self_deferred:
                output = task.result()
                self.flow.emit(ToolResult(call_id=call_id, name=tc.name, output=output))
                self.messages.append(yuullm.tool(call_id, output))
            else:
                run_id = child.id
                # Include partial output from self-defer if available.
                partial = child._defer_partial
                if partial:
                    content = f"Moved to background. id:{run_id}\n{partial}"
                else:
                    content = f"Moved to background. id:{run_id}"
                self.flow.emit(ToolResult(call_id, tc.name, content))
                self.messages.append(
                    yuullm.tool(call_id, content)
                )
                # Pass span to bg_finish so it can end the span when done
                asyncio.create_task(self._bg_finish(run_id, task, span))

        # End tools batch span (background tool spans outlive it — fine per OTEL)
        if tools_ctx is not None:
            tools_ctx.end()

    async def _bg_finish(
        self,
        run_id: str,
        task: asyncio.Task[str],
        span: ytrace.ToolSpan | None,
    ) -> None:
        try:
            result = await task
        except asyncio.CancelledError:
            result = "cancelled"
            if span is not None:
                span.fail(result)
                span.end()
        except Exception as exc:
            result = f"{type(exc).__name__}: {exc}"
            if span is not None:
                span.fail(result)
                span.end()
        else:
            # span.ok/end already called by _exec_tool on success
            pass
        text = f"[bg:{run_id}] {result}"
        if self._chat is not None:
            self._chat.user(text)
        self.flow.send(yuullm.user(text))

    def _drain_mailbox(self) -> None:
        while not self.flow.mailbox.empty():
            msg: yuullm.Message = self.flow.mailbox.get_nowait()
            text = _msg_text(msg)
            self.flow.emit(UserMessage(text))
            self.messages.append(msg)
            if self._chat is not None:
                self._chat.user(text)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_item(item: Any) -> Any:
    """Convert an assistant item to a trace-friendly form."""
    if isinstance(item, str):
        return {"type": "text", "text": item}
    if isinstance(item, yuullm.Reasoning):
        return {"type": "reasoning", "text": item.item}
    if isinstance(item, dict):
        return item
    return {"type": "text", "text": str(item)}


def _normalize_assistant_items(items: list[Any], *, group_tool_calls: bool = True) -> list[Any]:
    """Collapse streamed chunks and restore stable trace item shapes."""
    normalized: list[Any] = []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def _flush_text() -> None:
        if text_parts:
            normalized.append("".join(text_parts))
            text_parts.clear()

    def _flush_reasoning() -> None:
        if reasoning_parts:
            normalized.append(yuullm.Reasoning(item="".join(reasoning_parts)))
            reasoning_parts.clear()

    def _flush_tool_calls() -> None:
        if pending_tool_calls:
            if group_tool_calls:
                normalized.append({
                    "type": "tool_calls",
                    "tool_calls": list(pending_tool_calls),
                })
            else:
                normalized.extend(
                    {
                        "type": "tool_call",
                        "id": item["id"],
                        "name": item["function"],
                        "arguments": item["arguments"],
                    }
                    for item in pending_tool_calls
                )
            pending_tool_calls.clear()

    for item in items:
        if isinstance(item, str):
            _flush_reasoning()
            _flush_tool_calls()
            text_parts.append(item)
            continue
        if isinstance(item, yuullm.Reasoning):
            _flush_text()
            _flush_tool_calls()
            reasoning_parts.append(item.item)
            continue
        if isinstance(item, dict) and item.get("type") == "tool_call":
            _flush_text()
            _flush_reasoning()
            pending_tool_calls.append({
                "id": item.get("id", ""),
                "function": item.get("name", ""),
                "arguments": item.get("arguments", {}),
            })
            continue
        _flush_text()
        _flush_reasoning()
        _flush_tool_calls()
        normalized.append(item)

    _flush_text()
    _flush_reasoning()
    _flush_tool_calls()
    return normalized


def _trace_items_for_log(reasoning_items: list[str], assistant_items: list[Any]) -> list[Any]:
    """Convert runtime assistant items to stable trace payload items."""
    items: list[Any] = []
    if reasoning_items:
        items.append(yuullm.Reasoning(item="".join(reasoning_items)))
    items.extend(assistant_items)
    return [_trace_item(item) for item in _normalize_assistant_items(items)]


def _msg_text(msg: yuullm.Message) -> str:
    """Extract display text from a yuullm.Message."""
    _, content = msg
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c if isinstance(c, str) else json.dumps(c, ensure_ascii=False) for c in content]
        return "".join(parts)
    return str(content)
