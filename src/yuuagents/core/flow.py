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
from collections.abc import AsyncGenerator, Callable, Coroutine
from typing import Any, Generic, TypeVar
from uuid import UUID, uuid4

import msgspec
import yuullm
import yuutrace as ytrace
from attrs import define, field

from yuuagents.agent import AgentConfig
from yuuagents.types import StepResult

S = TypeVar("S")
M = TypeVar("M")
Ctx = TypeVar("Ctx")


# ---------------------------------------------------------------------------
# Agent state snapshot — minimal restorable state
# ---------------------------------------------------------------------------


class AgentState(msgspec.Struct, frozen=True):
    """Minimal restorable state for an Agent.

    Every tool_call in messages is paired with a tool result — no dangling
    calls.  A new Agent can be constructed from this state with zero issues.
    """

    messages: tuple[yuullm.Message, ...]
    total_usage: yuullm.Usage | None
    total_cost_usd: float
    rounds: int
    conversation_id: str | None  # UUID hex or None


# ---------------------------------------------------------------------------
# Background task metadata
# ---------------------------------------------------------------------------


@define(frozen=True)
class _BgTaskInfo:
    """Structured metadata for a background (deferred) tool task."""

    call_id: str
    tool_name: str
    child: "Flow[Any, Any]"
    task: "asyncio.Task[None]"    # the _bg_finish wrapper task
    tool_task: "asyncio.Task[str]"  # the actual tool execution task


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
        case yuullm.Response(item=item):
            return item.get("text", "") if item.get("type") == "text" else str(item)
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

    config: AgentConfig
    ctx: Ctx
    conversation_id: UUID | None = None
    initial_messages: list[yuullm.Message] = field(factory=list)

    flow: Flow[AgentEvent, yuullm.Message] = field(init=False)
    messages: list[yuullm.Message] = field(factory=list, init=False)
    total_tokens: int = field(default=0, init=False)
    last_usage: yuullm.Usage | None = field(default=None, init=False)
    total_usage: yuullm.Usage | None = field(default=None, init=False)
    last_cost_usd: float = field(default=0.0, init=False)
    total_cost_usd: float = field(default=0.0, init=False)
    rounds: int = field(default=0, init=False)
    _defer: asyncio.Event = field(factory=asyncio.Event, init=False)
    _chat: ytrace.ConversationContext | None = field(default=None, init=False)
    _bg_tasks: dict[str, _BgTaskInfo] = field(factory=dict, init=False)

    def start(self) -> None:
        """Create the flow. Does NOT launch a background task."""
        self.flow = Flow(kind="agent")

    def send_first(self, task: str) -> None:
        """Queue the first user message. Must be called before steps()."""
        self.flow.send(yuullm.user(task))

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

    # -- state model --

    @property
    def has_pending_background(self) -> bool:
        """True if any background (deferred) tasks are still running."""
        return bool(self._bg_tasks)

    async def snapshot(self, *, as_interrupted: bool = False) -> AgentState:
        """Return a restorable state snapshot.

        ``as_interrupted=False`` (default): if background tasks are pending,
        **await** them first so that all tool results are present.

        ``as_interrupted=True``: return immediately; for each pending bg task
        synthesise an ``[interrupted]`` tool result in the returned copy of
        messages.  Does NOT mutate ``self.messages``.
        """
        if not as_interrupted:
            if self._bg_tasks:
                await asyncio.wait(
                    [info.task for info in self._bg_tasks.values()],
                )
                self._drain_mailbox()
            return AgentState(
                messages=tuple(self.messages),
                total_usage=self.total_usage,
                total_cost_usd=self.total_cost_usd,
                rounds=self.rounds,
                conversation_id=self._conv_id().hex if self.conversation_id or self.flow else None,
            )

        # as_interrupted: synthetic copy with interrupted results
        msgs = list(self.messages)
        for info in self._bg_tasks.values():
            tail_output = info.child.render(render_agent_event, limit=50)
            content = f"[interrupted] {tail_output}" if tail_output else "[interrupted]"
            msgs.append(yuullm.tool(info.call_id, content))
        return AgentState(
            messages=tuple(msgs),
            total_usage=self.total_usage,
            total_cost_usd=self.total_cost_usd,
            rounds=self.rounds,
            conversation_id=self._conv_id().hex if self.conversation_id or self.flow else None,
        )

    async def kill(self) -> None:
        """Cancel all background tasks and synthesise interrupted results.

        After kill(), ``self.messages`` is valid — ``snapshot()`` will return
        immediately.
        """
        if not self._bg_tasks:
            return

        # Snapshot before cancellation (gather may trigger _bg_finish which pops entries)
        infos = list(self._bg_tasks.values())

        # Cancel all tasks
        for info in infos:
            if not info.tool_task.done():
                info.tool_task.cancel()
            if not info.task.done():
                info.task.cancel()

        # Cancel child flows recursively
        for info in infos:
            info.child.cancel()

        # Wait for cancellation to propagate
        all_tasks = [info.task for info in infos]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        # Synthesise interrupted tool results into self.messages
        for info in infos:
            tail_output = info.child.render(render_agent_event, limit=50)
            content = f"[interrupted] {tail_output}" if tail_output else "[interrupted]"
            self.flow.emit(ToolResult(call_id=info.call_id, name=info.tool_name, output=content))
            self.messages.append(yuullm.tool(info.call_id, content))

        self._bg_tasks.clear()

    # -- LLM loop --

    def _conv_id(self) -> UUID:
        return self.conversation_id or UUID(self.flow.id)

    @property
    def conversation_id_value(self) -> UUID:
        return self._conv_id()

    async def steps(self) -> AsyncGenerator[StepResult, None]:
        """Host-driven async generator. Yields after each LLM round."""
        # Setup messages
        if self.initial_messages:
            self.messages.extend(self.initial_messages)
        elif self.config.system:
            self.messages.append(yuullm.system(self.config.system))

        first: yuullm.Message = await self.flow.mailbox.get()
        self.flow.emit(UserMessage(_msg_text(first)))
        self.messages.append(first)

        if ytrace.is_initialized():
            chat = ytrace.start_conversation(
                id=self._conv_id(),
                agent=self.config.agent_id,
                model=self.config.llm.default_model or "",
            )
            self._chat = chat
            if self.config.system:
                chat.system(self.config.system, tools=self.config.tools.specs() or None)
            _, first_items = first
            chat.user(*first_items)
            error: Exception | None = None
            try:
                async for step in self._step_loop():
                    yield step
            except Exception as exc:
                error = exc
                raise
            finally:
                self._chat = None
                chat.end(error)
        else:
            async for step in self._step_loop():
                yield step

    def _prune_children(self) -> None:
        """Remove completed children that are not tracked as bg tasks."""
        bg_child_ids = {info.child.id for info in self._bg_tasks.values()}
        self.flow.children = [
            child for child in self.flow.children
            if (child._task is not None and not child._task.done())
            or child.id in bg_child_ids
        ]

    async def _step_loop(self) -> AsyncGenerator[StepResult, None]:
        while True:
            self._prune_children()
            self.rounds += 1
            tool_calls = await self._call_llm()
            if not tool_calls:
                await self._wait_pending_bg()
                if self._drain_mailbox():
                    continue
                yield StepResult(done=True, tokens=self.total_tokens, rounds=self.rounds)
                return
            await self._run_tools(tool_calls)
            self._drain_mailbox()
            yield StepResult(done=False, tokens=self.total_tokens, rounds=self.rounds)

    async def _wait_pending_bg(self) -> None:
        """Wait for all background tasks to complete."""
        if not self._bg_tasks:
            return
        tasks = [info.task for info in self._bg_tasks.values()]
        await asyncio.wait(tasks)
        # Remove completed bg tasks
        done_ids = [
            call_id for call_id, info in self._bg_tasks.items()
            if info.task.done()
        ]
        for call_id in done_ids:
            del self._bg_tasks[call_id]

    async def _call_llm(self) -> list[yuullm.ToolCall]:
        return await self._stream_llm(self._chat)

    async def _stream_llm(
        self, chat: ytrace.ConversationContext | None,
    ) -> list[yuullm.ToolCall]:
        if chat is None:
            return await self._stream_llm_step(None)
        turn = chat.start_turn("assistant")
        try:
            result = await self._stream_llm_step(turn)
        except Exception as exc:
            turn.end(error=exc)
            raise
        else:
            turn.end()
            return result

    async def _stream_llm_step(
        self,
        turn: ytrace.TurnContext | None,
    ) -> list[yuullm.ToolCall]:
        stream, store = await self.config.llm.stream(
            self.messages, model=self.config.llm.default_model, tools=self.config.tools.specs() or None,
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

        # Record assistant turn items and usage/cost on the conversation span.
        if turn is not None:
            turn.add(*_trace_items_for_log(reasoning_items, assistant_items))
            if usage is not None:
                turn.usage(usage, cost=cost)

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
        bound = self.config.tools[tc.name].bind(ctx)
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

        # Optional batch-level timeout
        timeout_task: asyncio.Task[None] | None = None
        if self.config.tool_batch_timeout > 0:
            timeout_task = asyncio.create_task(asyncio.sleep(self.config.tool_batch_timeout))

        # Track per-tool self-defer requests.
        child_defer_tasks: dict[asyncio.Task[None], str] = {}
        for call_id, (tc, child, task, span) in tasks.items():
            cdt = asyncio.create_task(child._defer_requested.wait())
            child_defer_tasks[cdt] = call_id

        self_deferred: set[str] = set()  # call_ids that self-deferred

        while pending:
            signals = {defer_task} | set(child_defer_tasks)
            if timeout_task is not None:
                signals.add(timeout_task)
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

            if timeout_task is not None and timeout_task in done:
                # Defer all still-pending tool tasks to background
                for call_id, (tc, child, task, span) in tasks.items():
                    if task in pending:
                        self_deferred.add(call_id)
                break

        if not defer_task.done():
            defer_task.cancel()
        self._defer.clear()
        if timeout_task is not None and not timeout_task.done():
            timeout_task.cancel()
        for cdt in child_defer_tasks:
            cdt.cancel()

        # collect results
        for call_id, (tc, child, task, span) in tasks.items():
            if task.done() and call_id not in self_deferred:
                if task.cancelled():
                    content = "[cancelled] tool execution cancelled"
                    self.flow.emit(ToolResult(call_id=call_id, name=tc.name, output=content))
                    self.messages.append(yuullm.tool(call_id, content))
                    if span is not None:
                        span.fail(content)
                        span.end()
                else:
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
                bg = asyncio.create_task(self._bg_finish(call_id, run_id, task, span))
                self._bg_tasks[call_id] = _BgTaskInfo(
                    call_id=call_id,
                    tool_name=tc.name,
                    child=child,
                    task=bg,
                    tool_task=task,
                )

        # End tools batch span (background tool spans outlive it — fine per OTEL)
        if tools_ctx is not None:
            tools_ctx.end()

    async def _bg_finish(
        self,
        call_id: str,
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
        self._bg_tasks.pop(call_id, None)

    def _drain_mailbox(self) -> bool:
        """Drain pending messages from the mailbox. Returns True if any were drained."""
        drained = False
        while not self.flow.mailbox.empty():
            msg: yuullm.Message = self.flow.mailbox.get_nowait()
            text = _msg_text(msg)
            self.flow.emit(UserMessage(text))
            self.messages.append(msg)
            if self._chat is not None:
                _, msg_items = msg
                self._chat.user(*msg_items)
            drained = True
        return drained


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trace_item(item: Any) -> Any:
    """Convert an assistant item to a trace-friendly form."""
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
            normalized.append({"type": "text", "text": "".join(text_parts)})
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
                normalized.extend(pending_tool_calls)
            pending_tool_calls.clear()

    for item in items:
        if isinstance(item, yuullm.Reasoning):
            _flush_text()
            _flush_tool_calls()
            reasoning_parts.append(item.item)
            continue
        if isinstance(item, dict) and item.get("type") == "text":
            _flush_reasoning()
            _flush_tool_calls()
            text_parts.append(item.get("text", ""))
            continue
        if isinstance(item, dict) and item.get("type") == "tool_call":
            _flush_text()
            _flush_reasoning()
            pending_tool_calls.append({
                "type": "tool_call",
                "id": item.get("id", ""),
                "name": item.get("name", ""),
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
    _, items = msg
    return "".join(
        item["text"] if item.get("type") == "text" else json.dumps(item, ensure_ascii=False)
        for item in items
    )
