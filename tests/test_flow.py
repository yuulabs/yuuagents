"""End-to-end tests for core.flow — Flow and Agent."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator

import pytest
import yuullm
import yuutrace as ytrace
import yuutools as yt
from yuuagents.core.flow import (
    Agent,
    Flow,
    ToolResult,
    UserMessage,
    _normalize_assistant_items,
    _trace_items_for_log,
    _trace_item,
    render_agent_event,
)


# ---------------------------------------------------------------------------
# Helpers: fake LLM provider
# ---------------------------------------------------------------------------


class FakeProvider:
    """A scripted LLM provider that yields pre-defined responses."""

    def __init__(
        self,
        script: list[list[yuullm.StreamItem]],
        stores: list[dict] | None = None,
    ) -> None:
        # Each call to stream() pops the next entry from the script.
        self._script = list(script)
        self._stores = list(stores or [{} for _ in script])
        self._call_index = 0

    @property
    def provider(self) -> str:
        return "fake"

    @property
    def api_type(self) -> str:
        return "fake"

    async def stream(
        self,
        messages: list[yuullm.Message],
        *,
        model: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> yuullm.StreamResult:
        items = self._script[self._call_index]
        store = dict(self._stores[self._call_index])
        self._call_index += 1

        async def _gen() -> AsyncIterator[yuullm.StreamItem]:
            for item in items:
                yield item

        return _gen(), store


def make_client(
    script: list[list[yuullm.StreamItem]],
    *,
    stores: list[dict] | None = None,
) -> yuullm.YLLMClient:
    """Build a YLLMClient backed by a FakeProvider."""
    provider = FakeProvider(script, stores=stores)
    return yuullm.YLLMClient(provider=provider, default_model="fake-model")


# ---------------------------------------------------------------------------
# Tests: Flow basics
# ---------------------------------------------------------------------------


def test_flow_emit_and_tail():
    flow: Flow[str, str] = Flow(kind="test")
    flow.emit("a")
    flow.emit("b")
    flow.emit("c")
    assert flow.tail() == ("a", "b", "c")
    assert flow.tail(2) == ("b", "c")
    assert flow.tail(0) == ()


def test_flow_send_and_mailbox():
    flow: Flow[str, str] = Flow()
    flow.send("hello")
    assert not flow.mailbox.empty()
    msg = flow.mailbox.get_nowait()
    assert msg == "hello"


def test_flow_spawn_and_find():
    parent: Flow[str, str] = Flow(kind="parent")
    child = parent.spawn("child")
    assert child in parent.children
    assert parent.find(child.id) is child
    assert parent.find("nonexistent") is None
    assert parent.find(parent.id) is parent


def test_flow_inspect():
    parent: Flow[str, str] = Flow(kind="root")
    parent.emit("x")
    child = parent.spawn("leaf")
    child.emit("y")
    tree = parent.inspect()
    assert tree.kind == "root"
    assert tree.stem == ("x",)
    assert len(tree.children) == 1
    assert tree.children[0].stem == ("y",)


def test_flow_render():
    flow: Flow[str, str] = Flow()
    flow.emit("line1")
    flow.emit("line2")
    assert flow.render(str) == "line1\nline2"


# ---------------------------------------------------------------------------
# Tests: Agent — text-only response (no tool calls)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_text_response():
    """Agent receives a task, LLM responds with text only, agent finishes."""
    script = [
        [yuullm.Response(item="Hello from the LLM!")],
    ]
    client = make_client(script)
    manager: yt.ToolManager = yt.ToolManager()

    agent: Agent[None] = Agent(
        client=client, manager=manager, ctx=None,
        system="You are a test agent.", model="fake-model",
    )
    agent.start()
    agent.send("Hi there")
    await agent.wait()

    # Check messages: system + user + assistant
    assert len(agent.messages) == 3
    assert agent.messages[0] == yuullm.system("You are a test agent.")
    assert agent.messages[1] == yuullm.user("Hi there")
    # assistant message contains the response text
    role, content = agent.messages[2]
    assert role == "assistant"
    assert "Hello from the LLM!" in str(content)

    # Check stem events
    events = agent.flow.stem
    assert any(isinstance(e, UserMessage) and e.content == "Hi there" for e in events)
    assert any(isinstance(e, yuullm.Response) for e in events)


@pytest.mark.asyncio
async def test_agent_merges_streamed_text_chunks_before_persisting_history():
    client = make_client(
        [[yuullm.Response(item="现在 "), yuullm.Response(item="我需要"), yuullm.Response(item="回复用户。")]]
    )
    manager: yt.ToolManager = yt.ToolManager()

    agent: Agent[None] = Agent(
        client=client, manager=manager, ctx=None,
        model="fake-model",
    )
    agent.start()
    agent.send("hi")
    await agent.wait()

    assert agent.messages[-1] == ("assistant", ["现在 我需要回复用户。"])


@pytest.mark.asyncio
async def test_session_exposes_last_and_total_usage_and_cost():
    """Session should sync structured last/total usage + cost to host-facing state."""
    from yuuagents.agent import AgentConfig
    from yuuagents.context import AgentContext
    from yuuagents.runtime_session import Session

    usage = yuullm.Usage(
        provider="fake",
        model="fake-model",
        input_tokens=11,
        output_tokens=7,
        total_tokens=18,
    )
    cost = yuullm.Cost(
        input_cost=0.0011,
        output_cost=0.0007,
        total_cost=0.0018,
    )
    client = make_client(
        [[yuullm.Response(item="done")]],
        stores=[{"usage": usage, "cost": cost}],
    )
    manager: yt.ToolManager = yt.ToolManager()
    session = Session(
        config=AgentConfig(
            agent_id="host-facing",
            tools=manager,
            llm=client,
            system="",
        ),
        context=AgentContext(
            task_id="task-1",
            agent_id="host-facing",
            workdir="",
            docker_container="",
        ),
    )

    session.start("hello")
    await session.wait()

    assert session.last_usage == usage
    assert session.total_usage == usage
    assert session.last_cost_usd == pytest.approx(cost.total_cost)
    assert session.total_cost_usd == pytest.approx(cost.total_cost)
    assert session.total_tokens == 18


@pytest.mark.asyncio
async def test_llm_usage_and_cost_are_recorded_on_llm_gen_span():
    usage = yuullm.Usage(
        provider="fake",
        model="fake-model",
        request_id="req-1",
        input_tokens=11,
        output_tokens=7,
        cache_read_tokens=5,
        total_tokens=23,
    )
    cost = yuullm.Cost(
        input_cost=0.0011,
        output_cost=0.0007,
        total_cost=0.0018,
        source="fake-prices",
    )
    client = make_client(
        [[yuullm.Response(item="done")]],
        stores=[{"usage": usage, "cost": cost}],
    )
    manager: yt.ToolManager = yt.ToolManager()
    agent: Agent[None] = Agent(
        client=client,
        manager=manager,
        ctx=None,
        system="You are a traced test agent.",
        model="fake-model",
        agent_name="trace-test",
        conversation_id=uuid.uuid4(),
    )

    store = ytrace.init_memory()
    agent.start()
    agent.send("hello")
    await agent.wait()

    conv = store.get_conversation(str(agent.conversation_id_value))
    assert conv is not None
    spans = conv["spans"]

    llm_span = next(span for span in spans if span["name"] == "llm_gen")
    conversation_span = next(span for span in spans if span["name"] == "conversation")

    llm_event_names = [event["name"] for event in llm_span["events"]]
    conversation_event_names = [event["name"] for event in conversation_span["events"]]

    assert "yuu.llm.usage" in llm_event_names
    assert "yuu.cost" in llm_event_names
    assert "yuu.llm.usage" not in conversation_event_names
    assert "yuu.cost" not in conversation_event_names


@pytest.mark.asyncio
async def test_session_resume_preserves_conversation_id():
    from uuid import UUID

    from yuuagents.agent import AgentConfig
    from yuuagents.context import AgentContext
    from yuuagents.runtime_session import Session

    client = make_client(
        [
            [yuullm.Response(item="first")],
            [yuullm.Response(item="second")],
        ]
    )
    manager: yt.ToolManager = yt.ToolManager()
    session = Session(
        config=AgentConfig(
            agent_id="host-facing",
            tools=manager,
            llm=client,
            system="system prompt",
        ),
        context=AgentContext(
            task_id="task-1",
            agent_id="host-facing",
            workdir="",
            docker_container="",
        ),
    )

    session.start("hello")
    await session.wait()
    first_conversation_id = session.conversation_id

    resumed = Session(config=session.config, context=session.context)
    resumed.resume(
        "followup",
        history=session.history,
        conversation_id=first_conversation_id,
    )
    await resumed.wait()

    assert isinstance(first_conversation_id, UUID)
    assert resumed.conversation_id == first_conversation_id


@pytest.mark.asyncio
async def test_resume_trace_records_system_and_tools():
    from yuuagents.agent import AgentConfig
    from yuuagents.context import AgentContext
    from yuuagents.runtime_session import Session

    @yt.tool()
    async def ping() -> str:
        """Return a fixed value."""
        return "pong"

    store = ytrace.init_memory()
    manager: yt.ToolManager = yt.ToolManager()
    manager.register(ping)
    client = make_client(
        [
            [yuullm.Response(item="first")],
            [yuullm.Response(item="second")],
        ]
    )
    config = AgentConfig(
        agent_id="host-facing",
        tools=manager,
        llm=client,
        system="system prompt",
    )
    context = AgentContext(
        task_id="task-1",
        agent_id="host-facing",
        workdir="",
        docker_container="",
    )

    first = Session(config=config, context=context)
    first.start("hello")
    await first.wait()

    resumed = Session(config=config, context=context)
    resumed.resume(
        "followup",
        history=first.history,
        conversation_id=first.conversation_id,
    )
    await resumed.wait()

    spans = store.get_all_spans()
    conv_spans = [
        span for span in spans
        if span["name"] == "conversation"
        and span["conversation_id"] == str(first.conversation_id)
    ]
    assert conv_spans
    attrs = conv_spans[-1]["attributes"]
    assert attrs["yuu.context.system.persona"] == "system prompt"
    assert "ping" in attrs["yuu.context.system.tools"]


# ---------------------------------------------------------------------------
# Tests: Agent — with tool calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_tool_call():
    """Agent calls a tool, gets result, then LLM responds with text."""

    @yt.tool()
    async def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    manager: yt.ToolManager = yt.ToolManager()
    manager.register(add)

    # Script: first LLM call returns a tool call, second returns text
    script = [
        [yuullm.ToolCall(id="tc1", name="add", arguments='{"a": 2, "b": 3}')],
        [yuullm.Response(item="The sum is 5.")],
    ]
    client = make_client(script)

    agent: Agent[None] = Agent(
        client=client, manager=manager, ctx=None,
        model="fake-model",
    )
    agent.start()
    agent.send("What is 2 + 3?")
    await agent.wait()

    # Should have: user + assistant(tool_call) + tool_result + assistant(text)
    assert len(agent.messages) == 4

    # Check tool result in stem
    tool_results = [e for e in agent.flow.stem if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].output == "5"
    assert tool_results[0].name == "add"

    # Final response
    assert any(
        isinstance(e, yuullm.Response) and "The sum is 5" in str(e.item)
        for e in agent.flow.stem
    )


# ---------------------------------------------------------------------------
# Tests: Agent — send mid-run (mailbox drain)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_mailbox_drain():
    """Messages sent while tools run are drained into history."""

    @yt.tool()
    async def slow_tool() -> str:
        """A tool that takes a moment."""
        await asyncio.sleep(0.05)
        return "done"

    manager: yt.ToolManager = yt.ToolManager()
    manager.register(slow_tool)

    script = [
        [yuullm.ToolCall(id="tc1", name="slow_tool", arguments="{}")],
        [yuullm.Response(item="All done.")],
    ]
    client = make_client(script)

    agent: Agent[None] = Agent(
        client=client, manager=manager, ctx=None, model="fake-model",
    )
    agent.start()
    agent.send("Go")

    await asyncio.sleep(0.02)  # let tool start
    agent.send("ping")  # inject while tool runs
    await agent.wait()

    # The "ping" should appear in messages as a user message
    user_msgs = [m for m in agent.messages if m[0] == "user"]
    texts = [m[1] if isinstance(m[1], str) else str(m[1]) for m in user_msgs]
    assert any("ping" in t for t in texts)


@pytest.mark.asyncio
async def test_background_tool_flow_waits_for_completion():
    released = asyncio.Event()

    @yt.tool()
    async def hold() -> str:
        await released.wait()
        return "done"

    manager: yt.ToolManager = yt.ToolManager()
    manager.register(hold)
    client = make_client(
        [
            [yuullm.ToolCall(id="tc1", name="hold", arguments="{}")],
            [yuullm.Response(item="finished")],
        ]
    )
    agent: Agent[None] = Agent(
        client=client,
        manager=manager,
        ctx=None,
        model="fake-model",
    )
    agent.start()
    agent.send("Go")

    await asyncio.sleep(0.02)
    agent.send("background it", defer_tools=True)
    await asyncio.sleep(0.02)

    tool_flow = next(child for child in agent.flow.children if child.info.get("tool_name") == "hold")
    waiter = asyncio.create_task(tool_flow.wait())
    await asyncio.sleep(0.02)
    assert not waiter.done()

    released.set()
    await waiter
    await agent.wait()


# ---------------------------------------------------------------------------
# Tests: Agent — cancel
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_cancel():
    """Cancelling the flow stops the agent."""

    @yt.tool()
    async def hang() -> str:
        """Never returns."""
        await asyncio.sleep(999)
        return "unreachable"

    manager: yt.ToolManager = yt.ToolManager()
    manager.register(hang)

    script = [
        [yuullm.ToolCall(id="tc1", name="hang", arguments="{}")],
    ]
    client = make_client(script)

    agent: Agent[None] = Agent(
        client=client, manager=manager, ctx=None, model="fake-model",
    )
    agent.start()
    agent.send("Start")

    await asyncio.sleep(0.05)
    agent.flow.cancel()
    try:
        await agent.wait()
    except asyncio.CancelledError:
        pass
    # If we got here without hanging, cancel works.


# ---------------------------------------------------------------------------
# Tests: render_agent_event
# ---------------------------------------------------------------------------


def test_render_agent_event():
    assert "[user]" in render_agent_event(UserMessage("hi"))
    assert "[call add" in render_agent_event(
        yuullm.ToolCall(id="x", name="add", arguments='{"a":1}')
    )
    assert "[result add]" in render_agent_event(
        ToolResult(call_id="x", name="add", output="2")
    )


def test_normalize_assistant_items_merges_text_and_groups_tool_calls():
    items = [
        "消",
        "息",
        {
            "type": "tool_call",
            "id": "tc1",
            "name": "call_cap_cli",
            "arguments": {"command": "im send --ctx 1 -- [...]"},
        },
        {
            "type": "tool_call",
            "id": "tc2",
            "name": "read_cap_doc",
            "arguments": {"name": "im"},
        },
        "已",
        "发送",
        "！",
    ]

    normalized = _normalize_assistant_items(items)

    assert normalized == [
        "消息",
        {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": "call_cap_cli",
                    "arguments": {"command": "im send --ctx 1 -- [...]"},
                },
                {
                    "id": "tc2",
                    "function": "read_cap_doc",
                    "arguments": {"name": "im"},
                },
            ],
        },
        "已发送！",
    ]
    assert [_trace_item(item) for item in normalized] == [
        {"type": "text", "text": "消息"},
        normalized[1],
        {"type": "text", "text": "已发送！"},
    ]


def test_trace_items_for_log_keeps_runtime_tool_call_shape_out_of_messages():
    items = [
        {
            "type": "tool_call",
            "id": "tc1",
            "name": "call_cap_cli",
            "arguments": {"command": "im send --ctx 1 -- [...]"},
        }
    ]

    traced = _trace_items_for_log([], items)

    assert traced == [
        {
            "type": "tool_calls",
            "tool_calls": [
                {
                    "id": "tc1",
                    "function": "call_cap_cli",
                    "arguments": {"command": "im send --ctx 1 -- [...]"},
                }
            ],
        }
    ]
    assert items[0]["type"] == "tool_call"


def test_trace_items_for_log_merges_reasoning_chunks():
    traced = _trace_items_for_log(
        ["现在", " 我", " 需要", " 回复"],
        ["已", "发送"],
    )

    assert traced == [
        {"type": "reasoning", "text": "现在 我 需要 回复"},
        {"type": "text", "text": "已发送"},
    ]
