from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import yuullm
import yuutools as yt

from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.runtime_session import Session
from yuuagents.tools.control import (
    cancel_background,
    defer_background,
    input_background,
    inspect_background,
    wait_background,
)
from yuuagents.tools.delegate import delegate


class _FakeProvider:
    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)

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
        model: str,
        tools: list[dict[str, Any]] | None = None,
        on_raw_chunk: yuullm.RawChunkHook | None = None,
        **kwargs: Any,
    ) -> yuullm.StreamResult:
        del messages, model, tools, on_raw_chunk, kwargs
        reply = self._replies.pop(0)

        async def _iter() -> AsyncIterator[yuullm.StreamItem]:
            yield yuullm.Response(item={"type": "text", "text": reply})

        return _iter(), yuullm.Store()


def _make_llm(*replies: str) -> yuullm.YLLMClient:
    return yuullm.YLLMClient(
        provider=_FakeProvider(*replies),
        default_model="fake-model",
    )


class _FakeManager:
    def __init__(self) -> None:
        self.started: list[tuple[str, str, list[str] | None, int]] = []
        self.inspect_calls: list[tuple[str, int]] = []
        self.cancel_calls: list[str] = []
        self.defer_calls: list[tuple[str, str]] = []
        self.input_calls: list[tuple[str, str, bool]] = []
        self.wait_calls: list[list[str]] = []

    async def start_delegate(
        self,
        *,
        parent: Session,
        parent_run_id: str,
        agent: str,
        first_user_message: str,
        tools: list[str] | None,
        delegate_depth: int,
    ) -> Session:
        del parent
        self.started.append((parent_run_id, first_user_message, tools, delegate_depth))
        session = Session(
            config=AgentConfig(
                agent_id=agent,
                tools=yt.ToolManager(),
                llm=_make_llm("delegated done"),
                system="delegate system",
                max_steps=1,
            ),
            context=AgentContext(
                task_id="child-task",
                agent_id=agent,
                workdir="",
                docker_container="",
                manager=self,
            ),
        )
        session.start(first_user_message)
        return session

    def inspect_run(
        self,
        *,
        parent: Session,
        run_id: str,
        limit: int = 200,
        max_chars: int = 4000,
    ) -> str:
        del parent
        self.inspect_calls.append((run_id, limit))
        return f"inspect:{run_id}:{limit}:{max_chars}"

    def cancel_run(self, *, parent: Session, run_id: str) -> str:
        del parent
        self.cancel_calls.append(run_id)
        return f"cancel:{run_id}"

    def defer_run(self, *, parent: Session, run_id: str, message: str) -> str:
        del parent
        self.defer_calls.append((run_id, message))
        return f"defer:{run_id}:{message}"

    async def input_run(
        self,
        *,
        parent: Session,
        run_id: str,
        data: str,
        append_newline: bool = True,
    ) -> str:
        del parent
        self.input_calls.append((run_id, data, append_newline))
        return f"input:{run_id}:{data}:{append_newline}"

    async def wait_runs(self, *, parent: Session, run_ids: list[str]) -> str:
        del parent
        self.wait_calls.append(list(run_ids))
        return f"wait:{','.join(run_ids)}"


def _make_session(manager: _FakeManager, *, agent_id: str, reply: str) -> Session:
    return Session(
        config=AgentConfig(
            agent_id=agent_id,
            tools=yt.ToolManager(),
            llm=_make_llm(reply),
            system=f"{agent_id} system",
        ),
        context=AgentContext(
            task_id=f"{agent_id}-task",
            agent_id=agent_id,
            workdir="",
            docker_container="",
            manager=manager,
        ),
    )


def _bind(tool_obj: Any, ctx: AgentContext) -> Any:
    manager = yt.ToolManager()
    manager.register(tool_obj)
    return manager[tool_obj.spec.name].bind(ctx)


@pytest.mark.asyncio
async def test_delegate_tool_starts_child_session_and_returns_last_text():
    manager = _FakeManager()
    parent = _make_session(manager, agent_id="parent", reply="unused")
    parent.context = parent.context.evolve(session=parent)

    result = await _bind(
        delegate,
        parent.context.evolve(current_run_id="run-123"),
    ).run(
        agent="coder",
        context="repo=demo",
        task="fix bug",
        tools=["read_file"],
    )

    assert result == "delegated done"
    assert manager.started == [
        ("run-123", "Context:\nrepo=demo\n\nTask:\nfix bug", ["read_file"], 1)
    ]


@pytest.mark.asyncio
async def test_background_tools_call_manager():
    manager = _FakeManager()
    parent = _make_session(manager, agent_id="parent", reply="unused")

    ctx = AgentContext(
        task_id="task",
        agent_id="agent",
        workdir="",
        docker_container="",
        manager=manager,
        session=parent,
    )

    inspect_result = await _bind(inspect_background, ctx).run(
        run_id="bg-1",
        limit=50,
        max_chars=800,
    )
    cancel_result = await _bind(cancel_background, ctx).run(run_id="bg-2")
    input_result = await _bind(input_background, ctx).run(
        run_id="bg-4",
        data="hello",
        append_newline=False,
    )
    defer_result = await _bind(defer_background, ctx).run(
        run_id="bg-3",
        message="please report",
    )
    wait_result = await _bind(wait_background, ctx).run(run_ids=["bg-1", "bg-3"])

    assert inspect_result == "inspect:bg-1:50:800"
    assert cancel_result == "cancel:bg-2"
    assert input_result == "input:bg-4:hello:False"
    assert defer_result == "defer:bg-3:please report"
    assert wait_result == "wait:bg-1,bg-3"
    assert manager.inspect_calls == [("bg-1", 50)]
    assert manager.cancel_calls == ["bg-2"]
    assert manager.input_calls == [("bg-4", "hello", False)]
    assert manager.defer_calls == [("bg-3", "please report")]
    assert manager.wait_calls == [["bg-1", "bg-3"]]
