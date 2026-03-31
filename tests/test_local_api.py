from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
import yuullm
import yuutools as yt

from yuuagents.agent import AgentConfig
from yuuagents.context import AgentContext
from yuuagents.input import conversation_input_from_text
from yuuagents.local import final_response, run_once
from yuuagents.runtime_session import Session


class FakeProvider:
    def __init__(self, script: Sequence[Sequence[yuullm.StreamItem]]) -> None:
        self._script = [list(items) for items in script]
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
        **kwargs: object,
    ) -> yuullm.StreamResult:
        del messages, model, tools, kwargs
        items = self._script[self._call_index]
        self._call_index += 1

        async def _gen() -> AsyncIterator[yuullm.StreamItem]:
            for item in items:
                yield item

        return _gen(), yuullm.Store()


def make_client(script: Sequence[Sequence[yuullm.StreamItem]]) -> yuullm.YLLMClient:
    return yuullm.YLLMClient(provider=FakeProvider(script), default_model="fake-model")


@pytest.mark.asyncio
async def test_run_once_returns_completed_session(tmp_path) -> None:
    client = make_client(
        [[yuullm.Response(item={"type": "text", "text": "done"})]]
    )

    session = await run_once(
        "say hi",
        llm=client,
        system="you are helpful",
        workdir=str(tmp_path),
    )

    assert session.agent_id == "local"
    assert session.context.workdir == str(tmp_path)
    assert final_response(session.history) == "done"
    assert session.history[-1] == ("assistant", [{"type": "text", "text": "done"}])


@pytest.mark.asyncio
async def test_step_iter_delta_carries_new_messages(tmp_path) -> None:
    client = make_client(
        [
            [yuullm.ToolCall(id="call-1", name="noop", arguments="{}")],
            [yuullm.Response(item={"type": "text", "text": "finished"})],
        ]
    )

    @yt.tool(description="No-op tool", params={})
    async def noop() -> str:
        return "ok"

    config = AgentConfig(
        agent_id="sdk-agent",
        tools=yt.ToolManager([noop]),
        llm=client,
        system="",
    )
    ctx = AgentContext(task_id="test-task", agent_id="sdk-agent", workdir=str(tmp_path))
    session = Session(config=config, context=ctx)
    session.start(conversation_input_from_text("do work"))

    steps = [step async for step in session.step_iter()]

    assert [step.done for step in steps] == [False, True]
    # round 1: assistant tool-call message + tool result
    assert steps[0].delta[0][0] == "assistant"
    assert steps[0].delta[1][0] == "tool"
    # round 2: final assistant text
    assert steps[1].delta[0] == ("assistant", [{"type": "text", "text": "finished"}])
    assert final_response(session.history) == "finished"


def test_final_response_returns_empty_for_no_history() -> None:
    assert final_response([]) == ""


def test_final_response_skips_non_assistant_messages() -> None:
    messages: list[yuullm.Message] = [
        yuullm.user("hello"),
        yuullm.assistant({"type": "text", "text": "hi"}),
        yuullm.user("ok"),
    ]
    assert final_response(messages) == "hi"
