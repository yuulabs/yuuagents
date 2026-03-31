from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest
import yuullm
import yuutools as yt

from yuuagents.input import ConversationInput
from yuuagents.local import LocalAgent, run_once


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
async def test_run_once_builds_local_session_automatically(tmp_path) -> None:
    client = make_client(
        [[yuullm.Response(item={"type": "text", "text": "done"})]]
    )

    result = await run_once(
        "say hi",
        llm=client,
        system="you are helpful",
        workdir=str(tmp_path),
    )

    assert result.output_text == "done"
    assert result.session.agent_id == "local"
    assert result.session.context.workdir == str(tmp_path)
    assert result.session.context.capabilities.docker is None
    assert isinstance(result.input, ConversationInput)
    assert result.input.messages == [yuullm.user("say hi")]
    assert result.messages[-1] == (
        "assistant",
        [{"type": "text", "text": "done"}],
    )


@pytest.mark.asyncio
async def test_local_agent_step_iter_exposes_progress_and_final_result(tmp_path) -> None:
    client = make_client(
        [
            [yuullm.ToolCall(id="call-1", name="noop", arguments="{}")],
            [yuullm.Response(item={"type": "text", "text": "finished"})],
        ]
    )

    @yt.tool(description="No-op tool", params={})
    async def noop() -> str:
        return "ok"

    agent = LocalAgent.create(
        llm=client,
        tools=[noop],
        workdir=str(tmp_path),
        system="test system",
        agent_id="sdk-agent",
    )

    run = agent.start("do work")
    steps = [step async for step in run.step_iter()]
    result = await run.result()

    assert [step.done for step in steps] == [False, True]
    assert result.output_text == "finished"
    assert result.steps == tuple(steps)
    assert result.session.agent_id == "sdk-agent"
    assert result.session.context.workdir == str(tmp_path)


@pytest.mark.asyncio
async def test_local_run_requires_step_iter_to_finish_before_result() -> None:
    client = make_client(
        [[yuullm.Response(item={"type": "text", "text": "hello"})]]
    )
    run = LocalAgent.create(llm=client).start("hello")

    iterator = run.step_iter()
    await anext(iterator)

    with pytest.raises(RuntimeError, match="exhaust it before calling result"):
        await run.result()
