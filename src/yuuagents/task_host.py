"""TaskHost — SDK-first host over Flow/Basin rooted agents."""

from __future__ import annotations

import asyncio
import traceback
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import attrs
import msgspec
import yuullm
from attrs import define, field
from yuullm.types import is_text_item, is_tool_call_item

from yuuagents.agent import AgentConfig
from yuuagents.basin import Basin
from yuuagents.core.flow import Agent, Flow, FlowState
from yuuagents.input import AgentInput, agent_input_preview, render_message_text
from yuuagents.persistence import EphemeralPersistence, Persistence, RestoredTask
from yuuagents.types import AgentInfo, AgentStatus, ErrorInfo, TaskRequest


@define(frozen=True)
class BuiltRoot:
    """A fully constructed, already-started root or child agent."""

    agent: Agent
    input: AgentInput
    created_at: datetime
    system_prompt: str
    model: str
    tools: list[str]
    docker_container: str


BuildRoot = Callable[[str, TaskRequest, int], Awaitable[BuiltRoot]]
RestoreRoot = Callable[[RestoredTask], Awaitable[BuiltRoot]]


@define
class _RootRuntime:
    task_id: str
    built: BuiltRoot
    status: AgentStatus = AgentStatus.IDLE
    error: ErrorInfo | None = None
    snapshot_turn: int = 0

    @property
    def agent(self) -> Agent:
        return self.built.agent

    @property
    def created_at(self) -> datetime:
        return self.built.created_at


def _make_error(exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        message=traceback.format_exc(),
        error_type=type(exc).__name__,
        timestamp=datetime.now(timezone.utc),
    )


def _last_assistant_message(messages: list[yuullm.Message]) -> str:
    for role, items in reversed(messages):
        if role != "assistant":
            continue
        text_parts: list[str] = []
        tool_names: list[str] = []
        for item in items:
            if is_text_item(item):
                text_parts.append(item["text"])
            elif is_tool_call_item(item):
                tool_names.append(item["name"])
        text = "".join(text_parts).strip()
        if text:
            return text
        if tool_names:
            return "\n".join(f"tool_call: {name}" for name in tool_names)
    return ""


@define
class TaskHost:
    """Owns root task execution and indexes every live flow through Basin."""

    basin: Basin
    build_root: BuildRoot
    restore_root: RestoreRoot | None = None
    persistence: Persistence = field(factory=EphemeralPersistence)
    _roots: dict[str, _RootRuntime] = field(factory=dict, init=False)
    _agents: dict[str, Agent] = field(factory=dict, init=False)

    async def start(self) -> None:
        await self.persistence.start()

    async def stop(self) -> None:
        for runtime in self._roots.values():
            if runtime.status == AgentStatus.RUNNING:
                runtime.agent.flow.cancel()
        waits = [
            runtime.agent.flow.wait()
            for runtime in self._roots.values()
            if runtime.agent.flow._task is not None
        ]
        if waits:
            await asyncio.gather(*waits, return_exceptions=True)
        await self.persistence.stop()

    async def submit(self, req: TaskRequest, *, delegate_depth: int = 0) -> str:
        task_id = uuid4().hex
        built = await self.build_root(task_id, req, delegate_depth)
        runtime = _RootRuntime(
            task_id=task_id,
            built=attrs.evolve(built, created_at=built.created_at),
            status=AgentStatus.RUNNING,
        )
        self._roots[task_id] = runtime
        self._agents[built.agent.flow.id] = built.agent
        await self.persistence.create_task(
            task_id=task_id,
            agent_id=built.agent.config.agent_id,
            persona=built.system_prompt[:200],
            input=built.input,
            system_prompt=built.system_prompt,
            model=built.model,
            tools=built.tools,
            docker_container=built.docker_container,
            created_at=built.created_at,
        )
        built.agent.flow.start(self._drive_root(runtime))
        return task_id

    async def restore_unfinished(self) -> None:
        if self.restore_root is None:
            return
        restored_tasks = await self.persistence.load_unfinished()
        for restored in restored_tasks:
            if restored.state is None:
                continue
            built = await self.restore_root(restored)
            runtime = _RootRuntime(
                task_id=restored.task_id,
                built=attrs.evolve(built, created_at=restored.created_at),
                status=AgentStatus.RUNNING,
            )
            runtime.snapshot_turn = restored.head_turn
            self._roots[restored.task_id] = runtime
            self._agents[built.agent.flow.id] = built.agent
            built.agent.flow.start(self._drive_root(runtime))

    async def start_child_agent(
        self,
        *,
        parent_flow: Flow[Any, Any],
        agent: Agent,
    ) -> Agent:
        agent.flow.parent = parent_flow
        if agent.flow not in parent_flow.children:
            parent_flow.children.append(agent.flow)
        self._agents[agent.flow.id] = agent
        if agent.flow._task is None:
            agent.flow.start(self._drive_child(agent))
        return agent

    async def list_tasks(self) -> list[AgentInfo]:
        infos = await self.persistence.list_tasks()
        by_id = {info.task_id: info for info in infos}
        for task_id, runtime in self._roots.items():
            by_id[task_id] = self._info(runtime)
        return list(by_id.values())

    async def status(self, task_id: str) -> AgentInfo:
        runtime = self._roots.get(task_id)
        if runtime is not None:
            return self._info(runtime)
        info = await self.persistence.get_task_info(task_id)
        if info is None:
            raise KeyError(task_id)
        return info

    async def history(self, task_id: str) -> list[Any]:
        runtime = self._roots.get(task_id)
        if runtime is not None:
            return list(runtime.agent.messages)
        return await self.persistence.load_history(task_id)

    async def send(self, flow_id: str, content: Any) -> None:
        runtime = self._roots.get(flow_id)
        if runtime is not None and runtime.status is not AgentStatus.RUNNING:
            raise RuntimeError(f"task {flow_id!r} is not running")
        try:
            flow = self.basin.require(flow_id)
        except KeyError as exc:
            raise KeyError(flow_id) from exc
        flow.send(content)

    async def cancel(self, flow_id: str) -> None:
        try:
            flow = self.basin.require(flow_id)
        except KeyError as exc:
            raise KeyError(flow_id) from exc
        flow.cancel()
        try:
            await flow.wait()
        except asyncio.CancelledError:
            pass

    async def wait(self, flow_id: str) -> None:
        try:
            flow = self.basin.require(flow_id)
        except KeyError as exc:
            raise KeyError(flow_id) from exc
        await flow.wait()

    async def _drive_root(self, runtime: _RootRuntime) -> None:
        agent = runtime.agent
        try:
            async for _ in agent.steps():
                await self._persist_snapshot(runtime)
            runtime.status = AgentStatus.DONE
            await self._persist_snapshot(runtime)
            await self.persistence.update_task_terminal(
                task_id=runtime.task_id,
                status=AgentStatus.DONE,
                error_json=None,
            )
        except asyncio.CancelledError:
            await agent.kill()
            runtime.status = AgentStatus.CANCELLED
            agent.flow.state = FlowState.CANCELLED
            await self._persist_snapshot(runtime)
            await self.persistence.update_task_terminal(
                task_id=runtime.task_id,
                status=AgentStatus.CANCELLED,
                error_json=None,
            )
        except Exception as exc:
            await agent.kill()
            runtime.error = _make_error(exc)
            runtime.status = AgentStatus.ERROR
            agent.flow.state = FlowState.ERROR
            agent.flow.info["error"] = runtime.error.message
            await self._persist_snapshot(runtime)
            await self.persistence.update_task_terminal(
                task_id=runtime.task_id,
                status=AgentStatus.ERROR,
                error_json=msgspec.json.encode(runtime.error),
            )

    async def _drive_child(self, agent: Agent) -> None:
        try:
            async for _ in agent.steps():
                pass
        except asyncio.CancelledError:
            await agent.kill()
            agent.flow.state = FlowState.CANCELLED
        except Exception as exc:
            await agent.kill()
            agent.flow.state = FlowState.ERROR
            agent.flow.info["error"] = str(exc)

    async def _persist_snapshot(self, runtime: _RootRuntime) -> None:
        state = await runtime.agent.snapshot()
        runtime.snapshot_turn += 1
        await self.persistence.save_snapshot(
            task_id=runtime.task_id,
            turn=runtime.snapshot_turn,
            state=state,
            status=runtime.status,
        )

    def _info(self, runtime: _RootRuntime) -> AgentInfo:
        agent = runtime.agent
        return AgentInfo(
            task_id=runtime.task_id,
            agent_id=agent.config.agent_id,
            persona=runtime.built.system_prompt[:80],
            input_kind=runtime.built.input.kind,
            input_preview=agent_input_preview(runtime.built.input),
            status=runtime.status.value,
            created_at=runtime.created_at.isoformat(),
            last_assistant_message=_last_assistant_message(agent.messages),
            steps=agent.rounds,
            total_tokens=agent.total_tokens,
            last_usage=agent.last_usage,
            total_usage=agent.total_usage,
            last_cost_usd=agent.last_cost_usd,
            total_cost_usd=agent.total_cost_usd,
            error=runtime.error,
        )
