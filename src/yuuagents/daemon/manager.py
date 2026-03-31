"""AgentManager — daemon host for session trees."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

import msgspec
import yuullm
import yuutools as yt
from attrs import define, field
from yuullm.types import is_text_item, is_tool_call_item

from yuuagents.agent import AgentConfig
from yuuagents.capabilities import AgentCapabilities, DockerCapability, WebCapability
from yuuagents.config import Config, ProviderConfig
from yuuagents.context import AgentContext
from yuuagents.core.flow import render_agent_event
from yuuagents.daemon.docker import DOCKER_SYSTEM_PROMPT, DockerManager
from yuuagents.input import AgentInput
from yuuagents.persistence import TaskPersistence, TaskWriter
from yuuagents.prompts import get_vars as get_prompt_vars
from yuuagents.runtime_session import Session
from yuuagents.tools import get as get_builtin_tools
from yuuagents.types import AgentInfo, AgentStatus, ErrorInfo, TaskRequest


def _make_error(exc: Exception) -> ErrorInfo:
    import traceback
    from datetime import datetime, timezone

    return ErrorInfo(
        message=traceback.format_exc(),
        error_type=type(exc).__name__,
        timestamp=datetime.now(timezone.utc),
    )


def _build_system_prompt(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


_DOCKER_TOOL_NAMES = frozenset({
    "execute_bash",
    "read_file",
    "edit_file",
    "delete_file",
})

_DELEGATE_TOOL_NAMES = frozenset({
    "delegate",
    "inspect_background",
    "cancel_background",
    "input_background",
    "defer_background",
    "wait_background",
})

_WEB_TOOL_NAMES = frozenset({"web_search"})


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 32:
        return text[:max_chars]
    return text[: max_chars - 15].rstrip() + "\n...[truncated]"


@define
class AgentManager:
    config: Config
    docker: DockerManager
    db_url: str = ""
    _sessions: dict[str, Session] = field(factory=dict)
    _contexts: dict[str, AgentContext] = field(factory=dict)
    _tasks: dict[str, asyncio.Task[None]] = field(factory=dict)
    _persistence: TaskPersistence | None = None
    _writer: TaskWriter | None = None
    _snapshot_turn: dict[str, int] = field(factory=dict)  # task_id -> last persisted turn
    _delegate_sessions_by_run_id: dict[str, Session] = field(factory=dict)

    async def start(self) -> None:
        await self.docker.start()
        db_url = self.db_url or self.config.db_url
        self._persistence = TaskPersistence(db_url=db_url)
        await self._persistence.start()
        self._writer = TaskWriter(self._persistence)
        await self._writer.start()
        # Daemon crash = tasks lost; no restore on startup.

    async def stop(self) -> None:
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._sessions.clear()
        self._contexts.clear()
        self._snapshot_turn.clear()
        self._delegate_sessions_by_run_id.clear()
        if self._writer is not None:
            await self._writer.stop()
            self._writer = None
        if self._persistence is not None:
            await self._persistence.stop()
            self._persistence = None
        await self.docker.stop()

    async def submit(self, req: TaskRequest, *, delegate_depth: int = 0) -> str:
        task_id = uuid4().hex
        session = await self._build_root_session(
            task_id=task_id,
            req=req,
            delegate_depth=delegate_depth,
        )
        self._sessions[task_id] = session
        self._contexts[task_id] = session.context
        self._tasks[task_id] = asyncio.create_task(self._run(session, req.input))
        return task_id

    async def start_delegate(
        self,
        *,
        parent: Session,
        parent_run_id: str,
        agent: str,
        input: AgentInput,
        tools: list[str] | None,
        delegate_depth: int,
    ) -> Session:
        task_id = uuid4().hex
        req = TaskRequest(agent=agent, input=input, tools=tools or [])
        child = await self._build_child_session(
            parent=parent,
            task_id=task_id,
            req=req,
            delegate_depth=delegate_depth,
        )
        self._sessions[task_id] = child
        self._contexts[task_id] = child.context
        self._delegate_sessions_by_run_id[parent_run_id] = child
        child.status = AgentStatus.RUNNING
        child.start(input)
        parent_agent = parent.agent
        child_agent = child.agent
        if (
            parent_agent is not None
            and child_agent is not None
            and child_agent.flow not in parent_agent.flow.children
        ):
            parent_flow = parent_agent.flow.find(parent_run_id)
            if parent_flow is not None and child_agent.flow not in parent_flow.children:
                parent_flow.children.append(child_agent.flow)
        return child

    def inspect_run(
        self,
        *,
        parent: Session,
        run_id: str,
        limit: int = 200,
        max_chars: int = 4000,
    ) -> str:
        agent = parent.agent
        if agent is None:
            return f"[ERROR] parent session not started for run {run_id}"
        flow = agent.flow.find(run_id)
        if flow is None:
            return f"[ERROR] unknown run id {run_id!r}"
        lines = [
            f"run_id: {flow.id}",
            f"kind: {flow.kind}",
        ]
        input_kind = flow.info.get("input_kind")
        if isinstance(input_kind, str) and input_kind:
            lines.append(f"input_kind: {input_kind}")
        input_fields = flow.info.get("input_fields")
        if isinstance(input_fields, dict) and input_fields:
            lines.append("input_fields:")
            for name, text in input_fields.items():
                lines.append(f"{name}: {text}")
        tool_name = flow.info.get("tool_name")
        if isinstance(tool_name, str) and tool_name:
            lines.append(f"tool_name: {tool_name}")
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is not None:
            lines.append(f"delegate_task_id: {delegate.task_id}")
            lines.append(f"delegate_status: {delegate.status.value}")
            lines.append(f"delegate_input_kind: {delegate.input_kind}")
            lines.append(f"delegate_input_preview: {delegate.input_preview}")
            delegate_agent = delegate.agent
            if delegate_agent is not None:
                lines.append("delegate_stem:")
                lines.append(delegate_agent.render(limit=limit) or "<empty>")
        rendered = flow.render(render_agent_event if flow.kind == "agent" else str, limit=limit)
        lines.append("stem:")
        lines.append(rendered or "<empty>")
        return _truncate_text("\n".join(lines), max_chars)

    def cancel_run(
        self,
        *,
        parent: Session,
        run_id: str,
    ) -> str:
        agent = parent.agent
        if agent is None:
            return f"[ERROR] parent session not started for run {run_id}"
        flow = agent.flow.find(run_id)
        if flow is None:
            return f"[ERROR] unknown run id {run_id!r}"
        flow.cancel()
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is not None:
            delegate.status = AgentStatus.CANCELLED
        return f"Cancelled run {run_id}"

    def defer_run(
        self,
        *,
        parent: Session,
        run_id: str,
        message: str,
    ) -> str:
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is None:
            return f"[ERROR] run {run_id!r} is not a delegated agent"
        prompt = (
            message.strip()
            or "请立即停止等待中的前台工具，把当前工作移到后台，并先汇报简短进展。"
        )
        delegate.send(yuullm.user(prompt), defer_tools=True)
        return f"Sent defer signal to delegated run {run_id}"

    async def input_run(
        self,
        *,
        parent: Session,
        run_id: str,
        data: str,
        append_newline: bool = True,
    ) -> str:
        delegate = self._delegate_sessions_by_run_id.get(run_id)
        if delegate is not None:
            delegate.send(yuullm.user(data), defer_tools=False)
            return f"Input sent to delegated run {run_id}"
        agent = parent.agent
        if agent is None:
            return f"[ERROR] parent session not started for run {run_id}"
        flow = agent.flow.find(run_id)
        if flow is None:
            return f"[ERROR] unknown run id {run_id!r}"
        tool_name = flow.info.get("tool_name")
        if tool_name != "execute_bash":
            return f"[ERROR] run {run_id!r} does not accept input"
        docker = parent.context.capabilities.docker
        if docker is None or not docker.container_id:
            return f"[ERROR] docker terminal unavailable for run {run_id!r}"
        return await docker.executor.write_terminal(
            docker.container_id,
            run_id,
            data,
            append_newline=append_newline,
        )

    async def wait_runs(
        self,
        *,
        parent: Session,
        run_ids: list[str],
    ) -> str:
        agent = parent.agent
        if agent is None:
            return "[ERROR] parent session not started for wait"
        if not run_ids:
            return "[ERROR] run_ids must not be empty"

        waits: list[asyncio.Future | asyncio.Task[Any] | Any] = []
        for run_id in run_ids:
            delegate = self._delegate_sessions_by_run_id.get(run_id)
            if delegate is not None:
                task = self._tasks.get(delegate.task_id)
                if task is None:
                    return f"[ERROR] delegated run {run_id!r} not started"
                waits.append(task)
                continue
            flow = agent.flow.find(run_id)
            if flow is None:
                return f"[ERROR] unknown run id {run_id!r}"
            waits.append(flow.wait())

        await asyncio.gather(*waits)
        ids = ", ".join(run_ids)
        return f"Wait finished for runs: {ids}"

    async def _run(self, session: Session, agent_input: AgentInput) -> None:
        session.status = AgentStatus.RUNNING
        session.start(agent_input)
        try:
            async for _step in session.step_iter():
                if not session.has_pending_background:
                    await self._persist_snapshot(session)
            session.status = AgentStatus.DONE
            await self._write_terminal(session.task_id, AgentStatus.DONE, None)
        except asyncio.CancelledError:
            await session.kill()
            await self._persist_snapshot(session)
            session.status = AgentStatus.CANCELLED
            await self._write_terminal(session.task_id, AgentStatus.CANCELLED, None)
        except Exception as exc:
            await session.kill()
            err = _make_error(exc)
            session.status = AgentStatus.ERROR
            session.error = err
            err_json = msgspec.json.encode(err)
            await self._persist_snapshot(session)
            await self._write_terminal(session.task_id, AgentStatus.ERROR, err_json)
            logger.exception("agent {agent_id} task {task_id} failed",
                             agent_id=session.agent_id, task_id=session.task_id)

    async def list_agents(self) -> list[AgentInfo]:
        if self._persistence is None:
            return [self._info(session) for session in self._sessions.values()]
        infos = await self._persistence.list_tasks()
        by_id = {info.task_id: info for info in infos}
        for task_id, session in self._sessions.items():
            by_id[task_id] = self._info(session)
        return list(by_id.values())

    async def status(self, task_id: str) -> AgentInfo:
        session = self._sessions.get(task_id)
        if session is not None:
            return self._info(session)
        if self._persistence is None:
            raise KeyError(task_id)
        row = await self._persistence.get_task_row(task_id)
        if row is None:
            raise KeyError(task_id)
        err: ErrorInfo | None = None
        if row.error_json is not None:
            err = msgspec.json.decode(row.error_json, type=ErrorInfo)
        return AgentInfo(
            task_id=row.task_id,
            agent_id=row.agent_id,
            persona=row.persona[:80],
            input_kind=row.input_kind,
            input_preview=row.input_preview,
            status=row.status,
            created_at=row.created_at.isoformat(),
            last_assistant_message="",
            steps=row.head_turn,
            total_tokens=0,
            last_usage=None,
            total_usage=None,
            last_cost_usd=0.0,
            total_cost_usd=0.0,
            error=err,
        )

    async def history(self, task_id: str) -> list[Any]:
        session = self._sessions.get(task_id)
        if session is not None:
            return list(session.history)
        if self._persistence is None:
            raise KeyError(task_id)
        return await self._persistence.load_history(task_id)

    async def respond(self, task_id: str, message: yuullm.Message) -> None:
        session = self._sessions.get(task_id)
        if session is None:
            raise KeyError(task_id)
        session.send(message)

    async def cancel(self, task_id: str) -> None:
        if task_id not in self._sessions:
            raise KeyError(task_id)
        session = self._sessions.get(task_id)
        if session is not None:
            session.cancel()
            session.status = AgentStatus.CANCELLED
        task = self._tasks.get(task_id)
        if task is not None and not task.done():
            task.cancel()
        await self._write_terminal(task_id, AgentStatus.CANCELLED, None)

    def reload_config(self, new_config: Config) -> None:
        self.config = new_config

    def _default_agents_prompt(self, *, agent_id: str) -> str:
        if not self.config.agents:
            return ""
        entry = self.config.agents.get(agent_id)
        if entry is None or not entry.subagents:
            return ""
        allowed = entry.subagents
        if "*" in allowed:
            names = [name for name in sorted(self.config.agents) if name != agent_id]
        else:
            names = [name for name in allowed if name in self.config.agents and name != agent_id]
        if not names:
            return ""
        parts = [
            "<agents>",
            "以下是其他可调用的 Agent（不是你自己）。需要时使用 delegate 工具调用。",
        ]
        for name in names:
            other = self.config.agents[name]
            desc = " ".join(other.description.strip().split())
            if len(desc) > 200:
                desc = desc[:197] + "..."
            parts.append(f"- name: {name}\n- description: {desc}")
        parts.append("</agents>")
        return "\n".join(parts)

    async def _persist_snapshot(
        self,
        session: Session,
        *,
        as_interrupted: bool = False,
    ) -> None:
        """Persist the current agent state as a snapshot checkpoint."""
        if self._writer is None:
            return
        try:
            state = await session.snapshot(as_interrupted=as_interrupted)
        except RuntimeError:
            return
        turn = self._snapshot_turn.get(session.task_id, 0) + 1
        self._snapshot_turn[session.task_id] = turn
        from yuuagents.persistence import _BufferedCheckpoint
        from datetime import datetime, timezone
        payload = msgspec.json.encode(state)
        cp = _BufferedCheckpoint(
            task_id=session.task_id,
            turn=turn,
            phase="snapshot",
            ts=datetime.now(timezone.utc),
            payload=payload,
            status_after=session.status,
        )
        await self._writer.append_checkpoint(cp)

    async def _write_terminal(
        self,
        task_id: str,
        status: AgentStatus,
        error_json: bytes | None,
    ) -> None:
        if self._persistence is None:
            return
        if status in (AgentStatus.ERROR, AgentStatus.CANCELLED, AgentStatus.DONE):
            await self._persistence.update_task_terminal(
                task_id=task_id,
                status=status,
                error_json=error_json,
            )

    def _last_assistant_message(self, session: Session) -> str:
        for role, items in reversed(session.history):
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

    def _info(self, session: Session) -> AgentInfo:
        return AgentInfo(
            task_id=session.task_id,
            agent_id=session.agent_id,
            persona=session.config.system[:80],
            input_kind=session.input_kind,
            input_preview=session.input_preview,
            status=session.status.value,
            created_at=session.created_at.isoformat(),
            last_assistant_message=self._last_assistant_message(session),
            steps=session.steps,
            total_tokens=session.total_tokens,
            last_usage=session.last_usage,
            total_usage=session.total_usage,
            last_cost_usd=session.last_cost_usd,
            total_cost_usd=session.total_cost_usd,
            error=session.error,
        )

    def _make_llm(
        self,
        agent_name: str,
        model_override: str = "",
    ) -> yuullm.YLLMClient:
        agent_entry = self.config.agents.get(agent_name)
        provider_cfg: ProviderConfig | None = None
        provider_name = agent_entry.provider if agent_entry and agent_entry.provider else ""
        if agent_entry and agent_entry.provider:
            provider_cfg = self.config.providers.get(agent_entry.provider)
        if provider_cfg is None and self.config.providers:
            provider_name, provider_cfg = next(iter(self.config.providers.items()))
        if provider_cfg is None:
            provider_cfg = ProviderConfig()
            provider_name = "openai"
        model = (
            model_override
            or (agent_entry.model if agent_entry else "")
            or provider_cfg.default_model
        )
        api_key = os.environ.get(provider_cfg.api_key_env, "")
        provider: yuullm.Provider
        match provider_cfg.api_type:
            case "anthropic-messages":
                provider = yuullm.providers.AnthropicMessagesProvider(
                    api_key=api_key,
                    base_url=provider_cfg.base_url or None,
                    provider_name=provider_name,
                )
            case "openai-chat-completion":
                provider = yuullm.providers.OpenAIChatCompletionProvider(
                    api_key=api_key,
                    base_url=provider_cfg.base_url or None,
                    organization=provider_cfg.organization or None,
                    provider_name=provider_name,
                )
            case "openai-responses":
                provider_cls = getattr(yuullm.providers, "OpenAIResponsesProvider", None)
                if provider_cls is None:
                    raise RuntimeError(
                        "api_type 'openai-responses' requires yuullm.providers.OpenAIResponsesProvider"
                    )
                kwargs: dict[str, Any] = {
                    "api_key": api_key,
                    "base_url": provider_cfg.base_url or None,
                    "organization": provider_cfg.organization or None,
                    "provider_name": provider_name,
                }
                provider = provider_cls(**kwargs)
            case _:
                raise ValueError(f"unknown api_type {provider_cfg.api_type!r}")
        price_calc = (
            self._build_price_calculator(provider_name, provider_cfg)
            if provider_cfg.pricing
            else yuullm.PriceCalculator()
        )
        return yuullm.YLLMClient(
            provider=provider,
            default_model=model,
            price_calculator=price_calc,
        )

    @staticmethod
    def _build_price_calculator(
        provider_name: str,
        provider_cfg: ProviderConfig,
    ) -> yuullm.PriceCalculator:
        import yaml

        models = [
            {
                "id": entry.model,
                "prices": {
                    "input_mtok": entry.input_mtok,
                    "output_mtok": entry.output_mtok,
                    "cache_read_mtok": entry.cache_read_mtok,
                    "cache_write_mtok": entry.cache_write_mtok,
                },
            }
            for entry in provider_cfg.pricing
        ]
        data = [{"provider": provider_name, "models": models}]
        fd = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
        tmp = Path(fd.name)
        fd.close()
        try:
            tmp.write_text(yaml.dump(data), encoding="utf-8")
            return yuullm.PriceCalculator(yaml_path=tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def _default_tools(self, agent_name: str) -> list[str]:
        entry = self.config.agents.get(agent_name)
        if entry and entry.tools:
            return entry.tools
        return [
            "execute_bash",
            "inspect_background",
            "cancel_background",
            "input_background",
            "defer_background",
            "wait_background",
            "delegate",
            "read_file",
            "edit_file",
            "delete_file",
            "web_search",
        ]

    def _needs_docker(self, tool_names: list[str]) -> bool:
        return any(name in _DOCKER_TOOL_NAMES for name in tool_names)

    def _needs_delegate(self, tool_names: list[str]) -> bool:
        return any(name in _DELEGATE_TOOL_NAMES for name in tool_names)

    def _needs_web(self, tool_names: list[str]) -> bool:
        return any(name in _WEB_TOOL_NAMES for name in tool_names)

    async def _build_session(
        self,
        *,
        task_id: str,
        req: TaskRequest,
        delegate_depth: int,
        workdir: str = "",
        docker_capability: DockerCapability | None = None,
    ) -> Session:
        """Build a Session for both root and child (delegate) use cases.

        For root sessions, pass explicit capabilities as ``None`` to use defaults.
        For child sessions, pass inherited capabilities from the parent.
        """
        agent_id = req.agent
        agent_entry = self.config.agents.get(agent_id)
        llm_client = self._make_llm(agent_name=agent_id, model_override=req.model)
        tool_names = req.tools or self._default_tools(agent_id)
        tools = yt.ToolManager(get_builtin_tools(tool_names))
        needs_docker = self._needs_docker(tool_names)
        persona_text = (
            req.persona
            or (agent_entry.persona if agent_entry and agent_entry.persona else "")
            or agent_id
        )
        # Root sessions apply prompt variable substitution
        if docker_capability is None:
            prompt_vars = get_prompt_vars()
            for key, value in prompt_vars.items():
                persona_text = persona_text.replace(f"{{{key}}}", value)
        agents_prompt = self._default_agents_prompt(agent_id=agent_id)
        docker_prompt = DOCKER_SYSTEM_PROMPT if needs_docker else ""
        system_prompt = _build_system_prompt(persona_text, agents_prompt, docker_prompt)
        config = AgentConfig(
            agent_id=agent_id,
            tools=tools,
            llm=llm_client,
            system=system_prompt,
            soft_timeout=(agent_entry.soft_timeout or None) if agent_entry else None,
            silence_timeout=(agent_entry.silence_timeout or None) if agent_entry else None,
        )
        # Resolve docker container for root, or reuse parent's for child.
        if needs_docker and docker_capability is None:
            docker_container = await self.docker.resolve(
                task_id=task_id,
                container=req.container,
                image=req.image,
            )
            docker_capability = DockerCapability(
                executor=self.docker,
                container_id=docker_container,
            )
        elif not needs_docker:
            docker_capability = None

        capabilities = AgentCapabilities(
            docker=docker_capability,
            delegate=self if self._needs_delegate(tool_names) else None,
            web=(
                WebCapability(
                    api_key=os.environ.get(self.config.tavily.api_key_env, ""),
                )
                if self._needs_web(tool_names)
                else None
            ),
        )
        ctx = AgentContext(
            task_id=task_id,
            agent_id=agent_id,
            workdir=workdir or (
                self.docker.workdir if docker_capability is not None else str(Path.cwd())
            ),
            capabilities=capabilities,
            delegate_depth=delegate_depth,
        )
        session = Session(
            config=config,
            context=ctx,
            input=req.input,
            mailbox_id=task_id,
        )
        return session

    async def _build_root_session(
        self,
        *,
        task_id: str,
        req: TaskRequest,
        delegate_depth: int,
    ) -> Session:
        session = await self._build_session(
            task_id=task_id, req=req, delegate_depth=delegate_depth,
        )
        if self._persistence is not None:
            await self._persistence.create_task(
                task_id=task_id,
                agent_id=req.agent,
                persona=session.config.system[:200],
                input=req.input,
                system_prompt=session.config.system,
                model=session.config.llm.default_model,
                tools=req.tools or self._default_tools(req.agent),
                docker_container=(
                    session.context.capabilities.docker.container_id
                    if session.context.capabilities.docker is not None
                    else ""
                ),
                created_at=session.created_at,
            )
        return session

    async def _build_child_session(
        self,
        *,
        parent: Session,
        task_id: str,
        req: TaskRequest,
        delegate_depth: int,
    ) -> Session:
        return await self._build_session(
            task_id=task_id,
            req=req,
            delegate_depth=delegate_depth,
            workdir=parent.context.workdir,
            docker_capability=parent.context.capabilities.docker,
        )
