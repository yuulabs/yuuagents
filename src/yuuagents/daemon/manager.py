"""AgentManager — daemon host for session trees."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from loguru import logger

import yuullm
import yuutools as yt
from attrs import define, field
from yuullm.types import is_text_item, is_tool_call_item

from yuuagents.agent import AgentConfig
from yuuagents.capabilities import AgentCapabilities, DockerCapability, WebCapability
from yuuagents.config import Config, ProviderConfig
from yuuagents.context import AgentContext
from yuuagents.daemon.docker import DOCKER_SYSTEM_PROMPT
from yuuagents.input import AgentInput
from yuuagents.persistence import EphemeralPersistence, Persistence, RestoredTask
from yuuagents.pool import AgentPool
from yuuagents.prompts import get_vars as get_prompt_vars
from yuuagents.runtime_session import Session
from yuuagents.tools import get as get_builtin_tools
from yuuagents.types import AgentInfo, AgentStatus, TaskRequest


def _build_system_prompt(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


_DOCKER_TOOL_NAMES = frozenset({
    "execute_bash",
    "read_file",
    "edit_file",
    "delete_file",
})

_POOL_TOOL_NAMES = frozenset({
    "delegate",
    "inspect_background",
    "cancel_background",
    "input_background",
    "defer_background",
    "wait_background",
})

_WEB_TOOL_NAMES = frozenset({"web_search"})


@define
class AgentManager:
    config: Config
    docker: Any
    persistence: Persistence = field(factory=EphemeralPersistence)
    _pool: AgentPool | None = field(default=None, init=False)

    async def setup(self) -> None:
        """Initialize all sub-components, then create the pool."""
        await self.persistence.start()
        await self.docker.start()
        self._pool = AgentPool(
            persistence=self.persistence,
            session_builder=self._build_child_session_for_pool,
        )
        if self.config.snapshot.restore_on_start:
            await self._restore_unfinished_tasks()

    async def stop(self) -> None:
        if self._pool is not None:
            await self._pool.stop()
        await self.docker.stop()
        await self.persistence.stop()

    async def submit(self, req: TaskRequest, *, delegate_depth: int = 0) -> str:
        assert self._pool is not None, "AgentManager.setup() not called"
        task_id = uuid4().hex
        session = await self._build_root_session(
            task_id=task_id,
            req=req,
            delegate_depth=delegate_depth,
        )
        await self._pool.run(session, req.input)
        return task_id

    # -- Service API ---------------------------------------------------------

    async def list_agents(self) -> list[AgentInfo]:
        infos = await self.persistence.list_tasks()
        by_id = {info.task_id: info for info in infos}
        if self._pool is not None:
            for session in self._pool.iter_sessions():
                by_id[session.task_id] = self._info(session)
        return list(by_id.values())

    async def status(self, task_id: str) -> AgentInfo:
        if self._pool is not None:
            session = self._pool.get_session(task_id)
            if session is not None:
                return self._info(session)
        info = await self.persistence.get_task_info(task_id)
        if info is None:
            raise KeyError(task_id)
        return info

    async def history(self, task_id: str) -> list[Any]:
        if self._pool is not None:
            session = self._pool.get_session(task_id)
            if session is not None:
                return list(session.history)
        return await self.persistence.load_history(task_id)

    async def respond(self, task_id: str, message: yuullm.Message) -> None:
        if self._pool is None:
            raise KeyError(task_id)
        session = self._pool.get_session(task_id)
        if session is None:
            raise KeyError(task_id)
        session.send(message)

    async def cancel_task(self, task_id: str) -> None:
        if self._pool is None:
            raise KeyError(task_id)
        session = self._pool.get_session(task_id)
        if session is None:
            raise KeyError(task_id)
        session.cancel()
        session.status = AgentStatus.CANCELLED
        await self.persistence.update_task_terminal(
            task_id=task_id,
            status=AgentStatus.CANCELLED,
            error_json=None,
        )

    def reload_config(self, new_config: Config) -> None:
        self.config = new_config

    # -- Internal ------------------------------------------------------------

    async def _build_child_session_for_pool(
        self,
        *,
        parent: Session,
        task_id: str,
        parent_run_id: str,
        agent: str,
        input: AgentInput,
        tools: list[str] | None,
        delegate_depth: int,
    ) -> Session:
        req = TaskRequest(agent=agent, input=input, tools=tools or [])
        return await self._build_child_session(
            parent=parent,
            task_id=task_id,
            req=req,
            delegate_depth=delegate_depth,
        )

    async def _restore_unfinished_tasks(self) -> None:
        assert self._pool is not None
        restored_tasks = await self.persistence.load_unfinished()
        for restored in restored_tasks:
            if restored.state is None:
                logger.warning(
                    "Skipping restore for task {} because no snapshot was found",
                    restored.task_id,
                )
                continue
            session = await self._build_restored_session(restored)
            self._pool._sessions[restored.task_id] = session
            self._pool._snapshot_turn[restored.task_id] = restored.head_turn
            await self._pool.run(session, None)

    async def _build_restored_session(self, restored: RestoredTask) -> Session:
        if restored.state is None:
            raise ValueError("restored task requires a snapshot state")
        req = TaskRequest(
            agent=restored.agent_id,
            persona="",
            input=restored.input,
            tools=list(restored.tools),
            model=restored.model,
            container=restored.docker_container,
            image="",
        )
        session = await self._build_session(
            task_id=restored.task_id,
            req=req,
            delegate_depth=0,
            system_prompt_override=restored.system_prompt,
            docker_capability=(
                DockerCapability(
                    executor=self.docker,
                    container_id=restored.docker_container,
                )
                if restored.docker_container
                else None
            ),
        )
        conversation_id = restored.state.conversation_id
        session.resume(
            None,
            history=list(restored.state.messages),
            conversation_id=UUID(conversation_id) if conversation_id else None,
            system=restored.system_prompt,
        )
        if session.agent is not None:
            session.agent.rounds = restored.state.rounds
            session.agent.total_usage = restored.state.total_usage
            session.agent.total_cost_usd = restored.state.total_cost_usd
        session.history = list(restored.state.messages)
        session.status = restored.status
        session.stored_steps = restored.state.rounds
        return session

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

    def _needs_pool(self, tool_names: list[str]) -> bool:
        return any(name in _POOL_TOOL_NAMES for name in tool_names)

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
        system_prompt_override: str | None = None,
    ) -> Session:
        agent_id = req.agent
        agent_entry = self.config.agents.get(agent_id)
        llm_client = self._make_llm(agent_name=agent_id, model_override=req.model)
        tool_names = req.tools or self._default_tools(agent_id)
        tools = yt.ToolManager(get_builtin_tools(tool_names))
        needs_docker = self._needs_docker(tool_names)
        if system_prompt_override is not None:
            system_prompt = system_prompt_override
        else:
            persona_text = (
                req.persona
                or (agent_entry.persona if agent_entry and agent_entry.persona else "")
                or agent_id
            )
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
        )
        if needs_docker and docker_capability is None:
            try:
                docker_container = await self.docker.resolve(
                    task_id=task_id,
                    container=req.container,
                    image=req.image,
                )
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(
                    "docker tools requested but Docker is unavailable; "
                    "install `yuuagents[docker]` and ensure Docker Engine is reachable"
                ) from exc
            docker_capability = DockerCapability(
                executor=self.docker,
                container_id=docker_container,
            )
        elif not needs_docker:
            docker_capability = None

        capabilities = AgentCapabilities(
            docker=docker_capability,
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
            pool=self._pool if self._needs_pool(tool_names) else None,
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
        await self.persistence.create_task(
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
