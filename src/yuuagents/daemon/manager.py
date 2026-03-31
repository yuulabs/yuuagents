"""AgentManager — daemon adapter over TaskHost."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import yuullm
import yuutools as yt
from attrs import define, field

from yuuagents.agent import AgentConfig
from yuuagents.basin import Basin
from yuuagents.capabilities import AgentCapabilities, DockerCapability, WebCapability
from yuuagents.config import Config, ProviderConfig
from yuuagents.context import AgentContext
from yuuagents.core.flow import Agent, Flow
from yuuagents.daemon.docker import DOCKER_SYSTEM_PROMPT
from yuuagents.input import AgentInput
from yuuagents.persistence import EphemeralPersistence, Persistence, RestoredTask
from yuuagents.prompts import get_vars as get_prompt_vars
from yuuagents.task_host import BuiltRoot, TaskHost
from yuuagents.tools import get as get_builtin_tools
from yuuagents.types import AgentInfo, TaskRequest


def _build_system_prompt(*sections: str) -> str:
    return "\n\n".join(section for section in sections if section)


_DOCKER_TOOL_NAMES = frozenset({
    "execute_bash",
    "read_file",
    "edit_file",
    "delete_file",
})

_WEB_TOOL_NAMES = frozenset({"web_search"})


@define
class AgentManager:
    config: Config
    docker: Any
    persistence: Persistence = field(factory=EphemeralPersistence)
    _host: TaskHost | None = field(default=None, init=False)

    async def setup(self) -> None:
        await self.docker.start()
        host = TaskHost(
            persistence=self.persistence,
            basin=Basin(),
            build_root=self._build_root_agent,
            restore_root=self._restore_root_agent,
        )
        await host.start()
        self._host = host
        if self.config.snapshot.restore_on_start:
            await host.restore_unfinished()

    async def stop(self) -> None:
        if self._host is not None:
            await self._host.stop()
        await self.docker.stop()

    async def submit(self, req: TaskRequest, *, delegate_depth: int = 0) -> str:
        assert self._host is not None, "AgentManager.setup() not called"
        return await self._host.submit(req, delegate_depth=delegate_depth)

    async def list_agents(self) -> list[AgentInfo]:
        assert self._host is not None, "AgentManager.setup() not called"
        return await self._host.list_tasks()

    async def status(self, task_id: str) -> AgentInfo:
        assert self._host is not None, "AgentManager.setup() not called"
        return await self._host.status(task_id)

    async def history(self, task_id: str) -> list[Any]:
        assert self._host is not None, "AgentManager.setup() not called"
        return await self._host.history(task_id)

    async def respond(self, task_id: str, message: yuullm.Message) -> None:
        assert self._host is not None, "AgentManager.setup() not called"
        await self._host.send(task_id, message)

    async def cancel_task(self, task_id: str) -> None:
        assert self._host is not None, "AgentManager.setup() not called"
        await self._host.cancel(task_id)

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
                provider = provider_cls(
                    api_key=api_key,
                    base_url=provider_cfg.base_url or None,
                    organization=provider_cfg.organization or None,
                    provider_name=provider_name,
                )
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

    def _needs_web(self, tool_names: list[str]) -> bool:
        return any(name in _WEB_TOOL_NAMES for name in tool_names)

    def _make_spawn_agent(
        self,
        *,
        task_id: str,
        workdir: str,
        docker_capability: DockerCapability | None,
    ):
        async def _spawn(
            parent_flow: Flow[Any, Any],
            agent: str,
            input: AgentInput,
            tools: list[str] | None,
            delegate_depth: int,
        ) -> Agent:
            assert self._host is not None, "AgentManager.setup() not called"
            req = TaskRequest(agent=agent, input=input, tools=tools or [])
            built = await self._build_agent(
                task_id=task_id,
                req=req,
                delegate_depth=delegate_depth,
                workdir=workdir,
                docker_capability=docker_capability,
            )
            return await self._host.start_child_agent(
                parent_flow=parent_flow,
                agent=built.agent,
            )

        return _spawn

    async def _build_agent(
        self,
        *,
        task_id: str,
        req: TaskRequest,
        delegate_depth: int,
        workdir: str = "",
        docker_capability: DockerCapability | None = None,
        system_prompt_override: str | None = None,
        initial_messages: list[yuullm.Message] | None = None,
        conversation_id: UUID | None = None,
        flow_id: str | None = None,
    ) -> BuiltRoot:
        assert self._host is not None, "AgentManager.setup() not called"
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

        ctx = AgentContext(
            task_id=task_id,
            agent_id=agent_id,
            workdir=workdir or (
                self.docker.workdir if docker_capability is not None else str(Path.cwd())
            ),
            capabilities=AgentCapabilities(
                docker=docker_capability,
                web=(
                    WebCapability(
                        api_key=os.environ.get(self.config.tavily.api_key_env, ""),
                    )
                    if self._needs_web(tool_names)
                    else None
                ),
                basin=self._host.basin,
                spawn_agent=self._make_spawn_agent(
                    task_id=task_id,
                    workdir=workdir or (
                        self.docker.workdir if docker_capability is not None else str(Path.cwd())
                    ),
                    docker_capability=docker_capability,
                ),
            ),
            delegate_depth=delegate_depth,
        )
        agent = Agent(
            config=config,
            ctx=ctx,
            flow_id=flow_id,
            conversation_id=conversation_id,
            initial_messages=list(initial_messages or []),
        )
        agent.start(req.input if initial_messages is None else None)
        return BuiltRoot(
            agent=agent,
            input=req.input,
            created_at=datetime.now(timezone.utc),
            system_prompt=system_prompt,
            model=config.llm.default_model,
            tools=tool_names,
            docker_container=(
                docker_capability.container_id
                if docker_capability is not None
                else ""
            ),
        )

    async def _build_root_agent(
        self,
        task_id: str,
        req: TaskRequest,
        delegate_depth: int,
    ) -> BuiltRoot:
        return await self._build_agent(
            task_id=task_id,
            req=req,
            delegate_depth=delegate_depth,
            flow_id=task_id,
        )

    async def _restore_root_agent(self, restored: RestoredTask) -> BuiltRoot:
        if restored.state is None:
            raise ValueError("restored task requires snapshot state")
        req = TaskRequest(
            agent=restored.agent_id,
            persona="",
            input=restored.input,
            tools=list(restored.tools),
            model=restored.model,
            container=restored.docker_container,
            image="",
        )
        built = await self._build_agent(
            task_id=restored.task_id,
            req=req,
            delegate_depth=0,
            workdir=self.docker.workdir if restored.docker_container else str(Path.cwd()),
            docker_capability=(
                DockerCapability(
                    executor=self.docker,
                    container_id=restored.docker_container,
                )
                if restored.docker_container
                else None
            ),
            system_prompt_override=restored.system_prompt,
            initial_messages=list(restored.state.messages),
            conversation_id=(
                UUID(restored.state.conversation_id)
                if restored.state.conversation_id
                else None
            ),
            flow_id=restored.task_id,
        )
        built.agent.rounds = restored.state.rounds
        built.agent.total_usage = restored.state.total_usage
        built.agent.total_cost_usd = restored.state.total_cost_usd
        return built
