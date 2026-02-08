"""AgentManager — multi-agent lifecycle management."""

from __future__ import annotations

import asyncio
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger

import yuullm
import yuutools as yt
from attrs import define, field

from yuuagents.agent import Agent, AgentConfig, SimplePromptBuilder
from yuuagents.config import Config, ProviderConfig
from yuuagents.context import AgentContext
from yuuagents.daemon.docker import DOCKER_SYSTEM_PROMPT, DockerManager
from yuuagents.loop import run as run_agent
from yuuagents.skills import discovery
from yuuagents.tools import BUILTIN_TOOLS
from yuuagents.types import AgentInfo, AgentStatus, SkillInfo, TaskRequest


@define
class AgentManager:
    """Owns all running agents and their asyncio tasks."""

    config: Config
    docker: DockerManager
    _agents: dict[str, Agent] = field(factory=dict)
    _contexts: dict[str, AgentContext] = field(factory=dict)
    _tasks: dict[str, asyncio.Task] = field(factory=dict)
    _skills: list[SkillInfo] = field(factory=list)

    async def start(self) -> None:
        await self.docker.start()
        self._skills = discovery.scan(self.config.skills.paths)

    async def stop(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        await self.docker.stop()

    # -- agent lifecycle --

    async def submit(self, req: TaskRequest) -> str:
        agent_id = uuid4().hex

        # Resolve agent entry by name (e.g. "main", "researcher")
        agent_entry = self.config.agents.get(req.agent)

        llm_client = self._make_llm(
            agent_name=req.agent,
            model_override=req.model,
        )
        tool_names = req.tools or self._default_tools(req.agent)
        tool_objs = [BUILTIN_TOOLS[n] for n in tool_names if n in BUILTIN_TOOLS]
        manager = yt.ToolManager(tool_objs)

        # persona: explicit override > agent config > fallback to agent name
        persona_text = (
            req.persona
            or (agent_entry.persona if agent_entry and agent_entry.persona else "")
            or req.agent
        )
        skills_xml = self._resolve_skills(
            req.skills
            if req.skills
            else (agent_entry.skills if agent_entry else [])
        )

        container_id = await self.docker.resolve(
            container=req.container,
            image=req.image,
        )

        # Build prompt using the builder pattern
        prompt_builder = SimplePromptBuilder()
        prompt_builder.add_section(persona_text)
        if DOCKER_SYSTEM_PROMPT:
            prompt_builder.add_section(DOCKER_SYSTEM_PROMPT)
        if skills_xml:
            prompt_builder.add_section(skills_xml)

        config = AgentConfig(
            agent_id=agent_id,
            persona=persona_text,
            tools=manager,
            llm=llm_client,
            prompt_builder=prompt_builder,
        )
        agent = Agent(config=config)

        ctx = AgentContext(
            agent_id=agent_id,
            workdir="/root",
            docker_container=container_id,
            docker=self.docker,
            tavily_api_key=os.environ.get(self.config.tavily.api_key_env, ""),
        )

        self._agents[agent_id] = agent
        self._contexts[agent_id] = ctx
        task = asyncio.create_task(self._run(agent, req.task, ctx))
        self._tasks[agent_id] = task
        return agent_id

    async def _run(self, agent: Agent, task: str, ctx: AgentContext) -> None:
        try:
            await run_agent(agent, task, ctx)
        except asyncio.CancelledError:
            agent.status = AgentStatus.CANCELLED
        except Exception as exc:
            logger.exception("Agent {} failed", agent.agent_id)
            agent.fail(exc)

    def list_agents(self) -> list[AgentInfo]:
        return [self._info(a) for a in self._agents.values()]

    def status(self, agent_id: str) -> AgentInfo:
        return self._info(self._agents[agent_id])

    def history(self, agent_id: str) -> list[Any]:
        return self._agents[agent_id].history

    async def respond(self, agent_id: str, content: str) -> None:
        ctx = self._contexts[agent_id]
        await ctx.input_queue.put(content)

    async def cancel(self, agent_id: str) -> None:
        task = self._tasks.get(agent_id)
        if task and not task.done():
            task.cancel()
        agent = self._agents.get(agent_id)
        if agent:
            agent.status = AgentStatus.CANCELLED

    def skills(self) -> list[SkillInfo]:
        return list(self._skills)

    def rescan_skills(self) -> list[SkillInfo]:
        self._skills = discovery.scan(self.config.skills.paths)
        return self._skills

    # -- private helpers --

    def _info(self, agent: Agent) -> AgentInfo:
        return AgentInfo(
            agent_id=agent.agent_id,
            persona=agent.persona[:80],
            task=agent.task,
            status=agent.status.value,
            created_at=agent.created_at.isoformat(),
            steps=agent.steps,
            total_tokens=agent.total_tokens,
            total_cost_usd=agent.total_cost_usd,
            error=agent.error,
        )

    def _make_llm(
        self,
        agent_name: str,
        model_override: str = "",
    ) -> yuullm.YLLMClient:
        """Build a YLLMClient from the new multi-provider config.

        Resolution order:
        1. Look up ``config.agents[agent_name]`` for provider reference + model.
        2. Fall back to the first provider in ``config.providers`` if no match.
        3. ``model_override`` always wins over configured model.
        """
        agent_entry = self.config.agents.get(agent_name)

        # Resolve provider config
        provider_cfg: ProviderConfig | None = None
        if agent_entry and agent_entry.provider:
            provider_cfg = self.config.providers.get(agent_entry.provider)

        if provider_cfg is None and self.config.providers:
            # Fall back to first provider
            provider_cfg = next(iter(self.config.providers.values()))

        if provider_cfg is None:
            # No providers configured at all — use bare defaults
            provider_cfg = ProviderConfig()

        # Determine model
        model = (
            model_override
            or (agent_entry.model if agent_entry else "")
            or provider_cfg.default_model
        )

        # Build yuullm Provider
        api_key = os.environ.get(provider_cfg.api_key_env, "")
        provider: yuullm.Provider

        if provider_cfg.kind == "anthropic":
            kwargs: dict[str, Any] = {"api_key": api_key}
            if provider_cfg.base_url:
                kwargs["base_url"] = provider_cfg.base_url
            provider = yuullm.providers.AnthropicMessagesProvider(**kwargs)
        else:
            kwargs = {"api_key": api_key}
            if provider_cfg.base_url:
                kwargs["base_url"] = provider_cfg.base_url
            if provider_cfg.organization:
                kwargs["organization"] = provider_cfg.organization
            provider = yuullm.providers.OpenAIChatCompletionProvider(**kwargs)

        # Build PriceCalculator from inline pricing
        price_calc: yuullm.PriceCalculator | None = None
        if provider_cfg.pricing:
            price_calc = self._build_price_calculator(provider_cfg)

        return yuullm.YLLMClient(
            provider=provider,
            default_model=model,
            price_calculator=price_calc,
        )

    @staticmethod
    def _build_price_calculator(
        provider_cfg: ProviderConfig,
    ) -> yuullm.PriceCalculator:
        """Convert inline pricing entries to a temporary YAML file and
        build a PriceCalculator from it.

        The yuullm PriceCalculator expects YAML in this format::

            - provider: <name>
              models:
                - id: <model>
                  prices:
                    input_mtok: ...
                    output_mtok: ...
        """
        import yaml

        provider_name = provider_cfg.kind
        models = []
        for entry in provider_cfg.pricing:
            models.append(
                {
                    "id": entry.model,
                    "prices": {
                        "input_mtok": entry.input_mtok,
                        "output_mtok": entry.output_mtok,
                        "cache_read_mtok": entry.cache_read_mtok,
                        "cache_write_mtok": entry.cache_write_mtok,
                    },
                }
            )

        data = [{"provider": provider_name, "models": models}]

        # Write to a temp file and load
        tmp = Path(tempfile.mktemp(suffix=".yaml"))
        try:
            tmp.write_text(yaml.dump(data), encoding="utf-8")
            return yuullm.PriceCalculator(yaml_path=tmp)
        finally:
            tmp.unlink(missing_ok=True)

    def _default_tools(self, agent_name: str) -> list[str]:
        entry = self.config.agents.get(agent_name)
        if entry and entry.tools:
            return entry.tools
        return ["execute_bash", "read_file", "write_file", "delete_file", "web_search"]

    def _resolve_skills(self, requested: list[str]) -> str:
        if not requested:
            return ""
        if "*" in requested:
            return discovery.render(self._skills)
        matched = [s for s in self._skills if s.name in requested]
        return discovery.render(matched)
