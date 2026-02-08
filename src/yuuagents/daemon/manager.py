"""AgentManager — multi-agent lifecycle management."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from loguru import logger

import yuullm
import yuutools as yt
from attrs import define, field

from yuuagents.agent import Agent
from yuuagents.config import Config
from yuuagents.context import AgentContext
from yuuagents.daemon.docker import DOCKER_SYSTEM_PROMPT, DockerManager
from yuuagents.loop import run as run_agent
from yuuagents.skills import discovery
from yuuagents.tools import BUILTIN_TOOLS
from yuuagents.types import AgentInfo, AgentStatus, SkillInfo, TaskRequest


@define
class AgentManager:
    """Owns all running agents and their asyncio tasks."""

    _config: Config
    _docker: DockerManager
    _agents: dict[str, Agent] = field(factory=dict)
    _contexts: dict[str, AgentContext] = field(factory=dict)
    _tasks: dict[str, asyncio.Task] = field(factory=dict)
    _skills: list[SkillInfo] = field(factory=list)

    async def start(self) -> None:
        await self._docker.start()
        self._skills = discovery.scan(self._config.skills.paths)

    async def stop(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        await self._docker.stop()

    # -- agent lifecycle --

    async def submit(self, req: TaskRequest) -> str:
        agent_id = uuid4().hex
        llm_client = self._make_llm(req.model)
        tool_names = req.tools or self._default_tools(req.persona)
        tool_objs = [BUILTIN_TOOLS[n] for n in tool_names if n in BUILTIN_TOOLS]
        manager = yt.ToolManager(tool_objs)

        persona_text = self._resolve_persona(req.persona)
        skills_xml = self._resolve_skills(req.skills)

        container_id = await self._docker.resolve(
            container=req.container,
            image=req.image,
        )

        agent = Agent(
            agent_id=agent_id,
            persona=persona_text,
            tools=manager,
            llm=llm_client,
            skills_xml=skills_xml,
            docker_prompt=DOCKER_SYSTEM_PROMPT,
        )

        ctx = AgentContext(
            agent_id=agent_id,
            workdir="/root",
            docker_container=container_id,
            docker=self._docker,
            tavily_api_key=os.environ.get(self._config.tavily.api_key_env, ""),
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
        except Exception:
            logger.exception("Agent {} failed", agent.agent_id)
            agent.status = AgentStatus.ERROR

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
        self._skills = discovery.scan(self._config.skills.paths)
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
        )

    def _make_llm(self, model_override: str) -> yuullm.YLLMClient:
        provider_name = self._config.llm.provider
        api_key = os.environ.get(self._config.llm.api_key_env, "")
        model = model_override or self._config.llm.default_model

        if provider_name == "anthropic":
            provider = yuullm.providers.AnthropicProvider(api_key=api_key)
        else:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if self._config.llm.base_url:
                kwargs["base_url"] = self._config.llm.base_url
            provider = yuullm.providers.OpenAIProvider(**kwargs)

        return yuullm.YLLMClient(provider=provider, default_model=model)

    def _resolve_persona(self, persona: str) -> str:
        cfg = self._config.personas.get(persona)
        if cfg:
            return cfg.system_prompt
        return persona

    def _default_tools(self, persona: str) -> list[str]:
        cfg = self._config.personas.get(persona)
        if cfg and cfg.tools:
            return cfg.tools
        return ["execute_bash", "read_file", "write_file", "delete_file", "web_search"]

    def _resolve_skills(self, requested: list[str]) -> str:
        if not requested:
            return ""
        if "*" in requested:
            return discovery.render(self._skills)
        matched = [s for s in self._skills if s.name in requested]
        return discovery.render(matched)
