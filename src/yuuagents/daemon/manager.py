"""AgentManager — multi-agent lifecycle management."""

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

from yuuagents.agent import Agent, AgentConfig, SimplePromptBuilder
from yuuagents.config import Config, ProviderConfig
from yuuagents.context import AgentContext
from yuuagents.daemon.docker import DOCKER_SYSTEM_PROMPT, DockerManager
from yuuagents.loop import run as run_agent
from yuuagents.persistence import TaskPersistence, TaskRecorder, TaskWriter
from yuuagents.skills import discovery
from yuuagents.tools import BUILTIN_TOOLS
from yuuagents.types import AgentInfo, AgentStatus, ErrorInfo, SkillInfo, TaskRequest


@define
class AgentManager:
    """Owns all running agents and their asyncio tasks."""

    config: Config
    docker: DockerManager
    db_url: str = ""
    _agents: dict[str, Agent] = field(factory=dict)
    _contexts: dict[str, AgentContext] = field(factory=dict)
    _tasks: dict[str, asyncio.Task] = field(factory=dict)
    _skills: list[SkillInfo] = field(factory=list)
    _persistence: TaskPersistence | None = None
    _writer: TaskWriter | None = None
    _recorders: dict[str, TaskRecorder] = field(factory=dict)

    async def start(self) -> None:
        await self.docker.start()
        self._skills = discovery.scan(self.config.skills.paths)
        db_url = self.db_url or self.config.db_url
        self._persistence = TaskPersistence(db_url=db_url)
        await self._persistence.start()
        self._writer = TaskWriter(self._persistence)
        await self._writer.start()
        await self._restore_unfinished()

    async def stop(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
        self._agents.clear()
        self._contexts.clear()
        self._recorders.clear()
        if self._writer is not None:
            await self._writer.stop()
            self._writer = None
        if self._persistence is not None:
            await self._persistence.stop()
            self._persistence = None
        await self.docker.stop()

    # -- agent lifecycle --

    async def submit(self, req: TaskRequest) -> str:
        task_id = uuid4().hex
        agent_id = req.agent

        # Resolve agent entry by name (e.g. "main", "researcher")
        agent_entry = self.config.agents.get(agent_id)

        llm_client = self._make_llm(
            agent_name=agent_id,
            model_override=req.model,
        )
        tool_names = req.tools or self._default_tools(agent_id)
        tool_objs = [BUILTIN_TOOLS[n] for n in tool_names if n in BUILTIN_TOOLS]
        manager = yt.ToolManager(tool_objs)

        # persona: explicit override > agent config > fallback to agent name
        persona_text = (
            req.persona
            or (agent_entry.persona if agent_entry and agent_entry.persona else "")
            or agent_id
        )
        skills_xml = self._resolve_skills(
            req.skills if req.skills else (agent_entry.skills if agent_entry else [])
        )

        container_id = await self.docker.resolve(
            task_id=task_id,
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
        system_prompt = prompt_builder.build()

        config = AgentConfig(
            task_id=task_id,
            agent_id=agent_id,
            persona=persona_text,
            tools=manager,
            llm=llm_client,
            prompt_builder=prompt_builder,
        )
        agent = Agent(config=config)

        ctx = AgentContext(
            task_id=task_id,
            agent_id=agent_id,
            workdir=self.docker.workdir,
            docker_container=container_id,
            docker=self.docker,
            state=agent.state,
            tavily_api_key=os.environ.get(self.config.tavily.api_key_env, ""),
        )

        if self._persistence is not None:
            await self._persistence.create_task(
                task_id=task_id,
                agent_id=agent_id,
                persona=persona_text,
                task=req.task,
                system_prompt=system_prompt,
                model=llm_client.default_model,
                tools=tool_names,
                docker_container=container_id,
                created_at=agent.created_at,
            )

        self._agents[task_id] = agent
        self._contexts[task_id] = ctx
        task = asyncio.create_task(self._run(agent, req.task, ctx, resume=False))
        self._tasks[task_id] = task
        return task_id

    async def _run(
        self, agent: Agent, task: str, ctx: AgentContext, *, resume: bool
    ) -> None:
        recorder = self._get_recorder(agent.task_id)
        try:
            await run_agent(agent, task, ctx, recorder=recorder, resume=resume)
        except asyncio.CancelledError:
            agent.status = AgentStatus.CANCELLED
            if self._writer is not None:
                await self._writer.flush()
            await self._write_terminal(agent.task_id, AgentStatus.CANCELLED, None)
        except Exception as exc:
            logger.exception("Task {} ({}) failed", agent.task_id, agent.agent_id)
            agent.fail(exc)
            err = agent.error
            err_json = msgspec.json.encode(err) if err is not None else None
            if self._writer is not None:
                await self._writer.flush()
            await self._write_terminal(agent.task_id, AgentStatus.ERROR, err_json)
        else:
            if agent.status == AgentStatus.DONE:
                if self._writer is not None:
                    await self._writer.flush()
                await self._write_terminal(agent.task_id, AgentStatus.DONE, None)

    async def list_agents(self) -> list[AgentInfo]:
        if self._persistence is None:
            return [self._info(a) for a in self._agents.values()]
        infos = await self._persistence.list_tasks()
        by_id = {i.task_id: i for i in infos}
        for task_id, agent in self._agents.items():
            by_id[task_id] = self._info(agent)
        return list(by_id.values())

    async def status(self, task_id: str) -> AgentInfo:
        agent = self._agents.get(task_id)
        if agent is not None:
            return self._info(agent)
        if self._persistence is None:
            raise KeyError(task_id)
        row = await self._persistence.get_task_row(task_id)
        if row is None:
            raise KeyError(task_id)
        err: ErrorInfo | None = None
        if row.error_json is not None:
            err = msgspec.json.decode(row.error_json, type=ErrorInfo)
        pending_prompt = ""
        if row.status == AgentStatus.BLOCKED_ON_INPUT.value:
            pending_prompt = await self._persistence.pending_input_prompt(task_id) or ""
        return AgentInfo(
            task_id=row.task_id,
            agent_id=row.agent_id,
            persona=row.persona[:80],
            task=row.task,
            status=row.status,
            created_at=row.created_at.isoformat(),
            last_assistant_message="",
            pending_input_prompt=pending_prompt,
            steps=row.head_turn,
            total_tokens=0,
            total_cost_usd=0.0,
            error=err,
        )

    async def history(self, task_id: str) -> list[Any]:
        agent = self._agents.get(task_id)
        if agent is not None:
            return agent.history
        if self._persistence is None:
            raise KeyError(task_id)
        return await self._persistence.load_history(task_id)

    async def respond(self, task_id: str, content: str) -> None:
        if task_id not in self._agents:
            await self._load_task(task_id)

        agent = self._agents[task_id]
        ctx = self._contexts[task_id]

        if agent.status in (AgentStatus.DONE, AgentStatus.CANCELLED, AgentStatus.ERROR):
            while not ctx.input_queue.empty():
                ctx.input_queue.get_nowait()

            agent.state.error = None
            agent.state.pending_input_prompt = ""
            agent.status = AgentStatus.RUNNING
            agent.steps += 1

            msg = yuullm.user(content)
            agent.history.append(msg)

            recorder = self._get_recorder(task_id)
            if recorder is not None:
                await recorder.record_user(turn=agent.steps, message=msg)

            existing = self._tasks.get(task_id)
            if existing is not None and not existing.done():
                existing.cancel()

            task = asyncio.create_task(self._run(agent, agent.task, ctx, resume=True))
            self._tasks[task_id] = task
            return

        await ctx.input_queue.put(content)

    async def cancel(self, task_id: str) -> None:
        if task_id not in self._agents:
            raise KeyError(task_id)

        task = self._tasks.get(task_id)
        if task and not task.done():
            task.cancel()
        agent = self._agents.get(task_id)
        if agent:
            agent.status = AgentStatus.CANCELLED
        await self._write_terminal(task_id, AgentStatus.CANCELLED, None)

    def skills(self) -> list[SkillInfo]:
        return list(self._skills)

    def rescan_skills(self) -> list[SkillInfo]:
        self._skills = discovery.scan(self.config.skills.paths)
        return self._skills

    def reload_config(self, new_config: Config) -> None:
        """Hot-reload configuration without restarting daemon.

        Updates the in-memory config and rescans skills if paths changed.
        """
        old_skill_paths = self.config.skills.paths
        self.config = new_config

        # Rescan skills if paths changed
        if old_skill_paths != new_config.skills.paths:
            self._skills = discovery.scan(self.config.skills.paths)

    # -- private helpers --
    def _get_recorder(self, task_id: str) -> TaskRecorder | None:
        if self._writer is None:
            return None
        existing = self._recorders.get(task_id)
        if existing is not None:
            return existing
        rec = TaskRecorder(task_id=task_id, writer=self._writer)
        self._recorders[task_id] = rec
        return rec

    async def _write_terminal(
        self, task_id: str, status: AgentStatus, error_json: bytes | None
    ) -> None:
        if self._persistence is None:
            return
        if status in (AgentStatus.ERROR, AgentStatus.CANCELLED, AgentStatus.DONE):
            await self._persistence.update_task_terminal(
                task_id=task_id,
                status=status,
                error_json=error_json,
            )

    async def _restore_unfinished(self) -> None:
        if self._persistence is None:
            return

        restored = await self._persistence.load_unfinished()
        recovered_any = False
        for t in restored:
            if t.status == AgentStatus.BLOCKED_ON_INPUT:
                recovered_any = (
                    recovered_any
                    or await self._persistence.recover_pending_tools(t.task_id)
                )
        if recovered_any:
            restored = await self._persistence.load_unfinished()

        for t in restored:
            llm_client = self._make_llm(agent_name=t.agent_id, model_override=t.model)
            tool_names = t.tools or self._default_tools(t.agent_id)
            tool_objs = [BUILTIN_TOOLS[n] for n in tool_names if n in BUILTIN_TOOLS]
            manager = yt.ToolManager(tool_objs)

            prompt_builder = SimplePromptBuilder()
            prompt_builder.add_section(t.system_prompt)

            config = AgentConfig(
                task_id=t.task_id,
                agent_id=t.agent_id,
                persona=t.persona,
                tools=manager,
                llm=llm_client,
                prompt_builder=prompt_builder,
            )
            agent = Agent(config=config)
            agent.state.task = t.task
            agent.state.history = t.history
            agent.state.status = t.status
            agent.state.steps = t.head_turn
            agent.state.created_at = t.created_at
            if self._persistence is not None and t.status == AgentStatus.BLOCKED_ON_INPUT:
                agent.state.pending_input_prompt = (
                    await self._persistence.pending_input_prompt(t.task_id) or ""
                )

            container_id = await self.docker.resolve(
                task_id=t.task_id,
                container=t.docker_container,
            )
            ctx = AgentContext(
                task_id=t.task_id,
                agent_id=t.agent_id,
                workdir=self.docker.workdir,
                docker_container=container_id,
                docker=self.docker,
                state=agent.state,
                tavily_api_key=os.environ.get(self.config.tavily.api_key_env, ""),
            )

            self._agents[t.task_id] = agent
            self._contexts[t.task_id] = ctx
            task = asyncio.create_task(self._run(agent, t.task, ctx, resume=True))
            self._tasks[t.task_id] = task

    def _last_assistant_message(self, agent: Agent) -> str:
        for msg in reversed(agent.history):
            role: str | None = None
            items: list[Any] | None = None

            if isinstance(msg, tuple) and len(msg) == 2:
                role, items = msg

            if role != "assistant" or not isinstance(items, list):
                continue

            text_parts: list[str] = []
            tool_names: list[str] = []
            for item in items:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if (
                    isinstance(item, dict)
                    and item.get("type") == "tool_call"
                    and isinstance(item.get("name"), str)
                ):
                    tool_names.append(item["name"])  # type:ignore

            text = "".join(text_parts).strip()
            if text:
                return text
            if tool_names:
                return "\n".join(f"tool_call: {n}" for n in tool_names)
            return ""
        return ""

    def _info(self, agent: Agent) -> AgentInfo:
        return AgentInfo(
            task_id=agent.task_id,
            agent_id=agent.agent_id,
            persona=agent.persona[:80],
            task=agent.task,
            status=agent.status.value,
            created_at=agent.created_at.isoformat(),
            last_assistant_message=self._last_assistant_message(agent),
            pending_input_prompt=agent.state.pending_input_prompt,
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
        provider_name = (
            agent_entry.provider if agent_entry and agent_entry.provider else ""
        )
        if agent_entry and agent_entry.provider:
            provider_cfg = self.config.providers.get(agent_entry.provider)

        if provider_cfg is None and self.config.providers:
            # Fall back to first provider
            provider_name, provider_cfg = next(iter(self.config.providers.items()))

        if provider_cfg is None:
            # No providers configured at all — use bare defaults
            provider_cfg = ProviderConfig()
            provider_name = "openai"

        # Determine model
        model = (
            model_override
            or (agent_entry.model if agent_entry else "")
            or provider_cfg.default_model
        )

        # Build yuullm Provider
        api_key = os.environ.get(provider_cfg.api_key_env, "")
        provider: yuullm.Provider

        match provider_cfg.api_type:
            case "anthropic-messages":
                kwargs: dict[str, Any] = {
                    "api_key": api_key,
                    "provider_name": provider_name,
                }
                if provider_cfg.base_url:
                    kwargs["base_url"] = provider_cfg.base_url
                provider = yuullm.providers.AnthropicMessagesProvider(**kwargs)
            case "openai-chat-completion":
                kwargs = {"api_key": api_key, "provider_name": provider_name}
                if provider_cfg.base_url:
                    kwargs["base_url"] = provider_cfg.base_url
                if provider_cfg.organization:
                    kwargs["organization"] = provider_cfg.organization
                provider = yuullm.providers.OpenAIChatCompletionProvider(**kwargs)
            case "openai-responses":
                provider_cls = getattr(
                    yuullm.providers, "OpenAIResponsesProvider", None
                )
                if provider_cls is None:
                    raise RuntimeError(
                        "api_type 'openai-responses' requires yuullm.providers.OpenAIResponsesProvider"
                    )
                kwargs = {"api_key": api_key, "provider_name": provider_name}
                if provider_cfg.base_url:
                    kwargs["base_url"] = provider_cfg.base_url
                if provider_cfg.organization:
                    kwargs["organization"] = provider_cfg.organization
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
        return [
            "execute_bash",
            "read_file",
            "write_file",
            "delete_file",
            "user_input",
            "web_search",
        ]

    async def _load_task(self, task_id: str) -> None:
        if self._persistence is None:
            raise KeyError(task_id)
        if task_id in self._agents:
            return

        row = await self._persistence.get_task_row(task_id)
        if row is None:
            raise KeyError(task_id)

        err: ErrorInfo | None = None
        if row.error_json is not None:
            err = msgspec.json.decode(row.error_json, type=ErrorInfo)

        llm_client = self._make_llm(agent_name=row.agent_id, model_override=row.model)
        tool_names = msgspec.json.decode(row.tools_json, type=list[str])
        tool_names = tool_names or self._default_tools(row.agent_id)
        tool_objs = [BUILTIN_TOOLS[n] for n in tool_names if n in BUILTIN_TOOLS]
        manager = yt.ToolManager(tool_objs)

        prompt_builder = SimplePromptBuilder()
        prompt_builder.add_section(row.system_prompt)

        config = AgentConfig(
            task_id=row.task_id,
            agent_id=row.agent_id,
            persona=row.persona,
            tools=manager,
            llm=llm_client,
            prompt_builder=prompt_builder,
        )
        agent = Agent(config=config)
        agent.state.task = row.task
        agent.state.history = await self._persistence.load_history(task_id)
        agent.state.status = AgentStatus(row.status)
        agent.state.steps = row.head_turn
        agent.state.created_at = row.created_at
        agent.state.error = err
        if agent.state.status == AgentStatus.BLOCKED_ON_INPUT:
            agent.state.pending_input_prompt = (
                await self._persistence.pending_input_prompt(task_id) or ""
            )

        container_id = await self.docker.resolve(
            task_id=row.task_id,
            container=row.docker_container,
        )
        ctx = AgentContext(
            task_id=row.task_id,
            agent_id=row.agent_id,
            workdir=self.docker.workdir,
            docker_container=container_id,
            docker=self.docker,
            state=agent.state,
            tavily_api_key=os.environ.get(self.config.tavily.api_key_env, ""),
        )

        self._agents[task_id] = agent
        self._contexts[task_id] = ctx

    def _resolve_skills(self, requested: list[str]) -> str:
        if not requested:
            return ""
        if "*" in requested:
            return discovery.render(self._skills)
        matched = [s for s in self._skills if s.name in requested]
        return discovery.render(matched)
