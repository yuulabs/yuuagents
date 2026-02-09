"""Integration tests for yuuagents — tests real component interactions."""

from __future__ import annotations

from collections.abc import AsyncIterator
import os
import tempfile
from pathlib import Path

import pytest
import yuullm
import yuutools as yt

from yuuagents.agent import Agent, AgentConfig, SimplePromptBuilder
from yuuagents.config import Config, load as load_config
from yuuagents.context import AgentContext
from yuuagents.daemon.docker import DockerManager
from yuuagents.daemon.manager import AgentManager
from yuuagents.loop import run as run_agent
from yuuagents.skills.discovery import scan
from yuuagents.tools import BUILTIN_TOOLS, get
from yuuagents.types import AgentStatus, TaskRequest


@pytest.mark.asyncio
class TestAgentLifecycle:
    """Integration tests for Agent lifecycle."""

    async def test_create_agent_without_llm(self) -> None:
        """Should create agent even without LLM (setup phase)."""
        # Create minimal agent config
        tools = yt.ToolManager([])
        builder = SimplePromptBuilder()
        builder.add_section("You are a test agent")

        # We can't create YLLMClient without API key, but we can test the structure
        config = AgentConfig(
            agent_id="test-agent",
            persona="test",
            tools=tools,
            llm=None,  # type: ignore[arg-type]
            prompt_builder=builder,
        )

        agent = Agent(config=config)
        assert agent.agent_id == "test-agent"
        assert agent.status == AgentStatus.IDLE

        # Setup should work without LLM
        agent.setup("test task")
        assert agent.status == AgentStatus.RUNNING
        assert agent.task == "test task"
        assert len(agent.history) == 2

    async def test_agent_run_records_usage_without_explicit_tracing_setup(self) -> None:
        class FakeLLM:
            default_model = "test-model"

            async def stream(
                self,
                history: yuullm.History,
                tools: list[dict[str, object]] | None = None,
            ) -> tuple[AsyncIterator[yuullm.StreamItem], yuullm.Store]:
                async def _iter() -> AsyncIterator[yuullm.StreamItem]:
                    yield yuullm.Response(item="hi")

                store: yuullm.Store = {
                    "usage": yuullm.Usage(
                        provider="test-provider",
                        model="test-model",
                        request_id="req_1",
                        input_tokens=1,
                        output_tokens=1,
                        cache_read_tokens=0,
                        cache_write_tokens=0,
                        total_tokens=2,
                    ),
                    "cost": yuullm.Cost(
                        input_cost=0.001,
                        output_cost=0.001,
                        total_cost=0.002,
                        source="test",
                    ),
                }
                return _iter(), store

        tools = yt.ToolManager([])
        builder = SimplePromptBuilder().add_section("test persona")
        config = AgentConfig(
            agent_id="12345678123456781234567812345678",
            persona="test",
            tools=tools,
            llm=FakeLLM(),  # type: ignore[arg-type]
            prompt_builder=builder,
        )
        agent = Agent(config=config)
        ctx = AgentContext(
            agent_id="test-agent",
            workdir="/tmp",
            docker_container="dummy",
        )

        await run_agent(agent, "hello", ctx)
        assert agent.status == AgentStatus.DONE
        assert agent.steps == 1
        assert agent.total_tokens == 2
        assert agent.total_cost_usd == 0.002

    async def test_agent_state_transitions(self) -> None:
        """Should transition through states correctly."""
        tools = yt.ToolManager([])
        builder = SimplePromptBuilder()
        builder.add_section("test")

        config = AgentConfig(
            agent_id="test",
            persona="test",
            tools=tools,
            llm=None,  # type: ignore[arg-type]
            prompt_builder=builder,
        )

        agent = Agent(config=config)

        # Initial state
        assert agent.status == AgentStatus.IDLE
        assert not agent.done()

        # After setup
        agent.setup("task")
        assert agent.status == AgentStatus.RUNNING
        assert not agent.done()

        # Done state
        agent.status = AgentStatus.DONE
        assert agent.done()

        # Error state
        agent.status = AgentStatus.ERROR
        assert agent.done()

        # Cancelled state
        agent.status = AgentStatus.CANCELLED
        assert agent.done()

    async def test_agent_steps_and_tokens_tracking(self) -> None:
        """Should track steps and tokens."""
        tools = yt.ToolManager([])
        builder = SimplePromptBuilder()
        builder.add_section("test")

        config = AgentConfig(
            agent_id="test",
            persona="test",
            tools=tools,
            llm=None,  # type: ignore[arg-type]
            prompt_builder=builder,
        )

        agent = Agent(config=config)
        agent.setup("task")

        # Simulate tracking
        agent.steps = 5
        agent.total_tokens = 1000
        agent.total_cost_usd = 0.05

        assert agent.steps == 5
        assert agent.total_tokens == 1000
        assert agent.total_cost_usd == 0.05


class TestToolIntegration:
    """Integration tests for tools."""

    def test_builtin_tools_registry(self) -> None:
        """Should have all expected tools in registry."""
        expected = {
            "execute_bash",
            "read_file",
            "write_file",
            "delete_file",
            "web_search",
        }
        actual = set(BUILTIN_TOOLS.keys())
        assert actual == expected

    def test_get_tools(self) -> None:
        """Should retrieve tools from registry."""
        tools = get(["execute_bash", "read_file"])
        assert len(tools) == 2
        assert all(isinstance(t, yt.Tool) for t in tools)

    def test_get_unknown_tool_raises(self) -> None:
        """Should raise KeyError for unknown tools."""
        with pytest.raises(KeyError) as exc_info:
            get(["unknown_tool"])
        assert "unknown" in str(exc_info.value).lower()

    def test_tool_spec(self) -> None:
        """Tools should have expected spec."""
        from yuuagents.tools.bash import execute_bash

        # Check tool has expected parameters via spec
        spec = execute_bash.spec
        param_names = {p.name for p in spec.params}
        assert "command" in param_names
        assert "timeout" in param_names


class TestSkillDiscovery:
    """Integration tests for skill discovery."""

    def test_scan_empty_directory(self) -> None:
        """Should return empty list for empty directory."""
        with tempfile.TemporaryDirectory() as tmp:
            skills = scan([tmp])
            assert skills == []

    def test_scan_with_skills(self) -> None:
        """Should discover skills in directory."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create a skill
            skill_dir = tmp_path / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: A test skill
---

Content here.
""")

            skills = scan([tmp])
            assert len(skills) == 1
            assert skills[0].name == "my-skill"
            assert skills[0].description == "A test skill"

    def test_scan_nonexistent_path(self) -> None:
        """Should return empty list for nonexistent paths."""
        skills = scan(["/nonexistent/path"])
        assert skills == []


class TestConfigLoading:
    """Integration tests for config loading."""

    def test_load_default_config(self) -> None:
        """Should load default config."""
        config = Config()
        assert config.docker.image == "ubuntu:24.04"

    def test_load_from_yaml(self) -> None:
        """Should load config from YAML file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("""\
docker:
  image: python:3.12
providers:
  openai-default:
    kind: openai
    default_model: gpt-3.5-turbo
""")
            f.flush()

            try:
                config = load_config(f.name)
                assert config.docker.image == "python:3.12"
                assert (
                    config.providers["openai-default"].default_model == "gpt-3.5-turbo"
                )
            finally:
                os.unlink(f.name)

    def test_socket_path_expansion(self) -> None:
        """Should expand user in socket path."""
        config = Config()
        path = config.socket_path
        assert str(path).startswith("/")
        assert not str(path).startswith("~")


class TestContextCreation:
    """Integration tests for AgentContext."""

    def test_create_context(self) -> None:
        """Should create context with required fields."""
        ctx = AgentContext(
            agent_id="test-agent",
            workdir="/root",
            docker_container="container-123",
        )

        assert ctx.agent_id == "test-agent"
        assert ctx.workdir == "/root"
        assert ctx.docker_container == "container-123"
        assert ctx.docker is None
        assert ctx.tavily_api_key == ""

    def test_context_queue(self) -> None:
        """Should have working input queue."""
        ctx = AgentContext(
            agent_id="test",
            workdir="/root",
            docker_container="c1",
        )

        # Test queue operations
        ctx.input_queue.put_nowait("message 1")
        ctx.input_queue.put_nowait("message 2")

        assert ctx.input_queue.qsize() == 2
        assert ctx.input_queue.get_nowait() == "message 1"


@pytest.mark.asyncio
class TestManagerIntegration:
    """Integration tests for AgentManager — requires Docker."""

    async def test_manager_lifecycle(self) -> None:
        """Should start and stop properly."""
        config = Config()
        docker = DockerManager(image=config.docker.image)
        manager = AgentManager(config, docker)

        try:
            await manager.start()
            assert len(manager.skills()) >= 0  # Should have scanned skills
        finally:
            await manager.stop()

    async def test_submit_agent(self) -> None:
        """Should submit new agent."""
        config = Config()
        docker = DockerManager(image=config.docker.image)
        manager = AgentManager(config, docker)

        try:
            await manager.start()

            req = TaskRequest(
                agent="main",
                task="test task",
                tools=[],
            )

            agent_id = await manager.submit(req)
            assert len(agent_id) > 0

            # Should be in list
            agents = manager.list_agents()
            assert len(agents) == 1
            assert agents[0].agent_id == agent_id
        finally:
            await manager.stop()

    async def test_cancel_agent(self) -> None:
        """Should cancel agent."""
        config = Config()
        docker = DockerManager(image=config.docker.image)
        manager = AgentManager(config, docker)

        try:
            await manager.start()

            req = TaskRequest(
                agent="main",
                task="long running task",
                tools=[],
            )

            agent_id = await manager.submit(req)

            # Cancel immediately
            await manager.cancel(agent_id)

            # Check status
            info = manager.status(agent_id)
            assert info.status == "cancelled"
        finally:
            await manager.stop()


class TestPromptBuilder:
    """Integration tests for SimplePromptBuilder."""

    def test_build_empty(self) -> None:
        """Should build empty prompt."""
        builder = SimplePromptBuilder()
        assert builder.build() == ""

    def test_build_single_section(self) -> None:
        """Should build single section."""
        builder = SimplePromptBuilder()
        builder.add_section("System prompt")
        assert builder.build() == "System prompt"

    def test_build_multiple_sections(self) -> None:
        """Should join multiple sections."""
        builder = SimplePromptBuilder()
        builder.add_section("Section 1")
        builder.add_section("Section 2")
        result = builder.build()
        assert "Section 1" in result
        assert "Section 2" in result
        assert "\n\n" in result

    def test_empty_section_ignored(self) -> None:
        """Should ignore empty sections."""
        builder = SimplePromptBuilder()
        builder.add_section("A")
        builder.add_section("")
        builder.add_section("B")
        assert builder.build() == "A\n\nB"


class TestTaskRequest:
    """Integration tests for TaskRequest."""

    def test_create_minimal(self) -> None:
        """Should create with minimal fields."""
        req = TaskRequest(task="test")
        assert req.agent == "main"
        assert req.persona == ""
        assert req.task == "test"
        assert req.tools == []
        assert req.skills == []

    def test_create_full(self) -> None:
        """Should create with all fields."""
        req = TaskRequest(
            agent="researcher",
            persona="Custom persona",
            task="test",
            tools=["execute_bash"],
            skills=["git-worktree"],
            model="gpt-4",
            container="my-container",
            image="ubuntu:24.04",
        )
        assert req.agent == "researcher"
        assert req.persona == "Custom persona"
        assert req.model == "gpt-4"
        assert req.container == "my-container"

    def test_frozen(self) -> None:
        """Should be immutable."""
        req = TaskRequest(task="test")
        with pytest.raises(AttributeError):
            req.task = "new"  # type: ignore[misc]
