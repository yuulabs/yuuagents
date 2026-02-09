"""Tests for yuuagents.context module."""

from __future__ import annotations

import asyncio

import pytest

from yuuagents.context import AgentContext


class TestAgentContext:
    """Tests for AgentContext dataclass."""

    def test_context_creation(self) -> None:
        """Should create context with all required fields."""
        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/root",
            docker_container="container-abc",
        )
        assert ctx.task_id == "task-123"
        assert ctx.agent_id == "agent-123"
        assert ctx.workdir == "/root"
        assert ctx.docker_container == "container-abc"

    def test_context_defaults(self) -> None:
        """Should have default values for optional fields."""
        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/root",
            docker_container="container-abc",
        )
        assert ctx.docker is None
        assert isinstance(ctx.input_queue, asyncio.Queue)
        assert ctx.tavily_api_key == ""

    def test_context_custom_values(self) -> None:
        """Should accept custom values for all fields."""
        mock_docker = object()
        queue: asyncio.Queue[str] = asyncio.Queue()

        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/home/user",
            docker_container="custom-container",
            docker=mock_docker,
            input_queue=queue,
            tavily_api_key="secret-key",
        )
        assert ctx.workdir == "/home/user"
        assert ctx.docker is mock_docker
        assert ctx.input_queue is queue
        assert ctx.tavily_api_key == "secret-key"

    def test_context_is_mutable(self) -> None:
        """AgentContext should be mutable."""
        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/root",
            docker_container="container-abc",
        )
        ctx.workdir = "/new/path"
        ctx.tavily_api_key = "new-key"
        assert ctx.workdir == "/new/path"
        assert ctx.tavily_api_key == "new-key"

    def test_context_input_queue_operations(self) -> None:
        """Should be able to use input_queue."""
        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/root",
            docker_container="container-abc",
        )

        # Put items in queue
        ctx.input_queue.put_nowait("message 1")
        ctx.input_queue.put_nowait("message 2")

        # Get items from queue
        assert ctx.input_queue.get_nowait() == "message 1"
        assert ctx.input_queue.get_nowait() == "message 2"

    @pytest.mark.asyncio
    async def test_context_input_queue_async(self) -> None:
        """Should work with async queue operations."""
        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/root",
            docker_container="container-abc",
        )

        # Put and get asynchronously
        await ctx.input_queue.put("async message")
        result = await ctx.input_queue.get()
        assert result == "async message"

    def test_context_docker_can_be_any_type(self) -> None:
        """docker field should accept any type."""

        class FakeDocker:
            pass

        fake = FakeDocker()
        ctx = AgentContext(
            task_id="task-123",
            agent_id="agent-123",
            workdir="/root",
            docker_container="container-abc",
            docker=fake,
        )
        assert ctx.docker is fake

    def test_context_missing_required_fields(self) -> None:
        """Should raise when required fields are missing."""
        with pytest.raises(TypeError):
            AgentContext()  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            AgentContext(agent_id="test")  # type: ignore[call-arg]

        with pytest.raises(TypeError):
            AgentContext(agent_id="test", workdir="/root")  # type: ignore[call-arg]

    def test_context_empty_strings(self) -> None:
        """Should accept empty strings."""
        ctx = AgentContext(
            task_id="",
            agent_id="",
            workdir="",
            docker_container="",
        )
        assert ctx.task_id == ""
        assert ctx.agent_id == ""
        assert ctx.workdir == ""
        assert ctx.docker_container == ""

    def test_context_special_characters(self) -> None:
        """Should handle special characters in strings."""
        ctx = AgentContext(
            task_id="task-with-special_chars.123",
            agent_id="agent-with-special_chars.123",
            workdir="/path with spaces/and-dashes",
            docker_container="container_name:test",
        )
        assert ctx.agent_id == "agent-with-special_chars.123"
        assert ctx.workdir == "/path with spaces/and-dashes"
        assert ctx.docker_container == "container_name:test"

    def test_context_queue_is_unique_per_instance(self) -> None:
        """Each context should have its own queue."""
        ctx1 = AgentContext(
            task_id="task-1",
            agent_id="agent-1",
            workdir="/root",
            docker_container="c1",
        )
        ctx2 = AgentContext(
            task_id="task-2",
            agent_id="agent-2",
            workdir="/root",
            docker_container="c2",
        )

        ctx1.input_queue.put_nowait("for agent 1")

        assert ctx1.input_queue.qsize() == 1
        assert ctx2.input_queue.qsize() == 0

        with pytest.raises(asyncio.QueueEmpty):
            ctx2.input_queue.get_nowait()
