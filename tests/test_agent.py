"""Tests for yuuagents.agent module."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import yuullm
import yuutools as yt

from yuuagents.agent import (
    Agent,
    AgentConfig,
    AgentState,
    SimplePromptBuilder,
)
from yuuagents.types import AgentStatus, ErrorInfo


class TestSimplePromptBuilder:
    """Tests for SimplePromptBuilder."""

    def test_empty_builder(self) -> None:
        """Empty builder should return empty string."""
        builder = SimplePromptBuilder()
        assert builder.build() == ""

    def test_single_section(self) -> None:
        """Builder with one section."""
        builder = SimplePromptBuilder()
        builder.add_section("System prompt")
        assert builder.build() == "System prompt"

    def test_multiple_sections(self) -> None:
        """Builder with multiple sections joined by double newline."""
        builder = SimplePromptBuilder()
        builder.add_section("Section 1")
        builder.add_section("Section 2")
        builder.add_section("Section 3")
        assert builder.build() == "Section 1\n\nSection 2\n\nSection 3"

    def test_empty_section_ignored(self) -> None:
        """Empty sections should be ignored."""
        builder = SimplePromptBuilder()
        builder.add_section("Section 1")
        builder.add_section("")
        builder.add_section("Section 2")
        assert builder.build() == "Section 1\n\nSection 2"

    def test_whitespace_section_not_ignored(self) -> None:
        """Whitespace-only sections are not empty strings."""
        builder = SimplePromptBuilder()
        builder.add_section("Section 1")
        builder.add_section("   ")
        builder.add_section("Section 2")
        result = builder.build()
        assert "   " in result

    def test_fluent_interface(self) -> None:
        """add_section should return self for chaining."""
        builder = SimplePromptBuilder()
        result = builder.add_section("A").add_section("B").add_section("C")
        assert result is builder
        assert builder.build() == "A\n\nB\n\nC"

    def test_section_with_special_characters(self) -> None:
        """Builder should handle special characters."""
        builder = SimplePromptBuilder()
        section = "Line 1\nLine 2\n<xml>tag</xml>"
        builder.add_section(section)
        assert builder.build() == section

    def test_builder_reuse(self) -> None:
        """Builder should be reusable after build."""
        builder = SimplePromptBuilder()
        builder.add_section("First")
        first = builder.build()
        builder.add_section("Second")
        second = builder.build()
        assert first == "First"
        assert second == "First\n\nSecond"


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_config_creation(self) -> None:
        """Should create AgentConfig with all fields."""
        mock_tools = MagicMock(spec=yt.ToolManager)
        mock_llm = MagicMock(spec=yuullm.YLLMClient)
        mock_builder = MagicMock(spec=SimplePromptBuilder)

        config = AgentConfig(
            agent_id="test-id",
            persona="You are a coder",
            tools=mock_tools,
            llm=mock_llm,
            prompt_builder=mock_builder,
        )
        assert config.agent_id == "test-id"
        assert config.persona == "You are a coder"
        assert config.tools is mock_tools
        assert config.llm is mock_llm
        assert config.prompt_builder is mock_builder

    def test_config_is_frozen(self) -> None:
        """AgentConfig should be immutable."""
        config = AgentConfig(
            agent_id="test-id",
            persona="coder",
            tools=MagicMock(),
            llm=MagicMock(),
            prompt_builder=MagicMock(),
        )
        with pytest.raises(AttributeError):
            config.agent_id = "new-id"  # type: ignore[misc]

    def test_config_hashable(self) -> None:
        """Frozen config should be hashable."""
        config = AgentConfig(
            agent_id="test-id",
            persona="coder",
            tools=MagicMock(),
            llm=MagicMock(),
            prompt_builder=MagicMock(),
        )
        # Should not raise
        hash(config)

    def test_config_missing_fields(self) -> None:
        """Should raise when required fields missing."""
        with pytest.raises(TypeError):
            AgentConfig()  # type: ignore[call-arg]


class TestAgentState:
    """Tests for AgentState dataclass."""

    def test_default_state(self) -> None:
        """Should create state with default values."""
        state = AgentState()
        assert state.task == ""
        assert state.history == []
        assert state.status == AgentStatus.IDLE
        assert state.error is None
        assert state.steps == 0
        assert state.total_tokens == 0
        assert state.total_cost_usd == 0.0
        assert isinstance(state.created_at, datetime)

    def test_custom_state(self) -> None:
        """Should create state with custom values."""
        error = ErrorInfo(
            message="error",
            error_type="Exception",
            timestamp=datetime.now(timezone.utc),
        )
        state = AgentState(
            task="my task",
            history=[yuullm.user("hello")],
            status=AgentStatus.RUNNING,
            error=error,
            steps=5,
            total_tokens=100,
            total_cost_usd=0.01,
        )
        assert state.task == "my task"
        assert len(state.history) == 1
        assert state.status == AgentStatus.RUNNING
        assert state.error == error
        assert state.steps == 5
        assert state.total_tokens == 100
        assert state.total_cost_usd == 0.01

    def test_state_is_mutable(self) -> None:
        """AgentState should be mutable."""
        state = AgentState()
        state.task = "new task"
        state.status = AgentStatus.DONE
        state.steps = 10
        assert state.task == "new task"
        assert state.status == AgentStatus.DONE
        assert state.steps == 10

    def test_history_append(self) -> None:
        """Should be able to append to history."""
        state = AgentState()
        state.history.append(yuullm.user("message 1"))
        state.history.append(yuullm.assistant("response"))
        assert len(state.history) == 2

    def test_state_accepts_positional_args(self) -> None:
        """attrs classes accept positional args by default."""
        state = AgentState("task", [yuullm.user("test")], AgentStatus.RUNNING)
        assert state.task == "task"
        assert len(state.history) == 1
        assert state.status == AgentStatus.RUNNING


class TestAgent:
    """Tests for Agent class."""

    @pytest.fixture
    def mock_config(self) -> AgentConfig:
        """Create a mock config for testing."""
        builder = SimplePromptBuilder()
        builder.add_section("You are a coder")
        return AgentConfig(
            agent_id="test-agent-123",
            persona="You are a coder",
            tools=MagicMock(spec=yt.ToolManager),
            llm=MagicMock(spec=yuullm.YLLMClient),
            prompt_builder=builder,
        )

    def test_agent_creation(self, mock_config: AgentConfig) -> None:
        """Should create agent with config."""
        agent = Agent(config=mock_config)
        assert agent.config is mock_config
        assert isinstance(agent.state, AgentState)

    def test_agent_default_state(self, mock_config: AgentConfig) -> None:
        """Agent should have default state if not provided."""
        agent = Agent(config=mock_config)
        assert agent.state.task == ""
        assert agent.state.status == AgentStatus.IDLE

    def test_agent_property_proxies(self, mock_config: AgentConfig) -> None:
        """Agent properties should proxy to config/state."""
        agent = Agent(config=mock_config)

        # Config proxies
        assert agent.agent_id == "test-agent-123"
        assert agent.persona == "You are a coder"
        assert agent.tools is mock_config.tools
        assert agent.llm is mock_config.llm

        # State proxies
        assert agent.task == ""
        assert agent.history == []
        assert agent.status == AgentStatus.IDLE
        assert agent.steps == 0
        assert agent.total_tokens == 0
        assert agent.total_cost_usd == 0.0

    def test_agent_status_setter(self, mock_config: AgentConfig) -> None:
        """Should be able to set status through agent."""
        agent = Agent(config=mock_config)
        agent.status = AgentStatus.RUNNING
        assert agent.state.status == AgentStatus.RUNNING
        assert agent.status == AgentStatus.RUNNING

    def test_agent_steps_setter(self, mock_config: AgentConfig) -> None:
        """Should be able to set steps through agent."""
        agent = Agent(config=mock_config)
        agent.steps = 5
        assert agent.state.steps == 5
        assert agent.steps == 5

    def test_agent_tokens_setter(self, mock_config: AgentConfig) -> None:
        """Should be able to set tokens through agent."""
        agent = Agent(config=mock_config)
        agent.total_tokens = 100
        assert agent.state.total_tokens == 100

    def test_agent_cost_setter(self, mock_config: AgentConfig) -> None:
        """Should be able to set cost through agent."""
        agent = Agent(config=mock_config)
        agent.total_cost_usd = 0.05
        assert agent.state.total_cost_usd == 0.05

    def test_agent_full_system_prompt(self, mock_config: AgentConfig) -> None:
        """Should build full system prompt from builder."""
        agent = Agent(config=mock_config)
        assert agent.full_system_prompt == "You are a coder"

    def test_agent_setup(self, mock_config: AgentConfig) -> None:
        """setup should initialize agent for task."""
        agent = Agent(config=mock_config)
        agent.setup("Write a hello world program")

        assert agent.task == "Write a hello world program"
        assert agent.status == AgentStatus.RUNNING
        assert len(agent.history) == 2
        # setup() should add system prompt and user message to history
        # We verify the behavior (messages added) rather than internal structure

    def test_agent_setup_multiple_calls(self, mock_config: AgentConfig) -> None:
        """Multiple setup calls should update task and history, preserve counters."""
        agent = Agent(config=mock_config)
        agent.setup("Task 1")
        agent.steps = 5
        agent.setup("Task 2")

        assert agent.task == "Task 2"
        # Note: setup() doesn't reset counters like steps - that's the actual behavior
        assert agent.steps == 5
        assert len(agent.history) == 2

    def test_agent_done_when_idle(self, mock_config: AgentConfig) -> None:
        """done() should return False when IDLE."""
        agent = Agent(config=mock_config)
        assert not agent.done()

    def test_agent_done_when_running(self, mock_config: AgentConfig) -> None:
        """done() should return False when RUNNING."""
        agent = Agent(config=mock_config)
        agent.setup("task")
        assert not agent.done()

    def test_agent_done_when_done(self, mock_config: AgentConfig) -> None:
        """done() should return True when DONE."""
        agent = Agent(config=mock_config)
        agent.status = AgentStatus.DONE
        assert agent.done()

    def test_agent_done_when_error(self, mock_config: AgentConfig) -> None:
        """done() should return True when ERROR."""
        agent = Agent(config=mock_config)
        agent.status = AgentStatus.ERROR
        assert agent.done()

    def test_agent_done_when_cancelled(self, mock_config: AgentConfig) -> None:
        """done() should return True when CANCELLED."""
        agent = Agent(config=mock_config)
        agent.status = AgentStatus.CANCELLED
        assert agent.done()

    def test_agent_done_when_blocked(self, mock_config: AgentConfig) -> None:
        """done() should return False when BLOCKED_ON_INPUT."""
        agent = Agent(config=mock_config)
        agent.status = AgentStatus.BLOCKED_ON_INPUT
        assert not agent.done()

    def test_agent_fail(self, mock_config: AgentConfig) -> None:
        """fail should set error state."""
        agent = Agent(config=mock_config)
        agent.setup("task")

        try:
            raise ValueError("Something went wrong")
        except ValueError as e:
            agent.fail(e)

        assert agent.status == AgentStatus.ERROR
        assert agent.error is not None
        assert agent.error.error_type == "ValueError"
        assert "Something went wrong" in agent.error.message

    def test_agent_fail_without_setup(self, mock_config: AgentConfig) -> None:
        """fail should work even without setup."""
        agent = Agent(config=mock_config)
        agent.fail(RuntimeError("Error"))

        assert agent.status == AgentStatus.ERROR
        assert agent.error is not None

    def test_agent_fail_preserves_traceback(self, mock_config: AgentConfig) -> None:
        """fail should capture full traceback."""
        agent = Agent(config=mock_config)

        def inner() -> None:
            raise ValueError("Deep error")

        def outer() -> None:
            inner()

        try:
            outer()
        except ValueError as e:
            agent.fail(e)

        assert agent.error is not None
        assert "inner" in agent.error.message
        assert "outer" in agent.error.message

    def test_agent_error_property(self, mock_config: AgentConfig) -> None:
        """error property should return ErrorInfo."""
        agent = Agent(config=mock_config)
        assert agent.error is None

        agent.fail(Exception("test"))
        assert isinstance(agent.error, ErrorInfo)

    def test_agent_created_at(self, mock_config: AgentConfig) -> None:
        """created_at should be set and not change."""
        agent = Agent(config=mock_config)
        created = agent.created_at
        agent.setup("task")
        assert agent.created_at == created

    def test_agent_missing_config(self) -> None:
        """Should raise when config not provided."""
        with pytest.raises(TypeError):
            Agent()  # type: ignore[call-arg]

    def test_agent_with_custom_state(self, mock_config: AgentConfig) -> None:
        """Should accept custom state."""
        state = AgentState(task="custom task")
        agent = Agent(config=mock_config, state=state)
        assert agent.task == "custom task"
