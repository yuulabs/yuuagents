"""Tests for yuuagents.types module."""

from __future__ import annotations

from datetime import datetime, timezone

import msgspec
import pytest

from yuuagents.types import (
    AgentInfo,
    AgentStatus,
    ErrorInfo,
    SkillInfo,
    TaskRequest,
)


class TestAgentStatus:
    """Tests for AgentStatus enum."""

    def test_agent_status_values(self) -> None:
        """AgentStatus should have expected values."""
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.RUNNING.value == "running"
        assert AgentStatus.DONE.value == "done"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.BLOCKED_ON_INPUT.value == "blocked_on_input"
        assert AgentStatus.CANCELLED.value == "cancelled"

    def test_agent_status_is_str_enum(self) -> None:
        """AgentStatus should inherit from str."""
        assert issubclass(AgentStatus, str)
        status = AgentStatus.RUNNING
        assert isinstance(status, str)
        assert status == "running"

    def test_agent_status_comparison(self) -> None:
        """AgentStatus should be comparable to strings."""
        assert AgentStatus.DONE == "done"
        assert AgentStatus.ERROR != "done"
        assert "running" == AgentStatus.RUNNING

    def test_all_statuses(self) -> None:
        """Should have exactly 6 statuses."""
        expected = {"idle", "running", "done", "error", "blocked_on_input", "cancelled"}
        actual = {s.value for s in AgentStatus}
        assert actual == expected


class TestErrorInfo:
    """Tests for ErrorInfo struct."""

    def test_error_info_creation(self) -> None:
        """Should create ErrorInfo with all fields."""
        now = datetime.now(timezone.utc)
        error = ErrorInfo(
            message="Something went wrong",
            error_type="ValueError",
            timestamp=now,
        )
        assert error.message == "Something went wrong"
        assert error.error_type == "ValueError"
        assert error.timestamp == now

    def test_error_info_with_traceback(self) -> None:
        """Should handle multi-line error messages with tracebacks."""
        traceback_msg = """Traceback (most recent call last):
  File "test.py", line 10, in <module>
    raise ValueError("test")
ValueError: test"""
        error = ErrorInfo(
            message=traceback_msg,
            error_type="ValueError",
            timestamp=datetime.now(timezone.utc),
        )
        assert "Traceback" in error.message
        assert error.error_type == "ValueError"

    def test_error_info_is_frozen(self) -> None:
        """ErrorInfo should be immutable."""
        error = ErrorInfo(
            message="test",
            error_type="Exception",
            timestamp=datetime.now(timezone.utc),
        )
        with pytest.raises(AttributeError):
            error.message = "new message"  # type: ignore[misc]

    def test_error_info_kw_only(self) -> None:
        """ErrorInfo should require keyword arguments."""
        with pytest.raises(TypeError):
            ErrorInfo("message", "Type", datetime.now(timezone.utc))  # type: ignore[misc]

    def test_error_info_serialization(self) -> None:
        """Should serialize to JSON via msgspec."""
        now = datetime.now(timezone.utc)
        error = ErrorInfo(
            message="test error",
            error_type="RuntimeError",
            timestamp=now,
        )
        encoded = msgspec.json.encode(error)
        decoded = msgspec.json.decode(encoded, type=ErrorInfo)
        assert decoded.message == error.message
        assert decoded.error_type == error.error_type


class TestTaskRequest:
    """Tests for TaskRequest struct."""

    def test_task_request_minimal(self) -> None:
        """Should create TaskRequest with minimal required fields."""
        req = TaskRequest(task="write hello world")
        assert req.agent == "main"
        assert req.persona == ""
        assert req.task == "write hello world"
        assert req.tools == []
        assert req.skills == []
        assert req.model == ""
        assert req.container == ""
        assert req.image == ""

    def test_task_request_full(self) -> None:
        """Should create TaskRequest with all fields."""
        req = TaskRequest(
            agent="researcher",
            persona="Custom persona",
            task="write hello world",
            tools=["execute_bash", "read_file"],
            skills=["git-worktree"],
            model="gpt-4o",
            container="my-container",
            image="ubuntu:24.04",
        )
        assert req.agent == "researcher"
        assert req.persona == "Custom persona"
        assert req.task == "write hello world"
        assert req.tools == ["execute_bash", "read_file"]
        assert req.skills == ["git-worktree"]
        assert req.model == "gpt-4o"
        assert req.container == "my-container"
        assert req.image == "ubuntu:24.04"

    def test_task_request_is_frozen(self) -> None:
        """TaskRequest should be immutable."""
        req = TaskRequest(task="test")
        with pytest.raises(AttributeError):
            req.task = "new task"  # type: ignore[misc]

    def test_task_request_kw_only(self) -> None:
        """TaskRequest should require keyword arguments."""
        with pytest.raises(TypeError):
            TaskRequest("main", "task")  # type: ignore[misc]

    def test_task_request_missing_required(self) -> None:
        """TaskRequest should require task."""
        with pytest.raises(TypeError):
            TaskRequest()  # type: ignore[call-arg]

    def test_task_request_serialization(self) -> None:
        """Should serialize/deserialize via msgspec."""
        req = TaskRequest(
            agent="researcher",
            persona="Custom persona",
            task="search for python",
            tools=["web_search"],
            model="gpt-4",
        )
        encoded = msgspec.json.encode(req)
        decoded = msgspec.json.decode(encoded, type=TaskRequest)
        assert decoded.agent == req.agent
        assert decoded.persona == req.persona
        assert decoded.task == req.task
        assert decoded.tools == req.tools
        assert decoded.model == req.model

    def test_task_request_json_decode(self) -> None:
        """Should decode from JSON bytes."""
        json_data = b'{"task": "hello", "tools": ["bash"]}'
        decoder = msgspec.json.Decoder(TaskRequest)
        req = decoder.decode(json_data)
        assert req.agent == "main"
        assert req.task == "hello"
        assert req.tools == ["bash"]

    def test_task_request_empty_lists(self) -> None:
        """Should handle empty lists for optional array fields."""
        req = TaskRequest(
            task="test",
            tools=[],
            skills=[],
        )
        assert req.tools == []
        assert req.skills == []

    def test_task_request_extra_fields_ignored(self) -> None:
        """Extra fields should be ignored during JSON decoding (default msgspec behavior)."""
        json_data = b'{"task": "hello", "unknown_field": "value"}'
        # msgspec ignores unknown fields by default
        req = msgspec.json.decode(json_data, type=TaskRequest)
        assert req.agent == "main"
        assert req.task == "hello"
        # unknown_field is silently ignored


class TestAgentInfo:
    """Tests for AgentInfo struct."""

    def test_agent_info_minimal(self) -> None:
        """Should create AgentInfo with required fields."""
        info = AgentInfo(
            agent_id="abc123",
            persona="coder",
            task="write code",
            status="running",
            created_at="2025-01-01T00:00:00Z",
        )
        assert info.agent_id == "abc123"
        assert info.persona == "coder"
        assert info.task == "write code"
        assert info.status == "running"
        assert info.created_at == "2025-01-01T00:00:00Z"
        assert info.steps == 0
        assert info.total_tokens == 0
        assert info.total_cost_usd == 0.0
        assert info.error is None

    def test_agent_info_full(self) -> None:
        """Should create AgentInfo with all fields."""
        error = ErrorInfo(
            message="error occurred",
            error_type="Exception",
            timestamp=datetime.now(timezone.utc),
        )
        info = AgentInfo(
            agent_id="abc123",
            persona="coder",
            task="write code",
            status="error",
            created_at="2025-01-01T00:00:00Z",
            steps=5,
            total_tokens=1000,
            total_cost_usd=0.05,
            error=error,
        )
        assert info.steps == 5
        assert info.total_tokens == 1000
        assert info.total_cost_usd == 0.05
        assert info.error == error

    def test_agent_info_is_frozen(self) -> None:
        """AgentInfo should be immutable."""
        info = AgentInfo(
            agent_id="abc123",
            persona="coder",
            task="test",
            status="running",
            created_at="2025-01-01T00:00:00Z",
        )
        with pytest.raises(AttributeError):
            info.status = "done"  # type: ignore[misc]

    def test_agent_info_kw_only(self) -> None:
        """AgentInfo should require keyword arguments."""
        with pytest.raises(TypeError):
            AgentInfo("id", "persona", "task", "status", "created_at")  # type: ignore[misc]

    def test_agent_info_missing_required(self) -> None:
        """AgentInfo should require all mandatory fields."""
        with pytest.raises(TypeError):
            AgentInfo()  # type: ignore[call-arg]

    def test_agent_info_serialization(self) -> None:
        """Should serialize to JSON via msgspec."""
        info = AgentInfo(
            agent_id="test-id",
            persona="researcher",
            task="search",
            status="done",
            created_at="2025-01-01T00:00:00Z",
            steps=3,
            total_tokens=500,
        )
        encoded = msgspec.json.encode(info)
        decoded = msgspec.json.decode(encoded, type=AgentInfo)
        assert decoded.agent_id == info.agent_id
        assert decoded.steps == info.steps

    def test_agent_info_with_error_serialization(self) -> None:
        """Should serialize AgentInfo with nested ErrorInfo."""
        error = ErrorInfo(
            message="test error",
            error_type="ValueError",
            timestamp=datetime.now(timezone.utc),
        )
        info = AgentInfo(
            agent_id="test-id",
            persona="coder",
            task="test",
            status="error",
            created_at="2025-01-01T00:00:00Z",
            error=error,
        )
        encoded = msgspec.json.encode(info)
        decoded = msgspec.json.decode(encoded, type=AgentInfo)
        assert decoded.error is not None
        assert decoded.error.message == "test error"


class TestSkillInfo:
    """Tests for SkillInfo struct."""

    def test_skill_info_creation(self) -> None:
        """Should create SkillInfo with all fields."""
        skill = SkillInfo(
            name="git-worktree",
            description="Manage git worktrees",
            location="~/.yagents/skills/git-worktree/SKILL.md",
        )
        assert skill.name == "git-worktree"
        assert skill.description == "Manage git worktrees"
        assert skill.location == "~/.yagents/skills/git-worktree/SKILL.md"

    def test_skill_info_is_frozen(self) -> None:
        """SkillInfo should be immutable."""
        skill = SkillInfo(
            name="test",
            description="test skill",
            location="/path/to/skill",
        )
        with pytest.raises(AttributeError):
            skill.name = "new-name"  # type: ignore[misc]

    def test_skill_info_kw_only(self) -> None:
        """SkillInfo should require keyword arguments."""
        with pytest.raises(TypeError):
            SkillInfo("name", "desc", "loc")  # type: ignore[misc]

    def test_skill_info_missing_required(self) -> None:
        """SkillInfo should require all fields."""
        with pytest.raises(TypeError):
            SkillInfo()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            SkillInfo(name="test")  # type: ignore[call-arg]

    def test_skill_info_serialization(self) -> None:
        """Should serialize to JSON via msgspec."""
        skill = SkillInfo(
            name="python-testing",
            description="Python testing patterns",
            location="/skills/python-testing/SKILL.md",
        )
        encoded = msgspec.json.encode(skill)
        decoded = msgspec.json.decode(encoded, type=SkillInfo)
        assert decoded.name == skill.name
        assert decoded.description == skill.description
        assert decoded.location == skill.location

    def test_skill_info_empty_strings(self) -> None:
        """Should allow empty strings (though not useful)."""
        skill = SkillInfo(name="", description="", location="")
        assert skill.name == ""
        assert skill.description == ""
        assert skill.location == ""


class TestTypeIntegration:
    """Integration tests for type interactions."""

    def test_agent_status_in_agent_info(self) -> None:
        """AgentStatus values should be usable in AgentInfo.status."""
        for status in AgentStatus:
            info = AgentInfo(
                agent_id="test",
                persona="test",
                task="test",
                status=status.value,
                created_at="2025-01-01T00:00:00Z",
            )
            assert info.status == status.value

    def test_error_info_in_agent_info_roundtrip(self) -> None:
        """ErrorInfo nested in AgentInfo should round-trip correctly."""
        error = ErrorInfo(
            message="Detailed error message with traceback",
            error_type="RuntimeError",
            timestamp=datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        info = AgentInfo(
            agent_id="agent-1",
            persona="coder",
            task="complex task",
            status="error",
            created_at="2025-01-01T12:00:00Z",
            error=error,
        )

        # Serialize and deserialize
        encoded = msgspec.json.encode(info)
        decoded = msgspec.json.decode(encoded, type=AgentInfo)

        assert decoded.error is not None
        assert decoded.error.message == error.message
        assert decoded.error.error_type == error.error_type
        # Note: timestamp comparison might need tolerance due to JSON serialization
        assert decoded.error.timestamp is not None

    def test_list_of_agent_info_serialization(self) -> None:
        """Should serialize list of AgentInfo."""
        agents = [
            AgentInfo(
                agent_id="agent-1",
                persona="coder",
                task="task1",
                status="running",
                created_at="2025-01-01T00:00:00Z",
            ),
            AgentInfo(
                agent_id="agent-2",
                persona="researcher",
                task="task2",
                status="done",
                created_at="2025-01-01T01:00:00Z",
            ),
        ]
        encoded = msgspec.json.encode(agents)
        decoded = msgspec.json.decode(encoded, type=list[AgentInfo])
        assert len(decoded) == 2
        assert decoded[0].agent_id == "agent-1"
        assert decoded[1].agent_id == "agent-2"

    def test_list_of_skill_info_serialization(self) -> None:
        """Should serialize list of SkillInfo."""
        skills = [
            SkillInfo(name="skill1", description="desc1", location="loc1"),
            SkillInfo(name="skill2", description="desc2", location="loc2"),
        ]
        encoded = msgspec.json.encode(skills)
        decoded = msgspec.json.decode(encoded, type=list[SkillInfo])
        assert len(decoded) == 2
        assert decoded[0].name == "skill1"
        assert decoded[1].name == "skill2"
