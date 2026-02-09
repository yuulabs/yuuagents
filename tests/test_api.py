"""Tests for yuuagents.daemon.api module — tests real API behavior."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from yuuagents.config import Config
from yuuagents.daemon.api import create_app
from yuuagents.daemon.docker import DockerManager
from yuuagents.daemon.manager import AgentManager


@pytest.fixture
def config() -> Config:
    """Create test config."""
    return Config()


@pytest.fixture
def docker_manager(config: Config) -> DockerManager:
    """Create and start DockerManager."""
    return DockerManager(image=config.docker.image)


@pytest.fixture
def agent_manager(config: Config, docker_manager: DockerManager) -> AgentManager:
    """Create AgentManager with real dependencies."""
    return AgentManager(config, docker_manager)


@pytest.fixture
def app(agent_manager: AgentManager):
    """Create Starlette app with real manager."""
    return create_app(agent_manager)


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client: TestClient) -> None:
        """Should return status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_get_only(self, client: TestClient) -> None:
        """Should only accept GET."""
        response = client.post("/health")
        assert response.status_code == 405


class TestCreateAgentEndpoint:
    """Tests for POST /api/agents endpoint."""

    def test_create_agent_success(self, client: TestClient) -> None:
        """Should create agent and return id."""
        request_data = {
            "agent": "main",
            "task": "write hello world",
            "tools": ["execute_bash"],
            "model": "gpt-4",
        }
        response = client.post("/api/agents", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "main"
        assert "task_id" in data
        assert len(data["task_id"]) > 0

    def test_create_agent_minimal(self, client: TestClient) -> None:
        """Should work with minimal required fields."""
        request_data = {
            "task": "test",
        }
        response = client.post("/api/agents", json=request_data)

        assert response.status_code == 201
        data = response.json()
        assert data["agent_id"] == "main"
        assert "task_id" in data

    def test_create_agent_invalid_json(self, client: TestClient) -> None:
        """Should return error for invalid JSON."""
        response = client.post(
            "/api/agents",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    def test_create_agent_missing_required_field(self, client: TestClient) -> None:
        """Should return error for missing required field."""
        request_data = {"agent": "main"}  # Missing task
        response = client.post("/api/agents", json=request_data)
        assert response.status_code == 400


class TestListAgentsEndpoint:
    """Tests for GET /api/agents endpoint."""

    def test_list_empty(self, client: TestClient) -> None:
        """Should return empty list initially."""
        response = client.get("/api/agents")

        assert response.status_code == 200
        assert response.json() == []

    def test_list_with_agents(self, client: TestClient) -> None:
        """Should return list of agents after creation."""
        # Create a couple of agents
        for i in range(2):
            client.post(
                "/api/agents",
                json={
                    "task": f"task{i}",
                },
            )

        response = client.get("/api/agents")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert all("task_id" in agent for agent in data)
        assert all("agent_id" in agent for agent in data)
        assert all("status" in agent for agent in data)


class TestGetAgentEndpoint:
    """Tests for GET /api/agents/{id} endpoint."""

    def test_get_agent_success(self, client: TestClient) -> None:
        """Should return agent info."""
        # Create an agent
        create_response = client.post(
            "/api/agents",
            json={
                "task": "test task",
            },
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/api/agents/{task_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == task_id
        assert data["agent_id"] == "main"
        assert data["task"] == "test task"
        assert "status" in data
        assert "created_at" in data

    def test_get_agent_not_found(self, client: TestClient) -> None:
        """Should return 404 for unknown agent."""
        response = client.get("/api/agents/nonexistent-task-id-12345")

        assert response.status_code == 404
        assert "error" in response.json()


class TestGetHistoryEndpoint:
    """Tests for GET /api/agents/{id}/history endpoint."""

    def test_get_history_success(self, client: TestClient) -> None:
        """Should return agent history."""
        # Create an agent
        create_response = client.post(
            "/api/agents",
            json={
                "task": "test",
            },
        )
        task_id = create_response.json()["task_id"]

        response = client.get(f"/api/agents/{task_id}/history")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # Should have at least system + user messages
        assert len(data) >= 2
        assert data[0]["role"] == "system"
        assert data[1]["role"] == "user"

    def test_get_history_not_found(self, client: TestClient) -> None:
        """Should return 404 for unknown agent."""
        response = client.get("/api/agents/nonexistent-task-id/history")

        assert response.status_code == 404


class TestPostInputEndpoint:
    """Tests for POST /api/agents/{id}/input endpoint."""

    def test_post_input_success(self, client: TestClient) -> None:
        """Should accept input for agent."""
        # Create an agent first
        create_response = client.post(
            "/api/agents",
            json={
                "task": "test",
            },
        )
        task_id = create_response.json()["task_id"]

        response = client.post(
            f"/api/agents/{task_id}/input", json={"content": "user response"}
        )

        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_post_input_not_found(self, client: TestClient) -> None:
        """Should return 404 for unknown agent."""
        response = client.post(
            "/api/agents/nonexistent-task-id/input", json={"content": "test"}
        )

        assert response.status_code == 404


class TestDeleteAgentEndpoint:
    """Tests for DELETE /api/agents/{id} endpoint."""

    def test_delete_agent_success(self, client: TestClient) -> None:
        """Should cancel agent."""
        # Create an agent
        create_response = client.post(
            "/api/agents",
            json={
                "task": "test",
            },
        )
        task_id = create_response.json()["task_id"]

        response = client.delete(f"/api/agents/{task_id}")

        assert response.status_code == 200
        assert response.json()["ok"] is True

        # Verify agent is marked as cancelled
        status_response = client.get(f"/api/agents/{task_id}")
        assert status_response.json()["status"] == "cancelled"

    def test_delete_agent_not_found(self, client: TestClient) -> None:
        """Should return 404 for unknown agent."""
        response = client.delete("/api/agents/nonexistent-task-id")

        assert response.status_code == 404


class TestListSkillsEndpoint:
    """Tests for GET /api/skills endpoint."""

    def test_list_skills(self, client: TestClient) -> None:
        """Should return skills list (may be empty)."""
        response = client.get("/api/skills")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestScanSkillsEndpoint:
    """Tests for POST /api/skills/scan endpoint."""

    def test_scan_skills(self, client: TestClient) -> None:
        """Should trigger rescan and return skills."""
        response = client.post("/api/skills/scan")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestGetConfigEndpoint:
    """Tests for GET /api/config endpoint."""

    def test_get_config(self, client: TestClient) -> None:
        """Should return sanitized config."""
        response = client.get("/api/config")

        assert response.status_code == 200
        data = response.json()
        assert "socket" in data
        assert "docker_image" in data
        assert "providers" in data
        assert "agents" in data
        assert "skill_paths" in data
        # Should NOT contain api keys
        assert "api_key" not in str(data).lower()
        assert "api_key_env" not in str(data).lower()


class TestAPIContentTypes:
    """Tests for API content type handling."""

    def test_json_content_type(self, client: TestClient) -> None:
        """Should return JSON content type."""
        response = client.get("/health")
        assert "application/json" in response.headers["content-type"]

    def test_post_requires_json(self, client: TestClient) -> None:
        """POST should require JSON content type."""
        response = client.post(
            "/api/agents",
            content=b'{"task": "test"}',
        )
        # Without Content-Type header, should still work or give clear error
        assert response.status_code in [201, 400, 415]
