"""Tests for yuuagents.init module."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from yuuagents.config import Config, DockerConfig, ProviderConfig, AgentEntry
from yuuagents.init import (
    _image_exists,
    _daemon_is_running,
    _runtime_dockerfile_text,
    build_runtime_image,
    runtime_image_tag,
    setup,
)


# ---------------------------------------------------------------------------
# runtime_image_tag
# ---------------------------------------------------------------------------


class TestRuntimeImageTag:
    def test_returns_versioned_tag(self) -> None:
        tag = runtime_image_tag()
        assert tag.startswith("yuuagents-runtime:")

    def test_fallback_to_latest(self) -> None:
        with patch("yuuagents.init.importlib.metadata.version", side_effect=Exception):
            tag = runtime_image_tag()
        assert tag == "yuuagents-runtime:latest"


# ---------------------------------------------------------------------------
# _runtime_dockerfile_text
# ---------------------------------------------------------------------------


class TestRuntimeDockerfileText:
    def test_returns_nonempty_string(self) -> None:
        text = _runtime_dockerfile_text()
        assert isinstance(text, str)
        assert "FROM" in text

    def test_raises_when_file_missing(self) -> None:
        with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                _runtime_dockerfile_text()


# ---------------------------------------------------------------------------
# _image_exists
# ---------------------------------------------------------------------------


class TestImageExists:
    def test_image_exists_true(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("yuuagents.init.subprocess.run", return_value=completed):
            assert _image_exists("some-image:latest") is True

    def test_image_exists_false(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=1)
        with patch("yuuagents.init.subprocess.run", return_value=completed):
            assert _image_exists("some-image:latest") is False

    def test_image_exists_docker_not_found(self) -> None:
        with patch("yuuagents.init.subprocess.run", side_effect=FileNotFoundError):
            assert _image_exists("some-image:latest") is False

    def test_image_exists_timeout(self) -> None:
        with patch(
            "yuuagents.init.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=10),
        ):
            assert _image_exists("some-image:latest") is False


# ---------------------------------------------------------------------------
# build_runtime_image
# ---------------------------------------------------------------------------


class TestBuildRuntimeImage:
    def test_build_success(self) -> None:
        success = subprocess.CompletedProcess(args=[], returncode=0)
        with patch("yuuagents.init.subprocess.run", return_value=success):
            assert build_runtime_image("test:v1") is True

    def test_build_failure(self) -> None:
        failure = subprocess.CompletedProcess(args=[], returncode=1)
        with patch("yuuagents.init.subprocess.run", return_value=failure):
            assert build_runtime_image("test:v1") is False

    def test_build_tags_latest(self) -> None:
        success = subprocess.CompletedProcess(args=[], returncode=0)
        calls: list = []

        def mock_run(*args, **kwargs):
            calls.append(args)
            return success

        with patch("yuuagents.init.subprocess.run", side_effect=mock_run):
            build_runtime_image("yuuagents-runtime:0.1.0")

        # Should have called docker build + docker tag
        assert len(calls) == 2

    def test_build_skips_tag_for_latest(self) -> None:
        success = subprocess.CompletedProcess(args=[], returncode=0)
        calls: list = []

        def mock_run(*args, **kwargs):
            calls.append(args)
            return success

        with patch("yuuagents.init.subprocess.run", side_effect=mock_run):
            build_runtime_image("yuuagents-runtime:latest")

        # Should only call docker build, no tag
        assert len(calls) == 1


# ---------------------------------------------------------------------------
# _daemon_is_running
# ---------------------------------------------------------------------------


class TestDaemonIsRunning:
    def test_not_running_no_socket(self, tmp_path: Path) -> None:
        assert _daemon_is_running(tmp_path / "nonexistent.sock") is False

    def test_not_running_stale_socket(self, tmp_path: Path) -> None:
        sock = tmp_path / "stale.sock"
        sock.touch()
        assert _daemon_is_running(sock) is False


# ---------------------------------------------------------------------------
# setup
# ---------------------------------------------------------------------------


class TestSetup:
    """Tests for the setup() function.

    These tests mock out Docker and daemon operations to test the
    directory/config/database logic in isolation.
    """

    @pytest.fixture
    def mock_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect YAGENTS_HOME to a temp directory."""
        home = tmp_path / ".yagents"
        monkeypatch.setattr("yuuagents.init.YAGENTS_HOME", home)
        monkeypatch.setattr("yuuagents.init.DEFAULT_CONFIG_PATH", home / "config.yaml")
        return home

    @pytest.fixture
    def basic_config(self, tmp_path: Path) -> Config:
        """A minimal Config that uses a temp database and public image."""
        db_path = tmp_path / "test.sqlite3"
        return Config(
            docker=DockerConfig(image="ubuntu:24.04"),
            providers={
                "test": ProviderConfig(
                    api_type="openai-chat-completion",
                    api_key_env="TEST_API_KEY",
                    default_model="gpt-4o",
                )
            },
            agents={
                "main": AgentEntry(
                    description="A minimal main agent",
                    provider="test",
                    model="gpt-4o",
                )
            },
            db=__import__("yuuagents.config", fromlist=["DbConfig"]).DbConfig(
                url=f"sqlite+aiosqlite:///{db_path}",
            ),
        )

    @pytest.mark.asyncio
    async def test_creates_directories(
        self, mock_home: Path, basic_config: Config
    ) -> None:
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(basic_config)

        assert mock_home.exists()
        assert (mock_home / "skills").exists()
        assert (mock_home / "dockers").exists()

    @pytest.mark.asyncio
    async def test_writes_config_file(
        self, mock_home: Path, basic_config: Config
    ) -> None:
        config_path = mock_home / "config.yaml"
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(basic_config)

        assert config_path.exists()
        data = yaml.safe_load(config_path.read_text())
        assert data["docker"]["image"] == "ubuntu:24.04"

    @pytest.mark.asyncio
    async def test_initializes_database(
        self, mock_home: Path, basic_config: Config
    ) -> None:
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(basic_config)

        db_path = basic_config.sqlite_path
        assert db_path is not None
        assert db_path.exists()

    @pytest.mark.asyncio
    async def test_builds_runtime_image_when_missing(self, mock_home: Path) -> None:
        cfg = Config(docker=DockerConfig(image="yuuagents-runtime:latest"))
        with (
            patch("yuuagents.init._image_exists", return_value=False),
            patch(
                "yuuagents.init.build_runtime_image", return_value=True
            ) as mock_build,
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(cfg)

        mock_build.assert_called_once_with("yuuagents-runtime:latest")

    @pytest.mark.asyncio
    async def test_skips_image_build_when_exists(self, mock_home: Path) -> None:
        cfg = Config(docker=DockerConfig(image="yuuagents-runtime:latest"))
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init.build_runtime_image") as mock_build,
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(cfg)

        mock_build.assert_not_called()

    @pytest.mark.asyncio
    async def test_pulls_public_image_when_missing(
        self, mock_home: Path, basic_config: Config
    ) -> None:
        success = subprocess.CompletedProcess(args=[], returncode=0)
        with (
            patch("yuuagents.init._image_exists", return_value=False),
            patch("yuuagents.init.subprocess.run", return_value=success) as mock_run,
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(basic_config)

        # Should have called docker pull
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "docker"
        assert call_args[1] == "pull"
        assert call_args[2] == "ubuntu:24.04"

    @pytest.mark.asyncio
    async def test_raises_on_image_build_failure(self, mock_home: Path) -> None:
        cfg = Config(docker=DockerConfig(image="yuuagents-runtime:latest"))
        with (
            patch("yuuagents.init._image_exists", return_value=False),
            patch("yuuagents.init.build_runtime_image", return_value=False),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            with pytest.raises(RuntimeError, match="Failed to build"):
                await setup(cfg)

    @pytest.mark.asyncio
    async def test_raises_on_invalid_config(self, mock_home: Path) -> None:
        cfg = Config(
            providers={"test": ProviderConfig()},
            agents={
                "main": AgentEntry(description="Main agent", provider="nonexistent")
            },
        )
        with pytest.raises(ValueError, match="validation failed"):
            await setup(cfg)

    @pytest.mark.asyncio
    async def test_starts_daemon_when_not_running(
        self, mock_home: Path, basic_config: Config
    ) -> None:
        call_count = 0

        def daemon_running_side_effect(path):
            nonlocal call_count
            call_count += 1
            # First call: not running; subsequent calls: running
            return call_count > 1

        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch(
                "yuuagents.init._daemon_is_running",
                side_effect=daemon_running_side_effect,
            ),
            patch("yuuagents.init._start_daemon") as mock_start,
        ):
            await setup(basic_config)

        mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_daemon_start_when_running(
        self, mock_home: Path, basic_config: Config
    ) -> None:
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
            patch("yuuagents.init._start_daemon") as mock_start,
        ):
            await setup(basic_config)

        mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_idempotent(self, mock_home: Path, basic_config: Config) -> None:
        """Calling setup() twice should not fail."""
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            await setup(basic_config)
            await setup(basic_config)

        assert mock_home.exists()

    @pytest.mark.asyncio
    async def test_accepts_path_string(self, mock_home: Path, tmp_path: Path) -> None:
        """setup() should accept a file path as string."""
        config_file = tmp_path / "my_config.yaml"
        config_file.write_text(
            yaml.dump(
                {"docker": {"image": "ubuntu:24.04"}},
                default_flow_style=False,
            )
        )
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            cfg = await setup(str(config_file))

        assert cfg.docker.image == "ubuntu:24.04"

    @pytest.mark.asyncio
    async def test_accepts_path_object(self, mock_home: Path, tmp_path: Path) -> None:
        """setup() should accept a Path object."""
        config_file = tmp_path / "my_config.yaml"
        config_file.write_text(
            yaml.dump(
                {"docker": {"image": "ubuntu:24.04"}},
                default_flow_style=False,
            )
        )
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            cfg = await setup(config_file)

        assert cfg.docker.image == "ubuntu:24.04"

    @pytest.mark.asyncio
    async def test_returns_config(self, mock_home: Path, basic_config: Config) -> None:
        with (
            patch("yuuagents.init._image_exists", return_value=True),
            patch("yuuagents.init._daemon_is_running", return_value=True),
        ):
            result = await setup(basic_config)

        assert isinstance(result, Config)
        assert result.docker.image == "ubuntu:24.04"
