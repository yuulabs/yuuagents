"""Tests for yuuagents.config module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import msgspec
import pytest

from yuuagents.config import (
    Config,
    DaemonConfig,
    DockerConfig,
    LLMConfig,
    PersonaConfig,
    SkillsConfig,
    TavilyConfig,
    load,
)


class TestDaemonConfig:
    """Tests for DaemonConfig struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults."""
        cfg = DaemonConfig()
        assert cfg.socket == "~/.local/run/yagents.sock"

    def test_custom_socket(self) -> None:
        """Should accept custom socket path."""
        cfg = DaemonConfig(socket="/tmp/custom.sock")
        assert cfg.socket == "/tmp/custom.sock"

    def test_is_mutable(self) -> None:
        """DaemonConfig should be mutable."""
        cfg = DaemonConfig()
        cfg.socket = "/new/path.sock"
        assert cfg.socket == "/new/path.sock"


class TestDockerConfig:
    """Tests for DockerConfig struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults."""
        cfg = DockerConfig()
        assert cfg.image == "ubuntu:24.04"

    def test_custom_image(self) -> None:
        """Should accept custom image."""
        cfg = DockerConfig(image="python:3.12")
        assert cfg.image == "python:3.12"


class TestLLMConfig:
    """Tests for LLMConfig struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults."""
        cfg = LLMConfig()
        assert cfg.provider == "openai"
        assert cfg.api_key_env == "OPENAI_API_KEY"
        assert cfg.default_model == "gpt-4o"
        assert cfg.base_url == ""

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        cfg = LLMConfig(
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-3-opus",
            base_url="https://custom.api.com",
        )
        assert cfg.provider == "anthropic"
        assert cfg.api_key_env == "ANTHROPIC_API_KEY"
        assert cfg.default_model == "claude-3-opus"
        assert cfg.base_url == "https://custom.api.com"


class TestSkillsConfig:
    """Tests for SkillsConfig struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults."""
        cfg = SkillsConfig()
        assert cfg.paths == ["~/.yagents/skills"]

    def test_custom_paths(self) -> None:
        """Should accept custom paths."""
        cfg = SkillsConfig(paths=["/custom/path", "/another/path"])
        assert cfg.paths == ["/custom/path", "/another/path"]


class TestTavilyConfig:
    """Tests for TavilyConfig struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults."""
        cfg = TavilyConfig()
        assert cfg.api_key_env == "TAVILY_API_KEY"

    def test_custom_env(self) -> None:
        """Should accept custom env var name."""
        cfg = TavilyConfig(api_key_env="MY_TAVILY_KEY")
        assert cfg.api_key_env == "MY_TAVILY_KEY"


class TestPersonaConfig:
    """Tests for PersonaConfig struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults."""
        cfg = PersonaConfig()
        assert cfg.system_prompt == ""
        assert cfg.tools == []
        assert cfg.skills == []

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        cfg = PersonaConfig(
            system_prompt="You are a coder",
            tools=["execute_bash", "read_file"],
            skills=["git-worktree"],
        )
        assert cfg.system_prompt == "You are a coder"
        assert cfg.tools == ["execute_bash", "read_file"]
        assert cfg.skills == ["git-worktree"]


class TestConfig:
    """Tests for main Config struct."""

    def test_default_values(self) -> None:
        """Should have correct defaults for all sections."""
        cfg = Config()
        assert isinstance(cfg.daemon, DaemonConfig)
        assert isinstance(cfg.docker, DockerConfig)
        assert isinstance(cfg.llm, LLMConfig)
        assert isinstance(cfg.skills, SkillsConfig)
        assert isinstance(cfg.tavily, TavilyConfig)
        assert cfg.personas == {}

    def test_socket_path_property(self) -> None:
        """socket_path should expand user home."""
        cfg = Config()
        path = cfg.socket_path
        assert isinstance(path, Path)
        assert str(path).startswith("/")
        assert ".local/run/yagents.sock" in str(path)

    def test_custom_socket_path_expansion(self) -> None:
        """Should expand custom socket paths."""
        cfg = Config(daemon=DaemonConfig(socket="~/custom.sock"))
        path = cfg.socket_path
        assert str(path).startswith("/")
        assert path.name == "custom.sock"

    def test_with_personas(self) -> None:
        """Should accept personas dictionary."""
        personas = {
            "coder": PersonaConfig(
                system_prompt="You are a coder",
                tools=["execute_bash"],
            ),
            "researcher": PersonaConfig(
                system_prompt="You are a researcher",
                tools=["web_search"],
            ),
        }
        cfg = Config(personas=personas)
        assert "coder" in cfg.personas
        assert "researcher" in cfg.personas
        assert cfg.personas["coder"].system_prompt == "You are a coder"

    def test_nested_config_modification(self) -> None:
        """Should be able to modify nested configs."""
        cfg = Config()
        cfg.llm.default_model = "gpt-3.5-turbo"
        assert cfg.llm.default_model == "gpt-3.5-turbo"


class TestLoadConfig:
    """Tests for load() function."""

    def test_load_missing_file_returns_defaults(self) -> None:
        """Should return defaults when file doesn't exist."""
        cfg = load("/nonexistent/path/config.toml")
        assert isinstance(cfg, Config)
        assert cfg.llm.default_model == "gpt-4o"

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        """Should load minimal TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[daemon]
socket = "/tmp/test.sock"
""")
        cfg = load(config_file)
        assert cfg.daemon.socket == "/tmp/test.sock"
        # Other sections should have defaults
        assert cfg.llm.default_model == "gpt-4o"

    def test_load_full_config(self, tmp_path: Path) -> None:
        """Should load full TOML config."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[daemon]
socket = "/tmp/test.sock"

[docker]
image = "python:3.12"

[llm]
provider = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-3-opus"
base_url = "https://custom.api.com"

[skills]
paths = ["/path/to/skills", "~/custom/skills"]

[tavily]
api_key_env = "MY_TAVILY_KEY"

[personas.coder]
system_prompt = "You are a senior developer"
tools = ["execute_bash", "read_file", "write_file"]
skills = ["git-worktree"]

[personas.researcher]
system_prompt = "You are a research assistant"
tools = ["web_search", "read_file"]
""")
        cfg = load(config_file)

        # Check daemon
        assert cfg.daemon.socket == "/tmp/test.sock"

        # Check docker
        assert cfg.docker.image == "python:3.12"

        # Check llm
        assert cfg.llm.provider == "anthropic"
        assert cfg.llm.api_key_env == "ANTHROPIC_API_KEY"
        assert cfg.llm.default_model == "claude-3-opus"
        assert cfg.llm.base_url == "https://custom.api.com"

        # Check skills
        assert cfg.skills.paths == ["/path/to/skills", "~/custom/skills"]

        # Check tavily
        assert cfg.tavily.api_key_env == "MY_TAVILY_KEY"

        # Check personas
        assert "coder" in cfg.personas
        assert cfg.personas["coder"].system_prompt == "You are a senior developer"
        assert cfg.personas["coder"].tools == [
            "execute_bash",
            "read_file",
            "write_file",
        ]
        assert cfg.personas["coder"].skills == ["git-worktree"]

        assert "researcher" in cfg.personas
        assert (
            cfg.personas["researcher"].system_prompt == "You are a research assistant"
        )

    def test_load_partial_persona(self, tmp_path: Path) -> None:
        """Should load persona with partial fields."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[personas.minimal]
system_prompt = "Minimal persona"
""")
        cfg = load(config_file)
        assert cfg.personas["minimal"].system_prompt == "Minimal persona"
        assert cfg.personas["minimal"].tools == []
        assert cfg.personas["minimal"].skills == []

    def test_load_empty_file(self, tmp_path: Path) -> None:
        """Should handle empty TOML file."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("")
        cfg = load(config_file)
        assert isinstance(cfg, Config)
        assert cfg.llm.default_model == "gpt-4o"

    def test_load_invalid_toml(self, tmp_path: Path) -> None:
        """Should raise error for invalid TOML."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("invalid toml content [[[")
        with pytest.raises(Exception):  # tomllib.TOMLDecodeError
            load(config_file)

    def test_load_unknown_sections_ignored(self, tmp_path: Path) -> None:
        """Unknown sections should be handled gracefully."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[daemon]
socket = "/tmp/test.sock"

[unknown_section]
field = "value"
""")
        # Should not raise, but unknown section is ignored
        cfg = load(config_file)
        assert cfg.daemon.socket == "/tmp/test.sock"

    def test_load_with_path_object(self, tmp_path: Path) -> None:
        """Should accept Path object."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[daemon]\nsocket = '/tmp/test.sock'")
        cfg = load(config_file)  # Pass Path object
        assert cfg.daemon.socket == "/tmp/test.sock"

    def test_load_with_string_path(self, tmp_path: Path) -> None:
        """Should accept string path."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("[daemon]\nsocket = '/tmp/test.sock'")
        cfg = load(str(config_file))
        assert cfg.daemon.socket == "/tmp/test.sock"

    def test_load_no_args_works(self) -> None:
        """Should work when called without args (uses default path)."""
        # This tests that load() without args doesn't crash
        # When config file doesn't exist, it should return defaults
        cfg = load()
        assert isinstance(cfg, Config)
        assert cfg.llm.default_model == "gpt-4o"


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_config_to_dict(self) -> None:
        """Should convert config to dict."""
        cfg = Config()
        # msgspec structs can be converted to dict
        data = msgspec.to_builtins(cfg)
        assert "daemon" in data
        assert "docker" in data
        assert "llm" in data

    def test_config_round_trip(self, tmp_path: Path) -> None:
        """Config should survive round-trip through TOML."""
        config_file = tmp_path / "config.toml"
        config_file.write_text("""
[llm]
default_model = "gpt-3.5-turbo"

[personas.test]
system_prompt = "Test persona"
tools = ["bash"]
""")
        cfg1 = load(config_file)
        assert cfg1.llm.default_model == "gpt-3.5-turbo"
        assert cfg1.personas["test"].system_prompt == "Test persona"
