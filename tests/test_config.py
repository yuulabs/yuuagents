"""Tests for yuuagents.config module."""

from __future__ import annotations

from pathlib import Path

import msgspec
import pytest

from yuuagents.config import (
    AgentEntry,
    Config,
    DaemonConfig,
    DbConfig,
    DockerConfig,
    PricingEntry,
    ProviderConfig,
    SkillsConfig,
    TavilyConfig,
    _deep_merge,
    load,
    load_merged,
)


class TestDaemonConfig:
    """Tests for DaemonConfig struct."""

    def test_default_values(self) -> None:
        cfg = DaemonConfig()
        assert cfg.socket == "~/.yagents/yagents.sock"
        assert cfg.log_level == "info"

    def test_custom_socket(self) -> None:
        cfg = DaemonConfig(socket="/tmp/custom.sock")
        assert cfg.socket == "/tmp/custom.sock"

    def test_custom_log_level(self) -> None:
        cfg = DaemonConfig(log_level="debug")
        assert cfg.log_level == "debug"


class TestDockerConfig:
    """Tests for DockerConfig struct."""

    def test_default_values(self) -> None:
        cfg = DockerConfig()
        assert cfg.image == "ubuntu:24.04"

    def test_custom_image(self) -> None:
        cfg = DockerConfig(image="python:3.12")
        assert cfg.image == "python:3.12"


class TestSkillsConfig:
    """Tests for SkillsConfig struct."""

    def test_default_values(self) -> None:
        cfg = SkillsConfig()
        assert cfg.paths == ["~/.yagents/skills"]

    def test_custom_paths(self) -> None:
        cfg = SkillsConfig(paths=["/custom/path", "/another/path"])
        assert cfg.paths == ["/custom/path", "/another/path"]


class TestTavilyConfig:
    """Tests for TavilyConfig struct."""

    def test_default_values(self) -> None:
        cfg = TavilyConfig()
        assert cfg.api_key_env == "TAVILY_API_KEY"

    def test_custom_env(self) -> None:
        cfg = TavilyConfig(api_key_env="MY_TAVILY_KEY")
        assert cfg.api_key_env == "MY_TAVILY_KEY"


class TestDbConfig:
    def test_default_values(self) -> None:
        cfg = DbConfig()
        assert cfg.url == "sqlite+aiosqlite:///~/.yagents/tasks.sqlite3"

    def test_custom_url(self) -> None:
        cfg = DbConfig(url="sqlite+aiosqlite:////tmp/x.sqlite3")
        assert cfg.url == "sqlite+aiosqlite:////tmp/x.sqlite3"


class TestPricingEntry:
    """Tests for PricingEntry struct."""

    def test_required_model(self) -> None:
        entry = PricingEntry(model="gpt-4o")
        assert entry.model == "gpt-4o"
        assert entry.input_mtok == 0.0
        assert entry.output_mtok == 0.0
        assert entry.cache_read_mtok == 0.0
        assert entry.cache_write_mtok == 0.0

    def test_full_pricing(self) -> None:
        entry = PricingEntry(
            model="gpt-4o",
            input_mtok=2.50,
            output_mtok=10.00,
            cache_read_mtok=1.25,
            cache_write_mtok=3.75,
        )
        assert entry.input_mtok == 2.50
        assert entry.output_mtok == 10.00
        assert entry.cache_read_mtok == 1.25
        assert entry.cache_write_mtok == 3.75


class TestProviderConfig:
    """Tests for ProviderConfig struct."""

    def test_default_values(self) -> None:
        cfg = ProviderConfig()
        assert cfg.api_type == "openai-chat-completion"
        assert cfg.api_key_env == "OPENAI_API_KEY"
        assert cfg.default_model == "gpt-4o"
        assert cfg.base_url == ""
        assert cfg.organization == ""
        assert cfg.pricing == []

    def test_anthropic_provider(self) -> None:
        cfg = ProviderConfig(
            api_type="anthropic-messages",
            api_key_env="ANTHROPIC_API_KEY",
            default_model="claude-sonnet-4-20250514",
        )
        assert cfg.api_type == "anthropic-messages"
        assert cfg.api_key_env == "ANTHROPIC_API_KEY"
        assert cfg.default_model == "claude-sonnet-4-20250514"

    def test_with_pricing(self) -> None:
        cfg = ProviderConfig(
            pricing=[
                PricingEntry(model="gpt-4o", input_mtok=2.50, output_mtok=10.00),
            ]
        )
        assert len(cfg.pricing) == 1
        assert cfg.pricing[0].model == "gpt-4o"

    def test_with_base_url(self) -> None:
        cfg = ProviderConfig(
            api_type="openai-chat-completion",
            base_url="https://openrouter.ai/api/v1",
        )
        assert cfg.base_url == "https://openrouter.ai/api/v1"


class TestAgentEntry:
    """Tests for AgentEntry struct."""

    def test_default_values(self) -> None:
        entry = AgentEntry()
        assert entry.provider == ""
        assert entry.model == ""
        assert entry.persona == ""
        assert entry.tools == []
        assert entry.skills == []

    def test_full_agent(self) -> None:
        entry = AgentEntry(
            provider="openai-default",
            model="gpt-4o",
            persona="You are a coder",
            tools=["execute_bash", "read_file"],
            skills=["git-worktree"],
        )
        assert entry.provider == "openai-default"
        assert entry.model == "gpt-4o"
        assert entry.persona == "You are a coder"
        assert entry.tools == ["execute_bash", "read_file"]
        assert entry.skills == ["git-worktree"]


class TestConfig:
    """Tests for main Config struct."""

    def test_default_values(self) -> None:
        cfg = Config()
        assert isinstance(cfg.daemon, DaemonConfig)
        assert isinstance(cfg.db, DbConfig)
        assert isinstance(cfg.docker, DockerConfig)
        assert isinstance(cfg.skills, SkillsConfig)
        assert isinstance(cfg.tavily, TavilyConfig)
        assert cfg.providers == {}
        assert cfg.agents == {}

    def test_socket_path_property(self) -> None:
        cfg = Config()
        path = cfg.socket_path
        assert isinstance(path, Path)
        assert str(path).startswith("/")
        assert "yagents/yagents.sock" in str(path)

    def test_custom_socket_path_expansion(self) -> None:
        cfg = Config(daemon=DaemonConfig(socket="~/custom.sock"))
        path = cfg.socket_path
        assert str(path).startswith("/")
        assert path.name == "custom.sock"

    def test_db_url_expansion(self) -> None:
        cfg = Config(db=DbConfig(url="sqlite+aiosqlite:///~/.yagents/x.sqlite3"))
        assert cfg.db_url.startswith("sqlite+aiosqlite:///")
        assert "~" not in cfg.db_url

    def test_with_providers_and_agents(self) -> None:
        cfg = Config(
            providers={
                "openai-default": ProviderConfig(
                    api_type="openai-chat-completion",
                    default_model="gpt-4o",
                ),
            },
            agents={
                "main": AgentEntry(
                    provider="openai-default",
                    model="gpt-4o",
                    persona="You are a coder",
                    tools=["execute_bash"],
                ),
            },
        )
        assert "openai-default" in cfg.providers
        assert "main" in cfg.agents
        assert cfg.agents["main"].provider == "openai-default"

    def test_validate_valid_config(self) -> None:
        cfg = Config(
            providers={"p1": ProviderConfig()},
            agents={"main": AgentEntry(provider="p1")},
        )
        assert cfg.validate() == []

    def test_validate_missing_provider_reference(self) -> None:
        cfg = Config(
            providers={},
            agents={"main": AgentEntry(provider="nonexistent")},
        )
        errors = cfg.validate()
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_validate_empty_provider_reference_ok(self) -> None:
        """Agent with empty provider should not trigger validation error."""
        cfg = Config(
            agents={"main": AgentEntry(provider="")},
        )
        assert cfg.validate() == []

    def test_nested_config_modification(self) -> None:
        cfg = Config()
        cfg.docker.image = "python:3.12"
        assert cfg.docker.image == "python:3.12"


class TestDeepMerge:
    """Tests for _deep_merge helper."""

    def test_simple_override(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3}

    def test_nested_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3}}
        result = _deep_merge(base, override)
        assert result == {"x": {"a": 1, "b": 3}}

    def test_add_new_key(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self) -> None:
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"b": 1}}

    def test_list_replaced_not_merged(self) -> None:
        base = {"a": [1, 2]}
        override = {"a": [3]}
        result = _deep_merge(base, override)
        assert result == {"a": [3]}


class TestLoadConfig:
    """Tests for load() function."""

    def test_load_missing_file_returns_defaults(self) -> None:
        cfg = load("/nonexistent/path/config.yaml")
        assert isinstance(cfg, Config)
        assert cfg.docker.image == "ubuntu:24.04"

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("daemon:\n  socket: /tmp/test.sock\n")
        cfg = load(config_file)
        assert cfg.daemon.socket == "/tmp/test.sock"
        # Other sections should have defaults
        assert cfg.docker.image == "ubuntu:24.04"

    def test_load_full_config(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""\
daemon:
  socket: /tmp/test.sock
  log_level: debug

docker:
  image: python:3.12

skills:
  paths:
    - /path/to/skills
    - ~/custom/skills

tavily:
  api_key_env: MY_TAVILY_KEY

providers:
  openai-main:
    api_type: openai-chat-completion
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o
    base_url: https://custom.api.com
    organization: my-org
    pricing:
      - model: gpt-4o
        input_mtok: 2.50
        output_mtok: 10.00

  anthropic-main:
    api_type: anthropic-messages
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-sonnet-4-20250514

agents:
  main:
    provider: openai-main
    model: gpt-4o
    persona: You are a senior developer
    tools:
      - execute_bash
      - read_file
      - write_file
    skills:
      - git-worktree

  researcher:
    provider: anthropic-main
    persona: You are a research assistant
    tools:
      - web_search
      - read_file
""")
        cfg = load(config_file)

        # Check daemon
        assert cfg.daemon.socket == "/tmp/test.sock"
        assert cfg.daemon.log_level == "debug"

        # Check docker
        assert cfg.docker.image == "python:3.12"

        # Check skills
        assert cfg.skills.paths == ["/path/to/skills", "~/custom/skills"]

        # Check tavily
        assert cfg.tavily.api_key_env == "MY_TAVILY_KEY"

        # Check providers
        assert "openai-main" in cfg.providers
        p = cfg.providers["openai-main"]
        assert p.api_type == "openai-chat-completion"
        assert p.default_model == "gpt-4o"
        assert p.base_url == "https://custom.api.com"
        assert p.organization == "my-org"
        assert len(p.pricing) == 1
        assert p.pricing[0].model == "gpt-4o"
        assert p.pricing[0].input_mtok == 2.50

        assert "anthropic-main" in cfg.providers
        assert cfg.providers["anthropic-main"].api_type == "anthropic-messages"

        # Check agents
        assert "main" in cfg.agents
        main = cfg.agents["main"]
        assert main.provider == "openai-main"
        assert main.persona == "You are a senior developer"
        assert main.tools == ["execute_bash", "read_file", "write_file"]
        assert main.skills == ["git-worktree"]

        assert "researcher" in cfg.agents
        assert cfg.agents["researcher"].provider == "anthropic-main"

        # Validate referential integrity
        assert cfg.validate() == []

    def test_load_empty_file(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cfg = load(config_file)
        assert isinstance(cfg, Config)
        assert cfg.docker.image == "ubuntu:24.04"

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [[[")
        with pytest.raises(Exception):
            load(config_file)

    def test_load_unknown_sections_ignored(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""\
daemon:
  socket: /tmp/test.sock
unknown_section:
  field: value
""")
        # msgspec should ignore unknown fields or raise — depends on strict mode
        # For now just verify it doesn't crash on known fields
        cfg = load(config_file)
        assert cfg.daemon.socket == "/tmp/test.sock"

    def test_load_with_path_object(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("daemon:\n  socket: /tmp/test.sock\n")
        cfg = load(config_file)
        assert cfg.daemon.socket == "/tmp/test.sock"

    def test_load_with_string_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("daemon:\n  socket: /tmp/test.sock\n")
        cfg = load(str(config_file))
        assert cfg.daemon.socket == "/tmp/test.sock"

    def test_load_no_args_works(self) -> None:
        cfg = load()
        assert isinstance(cfg, Config)


class TestLoadMerged:
    """Tests for load_merged() function."""

    def test_merge_overrides(self, tmp_path: Path) -> None:
        base = tmp_path / "config.yaml"
        base.write_text("""\
docker:
  image: ubuntu:24.04
providers:
  openai-default:
    api_type: openai-chat-completion
    default_model: gpt-4o
""")
        overrides = tmp_path / "overrides.yaml"
        overrides.write_text("""\
docker:
  image: python:3.12
""")
        cfg = load_merged(base, overrides)
        assert cfg.docker.image == "python:3.12"
        # Provider should survive merge
        assert "openai-default" in cfg.providers

    def test_missing_base_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_merged(tmp_path / "nonexistent.yaml")

    def test_missing_overrides_ok(self, tmp_path: Path) -> None:
        base = tmp_path / "config.yaml"
        base.write_text("docker:\n  image: ubuntu:24.04\n")
        cfg = load_merged(base, tmp_path / "nonexistent.yaml")
        assert cfg.docker.image == "ubuntu:24.04"


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_config_to_dict(self) -> None:
        cfg = Config()
        data = msgspec.to_builtins(cfg)
        assert "daemon" in data
        assert "docker" in data
        assert "providers" in data
        assert "agents" in data

    def test_config_round_trip(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""\
providers:
  test-provider:
    api_type: openai-chat-completion
    default_model: gpt-3.5-turbo
agents:
  main:
    provider: test-provider
    persona: Test persona
    tools:
      - execute_bash
""")
        cfg = load(config_file)
        assert cfg.providers["test-provider"].default_model == "gpt-3.5-turbo"
        assert cfg.agents["main"].persona == "Test persona"


class TestPathConsistency:
    """Verify all paths use ~/.yagents/ consistently."""

    def test_default_socket_under_yagents(self) -> None:
        cfg = DaemonConfig()
        assert "yagents" in cfg.socket
        assert "yuuagents" not in cfg.socket

    def test_default_skills_under_yagents(self) -> None:
        cfg = SkillsConfig()
        assert all("yagents" in p for p in cfg.paths)
        assert all("yuuagents" not in p for p in cfg.paths)
