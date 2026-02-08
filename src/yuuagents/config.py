"""TOML configuration for yuuagents daemon — msgspec-based."""

from __future__ import annotations

import tomllib
from pathlib import Path

import msgspec

_DEFAULT_CONFIG_PATH = Path("~/.config/yagents/config.toml")


class DaemonConfig(msgspec.Struct, kw_only=True):
    socket: str = "~/.local/run/yagents.sock"


class DockerConfig(msgspec.Struct, kw_only=True):
    """Default container setup.

    The daemon creates one shared container on startup:
    - host ``/`` is bind-mounted read-only to ``/mnt/host``
    - host ``~/.yuuagents/dockers/<container-id>`` is bind-mounted
      read-write to ``/root`` inside the container
    """

    image: str = "ubuntu:24.04"


class LLMConfig(msgspec.Struct, kw_only=True):
    provider: str = "openai"
    api_key_env: str = "OPENAI_API_KEY"
    default_model: str = "gpt-4o"
    base_url: str = ""


class SkillsConfig(msgspec.Struct, kw_only=True):
    paths: list[str] = msgspec.field(default_factory=lambda: ["~/.yagents/skills"])


class TavilyConfig(msgspec.Struct, kw_only=True):
    api_key_env: str = "TAVILY_API_KEY"


class PersonaConfig(msgspec.Struct, kw_only=True):
    system_prompt: str = ""
    tools: list[str] = msgspec.field(default_factory=list)
    skills: list[str] = msgspec.field(default_factory=list)


class Config(msgspec.Struct, kw_only=True):
    daemon: DaemonConfig = msgspec.field(default_factory=DaemonConfig)
    docker: DockerConfig = msgspec.field(default_factory=DockerConfig)
    llm: LLMConfig = msgspec.field(default_factory=LLMConfig)
    skills: SkillsConfig = msgspec.field(default_factory=SkillsConfig)
    tavily: TavilyConfig = msgspec.field(default_factory=TavilyConfig)
    personas: dict[str, PersonaConfig] = msgspec.field(default_factory=dict)

    @property
    def socket_path(self) -> Path:
        return Path(self.daemon.socket).expanduser()


def load(path: str | Path | None = None) -> Config:
    """Load and validate config from TOML. Returns defaults if file missing."""
    p = Path(path) if path else _DEFAULT_CONFIG_PATH.expanduser()
    if not p.exists():
        return Config()
    raw = p.read_bytes()
    # tomllib parses TOML → dict, then msgspec validates + converts
    data = tomllib.loads(raw.decode("utf-8"))
    return msgspec.convert(data, Config)
