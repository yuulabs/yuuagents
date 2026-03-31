"""YAML configuration for yagents daemon — msgspec-based."""

from __future__ import annotations

import copy
from importlib import resources
from pathlib import Path
from typing import Any

import msgspec
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

YAGENTS_HOME = Path("~/.yagents").expanduser()
DEFAULT_CONFIG_PATH = YAGENTS_HOME / "config.yaml"

# Packaged default config template.
_PACKAGE_DEFAULT_CONFIG_NAME = "config.default.yaml"

# Project-level config files (relative to repo root).
_PROJECT_CONFIG_NAME = "config.example.yaml"
_PROJECT_OVERRIDES_NAME = "config.overrides.yaml"

# ---------------------------------------------------------------------------
# Config structs
# ---------------------------------------------------------------------------


class DaemonConfig(msgspec.Struct, kw_only=True):
    socket: str = "~/.yagents/yagents.sock"
    log_level: str = "info"


class DockerConfig(msgspec.Struct, kw_only=True):
    """Default container setup."""

    image: str = "yuuagents-runtime:latest"


class TavilyConfig(msgspec.Struct, kw_only=True):
    api_key_env: str = "TAVILY_API_KEY"


class DbConfig(msgspec.Struct, kw_only=True):
    url: str = "sqlite+aiosqlite:///~/.yagents/tasks.sqlite3"


class YuuTraceConfig(msgspec.Struct, kw_only=True):
    db_path: str = "~/.yagents/traces.db"
    ui_port: int = 8080
    server_port: int = 4318


class SnapshotConfig(msgspec.Struct, kw_only=True):
    enabled: bool = False
    restore_on_start: bool = False


class PricingEntry(msgspec.Struct, kw_only=True):
    """Per-model pricing override (USD per million tokens)."""

    model: str
    input_mtok: float = 0.0
    output_mtok: float = 0.0
    cache_read_mtok: float = 0.0
    cache_write_mtok: float = 0.0


class ProviderConfig(msgspec.Struct, kw_only=True):
    """LLM provider configuration.

    ``api_type`` selects the wire protocol. One of:
    - ``"openai-chat-completion"``
    - ``"openai-responses"``
    - ``"anthropic-messages"``

    Provider-specific fields (``base_url``, ``organization``) are passed
    through to the underlying SDK.
    """

    api_type: str = "openai-chat-completion"
    api_key_env: str = "OPENAI_API_KEY"
    default_model: str = "gpt-4o"
    # Provider-specific
    base_url: str = ""
    organization: str = ""  # OpenAI only
    # Inline pricing overrides
    pricing: list[PricingEntry] = msgspec.field(default_factory=list)


class AgentEntry(msgspec.Struct, kw_only=True):
    """Per-agent configuration.

    Each agent references a named provider from ``providers:`` and may
    override the model, persona, and tools.
    """

    description: str
    provider: str = ""
    model: str = ""
    persona: str = ""
    subagents: list[str] = msgspec.field(default_factory=list)
    tools: list[str] = msgspec.field(default_factory=list)


class Config(msgspec.Struct, kw_only=True):
    daemon: DaemonConfig = msgspec.field(default_factory=DaemonConfig)
    db: DbConfig = msgspec.field(default_factory=DbConfig)
    yuutrace: YuuTraceConfig = msgspec.field(default_factory=YuuTraceConfig)
    snapshot: SnapshotConfig = msgspec.field(default_factory=SnapshotConfig)
    docker: DockerConfig = msgspec.field(default_factory=DockerConfig)
    tavily: TavilyConfig = msgspec.field(default_factory=TavilyConfig)
    providers: dict[str, ProviderConfig] = msgspec.field(default_factory=dict)
    agents: dict[str, AgentEntry] = msgspec.field(default_factory=dict)

    @property
    def socket_path(self) -> Path:
        return Path(self.daemon.socket).expanduser()

    @property
    def db_url(self) -> str:
        sp = self.sqlite_path
        if sp is not None:
            return "sqlite+aiosqlite:///" + str(sp)
        return self.db.url

    @property
    def sqlite_path(self) -> Path | None:
        """Return the resolved SQLite file :class:`Path`, or ``None`` if the
        database URL does not point to a local SQLite file."""
        prefix = "sqlite+aiosqlite:///"
        if not self.db.url.startswith(prefix):
            return None
        raw = self.db.url[len(prefix) :]
        return Path(raw).expanduser().resolve()

    def validate(self) -> list[str]:
        """Check referential integrity.  Returns a list of error messages."""
        errors: list[str] = []
        if not self.db.url:
            errors.append("db.url must not be empty")
        if not self.yuutrace.db_path:
            errors.append("yuutrace.db_path must not be empty")
        if not (1 <= self.yuutrace.ui_port <= 65535):
            errors.append("yuutrace.ui_port must be in range 1..65535")
        if not (1 <= self.yuutrace.server_port <= 65535):
            errors.append("yuutrace.server_port must be in range 1..65535")
        if self.snapshot.restore_on_start and not self.snapshot.enabled:
            errors.append(
                "snapshot.restore_on_start requires snapshot.enabled to be true"
            )
        for agent_name, entry in self.agents.items():
            if not entry.description.strip():
                errors.append(f"agents.{agent_name}.description must not be empty")
            if entry.provider and entry.provider not in self.providers:
                errors.append(
                    f"agents.{agent_name}.provider references unknown "
                    f"provider {entry.provider!r}"
                )
            for sub in entry.subagents:
                if sub == "*":
                    continue
                if sub == agent_name:
                    errors.append(
                        f"agents.{agent_name}.subagents must not include itself"
                    )
                    continue
                if sub not in self.agents:
                    errors.append(
                        f"agents.{agent_name}.subagents references unknown agent {sub!r}"
                    )
        return errors


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into a **copy** of *base*.

    - dict values are merged recursively.
    - All other types in *override* replace the value in *base*.
    """
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def _load_yaml_mapping(text: str, *, source: str) -> dict[str, Any]:
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a YAML mapping at the top level")
    return data


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a mapping."""
    p = Path(path)
    return _load_yaml_mapping(p.read_text(encoding="utf-8"), source=str(p))


def load_packaged_default_yaml() -> dict[str, Any]:
    """Load the packaged default config template."""
    ref = resources.files(__package__).joinpath(_PACKAGE_DEFAULT_CONFIG_NAME)
    return _load_yaml_mapping(ref.read_text(encoding="utf-8"), source=_PACKAGE_DEFAULT_CONFIG_NAME)


def find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from *start* (default: cwd) looking for a directory that
    contains ``config.example.yaml``.  Returns ``None`` if not found.
    """
    current = (start or Path.cwd()).resolve()
    for parent in [current, *current.parents]:
        if (parent / _PROJECT_CONFIG_NAME).exists():
            return parent
    return None


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load(path: str | Path | None = None) -> Config:
    """Load and validate config from YAML.  Returns defaults if file missing."""
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        return Config()
    data = load_yaml(p)
    if not data:
        return Config()
    return msgspec.convert(data, Config)


def load_packaged_default() -> Config:
    """Load the bundled default config template as a Config object."""
    return msgspec.convert(load_packaged_default_yaml(), Config)


def load_merged(
    base_path: str | Path,
    overrides_path: str | Path | None = None,
) -> Config:
    """Load *base_path*, optionally deep-merge *overrides_path* on top,
    and return the resulting ``Config``.
    """
    base_p = Path(base_path)
    if not base_p.exists():
        raise FileNotFoundError(f"Base config not found: {base_p}")

    base_data = load_yaml(base_p)

    if overrides_path:
        over_p = Path(overrides_path)
        if over_p.exists():
            over_data = load_yaml(over_p)
            base_data = _deep_merge(base_data, over_data)

    return msgspec.convert(base_data, Config)
