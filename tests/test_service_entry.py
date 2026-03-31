from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import msgspec
import pytest

from yuuagents import cli_entry, init as init_module
from yuuagents.config import Config, load_packaged_default_yaml


def test_packaged_default_config_is_available_without_repo_checkout() -> None:
    data = load_packaged_default_yaml()
    cfg = msgspec.convert(data, Config)

    assert cfg.yuutrace.db_path == "~/.yagents/traces.db"
    assert cfg.snapshot.enabled is False
    assert cfg.snapshot.restore_on_start is False
    assert "main" in cfg.agents


def test_trace_db_default_is_home_local() -> None:
    assert Config().yuutrace.db_path == "~/.yagents/traces.db"


def test_cli_entry_reports_missing_service_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str):
        if name == "yuuagents.cli.main":
            raise ModuleNotFoundError("No module named 'click'", name="click")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(cli_entry, "import_module", fake_import_module)

    with pytest.raises(SystemExit, match=r"Reinstall `yuuagents`"):
        cli_entry.main()


@pytest.mark.asyncio
async def test_init_setup_reports_missing_service_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import_module(name: str):
        if name == "yuuagents.config":
            raise ModuleNotFoundError("No module named 'yaml'", name="yaml")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(init_module, "import_module", fake_import_module)

    with pytest.raises(RuntimeError, match=r"Reinstall `yuuagents`"):
        await init_module.setup("config.yaml")


@pytest.mark.asyncio
async def test_init_setup_uses_packaged_defaults_for_missing_config_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import yuuagents.config as config_module

    config_path = tmp_path / "missing-config.yaml"
    config_home = tmp_path / ".yagents"
    fake_cfg = Config(
        daemon=config_module.DaemonConfig(socket=str(config_home / "yagents.sock")),
        db=config_module.DbConfig(
            url=f"sqlite+aiosqlite:///{config_home / 'tasks.sqlite3'}"
        ),
        yuutrace=config_module.YuuTraceConfig(
            db_path=str(config_home / "traces.db"),
        ),
    )

    class FakePersistence:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    fake_config_module = SimpleNamespace(
        YAGENTS_HOME=config_home,
        DEFAULT_CONFIG_PATH=config_home / "config.yaml",
        load=lambda path: pytest.fail(f"load() should not run for missing path: {path}"),
        load_packaged_default=lambda: fake_cfg,
    )
    fake_yaml_module = SimpleNamespace(
        dump=lambda data, **kwargs: "yaml",
    )
    fake_logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    fake_persistence_module = SimpleNamespace(SQLitePersistence=FakePersistence)
    recorded: dict[str, Path | None] = {}

    def fake_import_module(name: str):
        if name == "yuuagents.config":
            return fake_config_module
        if name == "yaml":
            return fake_yaml_module
        if name == "loguru":
            return SimpleNamespace(logger=fake_logger)
        if name == "yuuagents.persistence":
            return fake_persistence_module
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(init_module, "import_module", fake_import_module)
    monkeypatch.setattr(init_module, "_config_uses_docker_tools", lambda cfg: False)
    monkeypatch.setattr(init_module, "_daemon_is_running", lambda socket_path: False)
    monkeypatch.setattr(
        init_module,
        "_start_daemon",
        lambda *, config_path=None: recorded.setdefault("config_path", config_path),
    )

    result = await init_module.setup(config_path)

    assert result == fake_cfg
    assert recorded["config_path"] is None
    assert fake_config_module.DEFAULT_CONFIG_PATH.exists()


def test_cli_supports_version_flag_when_service_dependencies_are_available() -> None:
    pytest.importorskip("click")
    pytest.importorskip("httpx")
    pytest.importorskip("yaml")

    from click.testing import CliRunner
    from yuuagents.cli.main import cli

    result = CliRunner().invoke(cli, ["--version"])

    assert result.exit_code == 0
    assert "yagents, version " in result.output
