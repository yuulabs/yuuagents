from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import msgspec
import yuullm
import pytest

from yuuagents import cli_entry, init as init_module
from yuuagents.config import Config, SnapshotConfig, load_packaged_default_yaml
from yuuagents.daemon.manager import AgentManager
from yuuagents.core.flow import AgentState
from yuuagents.persistence import RestoredTask
from yuuagents.input import conversation_input_from_text
from yuuagents.types import TaskRequest


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


@pytest.mark.asyncio
async def test_agent_manager_start_does_not_eagerly_start_docker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDocker:
        workdir = "/tmp/no-docker"

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakePersistence:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    fake_persistence = FakePersistence(db_url="")
    manager = AgentManager(config=Config(), docker=FakeDocker(), persistence=fake_persistence)
    await manager.setup()
    await manager.stop()


@pytest.mark.asyncio
async def test_agent_manager_does_not_restore_when_snapshot_restore_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDocker:
        workdir = "/tmp/no-docker"

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakePersistence:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def load_unfinished(self) -> list[RestoredTask]:
            raise AssertionError("restore should not run when restore_on_start is false")

    fake_persistence = FakePersistence(db_url="")
    manager = AgentManager(
        config=Config(snapshot=SnapshotConfig(enabled=True, restore_on_start=False)),
        docker=FakeDocker(),
        persistence=fake_persistence,
    )
    await manager.setup()
    await manager.stop()


@pytest.mark.asyncio
async def test_docker_tools_fail_cleanly_when_docker_is_unavailable() -> None:
    class FakeDocker:
        workdir = "/tmp/no-docker"

        async def stop(self) -> None:
            return None

        async def resolve(
            self,
            *,
            task_id: str = "",
            container: str = "",
            image: str = "",
        ) -> str:
            del task_id, container, image
            raise RuntimeError("docker daemon unreachable")

    req = TaskRequest(
        agent="main",
        input=conversation_input_from_text("hello"),
        tools=["execute_bash"],
    )
    manager = AgentManager(config=Config(), docker=FakeDocker())

    with pytest.raises(ValueError, match="docker tools requested but Docker is unavailable"):
        await manager._build_root_session(task_id="task-1", req=req, delegate_depth=0)


@pytest.mark.asyncio
async def test_agent_manager_restores_unfinished_tasks_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import yuuagents.daemon.manager as manager_module

    class FakeDocker:
        workdir = "/tmp/no-docker"

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeProvider:
        def __init__(self) -> None:
            self.called = False

        @property
        def provider(self) -> str:
            return "fake"

        @property
        def api_type(self) -> str:
            return "fake"

        async def stream(
            self,
            messages: list[yuullm.Message],
            *,
            model: str | None = None,
            tools: list[dict] | None = None,
            **kwargs: object,
        ) -> yuullm.StreamResult:
            del messages, model, tools, kwargs
            if self.called:
                raise AssertionError("restore should only need one LLM turn")
            self.called = True

            async def _iter() -> AsyncIterator[yuullm.StreamItem]:
                yield yuullm.Response(item={"type": "text", "text": "restored done"})

            return _iter(), yuullm.Store()

    class FakePersistence:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url
            self.terminal_updates: list[tuple[str, str]] = []
            self.snapshots: list[object] = []

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def load_unfinished(self) -> list[RestoredTask]:
            state = AgentState(
                messages=(
                    yuullm.system("restored system"),
                    yuullm.user("hello"),
                ),
                total_usage=None,
                total_cost_usd=0.0,
                rounds=1,
                conversation_id=None,
            )
            return [
                RestoredTask(
                    task_id="task-restore",
                    agent_id="main",
                    persona="restored persona",
                    input=conversation_input_from_text("hello"),
                    system_prompt="restored system",
                    model="fake-model",
                    tools=["sleep"],
                    docker_container="",
                    status=manager_module.AgentStatus.RUNNING,
                    created_at=datetime.now(timezone.utc),
                    head_turn=1,
                    state=state,
                )
            ]

        async def save_snapshot(
            self,
            *,
            task_id: str,
            turn: int,
            state: object,
            status: object,
        ) -> None:
            self.snapshots.append((task_id, turn, state, status))

        async def update_task_terminal(
            self,
            *,
            task_id: str,
            status: manager_module.AgentStatus,
            error_json: bytes | None = None,
        ) -> None:
            del error_json
            self.terminal_updates.append((task_id, status.value))

    fake_provider = FakeProvider()
    fake_persistence = FakePersistence(db_url="")

    def fake_make_llm(
        self: AgentManager,
        agent_name: str,
        model_override: str = "",
    ) -> yuullm.YLLMClient:
        del self, agent_name, model_override
        return yuullm.YLLMClient(provider=fake_provider, default_model="fake-model")

    monkeypatch.setattr(manager_module.AgentManager, "_make_llm", fake_make_llm)

    manager = AgentManager(
        config=Config(snapshot=SnapshotConfig(enabled=True, restore_on_start=True)),
        docker=FakeDocker(),
        persistence=fake_persistence,
    )

    try:
        await manager.setup()
        assert manager._pool is not None
        await asyncio.wait_for(manager._pool._tasks["task-restore"], timeout=1)

        assert manager._pool._sessions["task-restore"].status == manager_module.AgentStatus.DONE
        assert manager._pool._sessions["task-restore"].steps >= 2
        assert fake_persistence.snapshots
    finally:
        await manager.stop()
