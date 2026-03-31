from __future__ import annotations

import pytest
from yuuagents import cli_entry, init as init_module
from yuuagents.config import Config
from yuuagents.daemon.manager import AgentManager
from yuuagents.input import conversation_input_from_text
from yuuagents.types import TaskRequest


def test_trace_db_default_is_project_local() -> None:
    assert Config().yuutrace.db_path == "./.yagents/traces.db"


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
    import yuuagents.daemon.manager as manager_module

    class FakeDocker:
        workdir = "/tmp/no-docker"

        async def start(self) -> None:
            raise AssertionError("docker.start should not be called during daemon startup")

        async def stop(self) -> None:
            return None

    class FakePersistence:
        def __init__(self, db_url: str) -> None:
            self.db_url = db_url

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    class FakeWriter:
        def __init__(self, persistence: FakePersistence) -> None:
            self.persistence = persistence

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

    monkeypatch.setattr(manager_module, "TaskPersistence", FakePersistence)
    monkeypatch.setattr(manager_module, "TaskWriter", FakeWriter)

    manager = AgentManager(config=Config(), docker=FakeDocker())
    await manager.start()
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
