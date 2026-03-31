from __future__ import annotations

import pytest

from yuuagents import cli_entry, init as init_module
from yuuagents.config import Config


def test_trace_db_default_is_project_local() -> None:
    assert Config().yuutrace.db_path == "./.yagents/traces.db"


def test_cli_entry_reports_missing_service_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import_module(name: str):
        if name == "yuuagents.cli.main":
            raise ModuleNotFoundError("No module named 'click'", name="click")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(cli_entry, "import_module", fake_import_module)

    with pytest.raises(SystemExit, match=r"Install `yuuagents\[daemon\]`"):
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

    with pytest.raises(RuntimeError, match=r"Install `yuuagents\[daemon\]`"):
        await init_module.setup("config.yaml")
