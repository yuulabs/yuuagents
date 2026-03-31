from __future__ import annotations

from typing import Any

import pytest

from yuuagents.daemon.docker import DockerManager


class _FixedUUID:
    hex = "softtimeouttoken"


@pytest.mark.asyncio
async def test_exec_terminal_soft_timeout_creates_resumable_pending_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = DockerManager()

    async def _noop(self: DockerManager, *args: Any, **kwargs: Any) -> None:
        del self, args, kwargs
        return None

    async def fake_exec_with_shell(
        self: DockerManager,
        container_id: str,
        shell: str,
        command: str,
        timeout: int,
        *,
        user: str | None = None,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> str:
        del self, container_id, shell, timeout, user, workdir, environment
        if "capture-pane" in command:
            return (
                "__YAGENTS_BEGIN__=softtimeouttoken\n"
                "done\n"
                "__YAGENTS_END__=softtimeouttoken __YAGENTS_EXIT_CODE__=0\n"
            )
        return ""

    async def fake_poll_tmux_until(
        self: DockerManager,
        container_id: str,
        session_name: str,
        begin: str,
        end_prefix: str,
        deadline: float,
    ) -> tuple[str, bool]:
        del self, container_id, session_name, end_prefix, deadline
        return (f"{begin}\npartial output", False)

    monkeypatch.setattr("yuuagents.daemon.docker.uuid.uuid4", lambda: _FixedUUID())
    monkeypatch.setattr(DockerManager, "_ensure_started", _noop)
    monkeypatch.setattr(DockerManager, "_ensure_required_tooling", _noop)
    monkeypatch.setattr(DockerManager, "_ensure_tmux_session", _noop)
    monkeypatch.setattr(DockerManager, "_exec_with_shell", fake_exec_with_shell)
    monkeypatch.setattr(DockerManager, "_poll_tmux_until", fake_poll_tmux_until)

    result = await manager.exec_terminal(
        "cid",
        "session-1",
        "sleep 5",
        timeout=30,
        soft_timeout=1,
    )

    assert result == "[SOFT_TIMEOUT] Command is still running.\npartial output"
    assert manager.get_pending("cid", "session-1") is not None

    resumed = await manager.resume_pending("cid", "session-1", timeout=30)

    assert resumed == "done"
    assert manager.get_pending("cid", "session-1") is None
