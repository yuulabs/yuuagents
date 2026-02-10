"""Tests for yagents CLI Docker checks."""

from __future__ import annotations

import subprocess

import pytest

from yuuagents.cli.main import _docker_check


class _Res:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_docker_check_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], capture_output: bool, text: bool, timeout: int):
        raise FileNotFoundError

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, detail = _docker_check()
    assert ok is False
    assert "not found" in detail.lower()


def test_docker_check_permission_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], capture_output: bool, text: bool, timeout: int):
        if cmd == ["docker", "--version"]:
            return _Res(0, stdout="Docker version 99.0.0\n")
        assert cmd == ["docker", "version", "--format", "{{.Server.Version}}"]
        return _Res(
            1,
            stderr="Got permission denied while trying to connect to the Docker daemon socket\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, detail = _docker_check()
    assert ok is False
    assert "permission denied" in detail.lower()


def test_docker_check_daemon_not_running(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], capture_output: bool, text: bool, timeout: int):
        if cmd == ["docker", "--version"]:
            return _Res(0, stdout="Docker version 99.0.0\n")
        assert cmd == ["docker", "version", "--format", "{{.Server.Version}}"]
        return _Res(
            1,
            stderr="Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?\n",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, detail = _docker_check()
    assert ok is False
    assert "daemon not reachable" in detail.lower()


def test_docker_check_info_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(cmd: list[str], capture_output: bool, text: bool, timeout: int):
        if cmd == ["docker", "--version"]:
            return _Res(0, stdout="Docker version 99.0.0\n")
        assert cmd == ["docker", "version", "--format", "{{.Server.Version}}"]
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok, detail = _docker_check()
    assert ok is False
    assert "timed out" in detail.lower()
