"""Systemd user service management for yagents."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_UNIT_NAME = "yagents.service"
_UNIT_DIR = Path("~/.config/systemd/user").expanduser()
_UNIT_PATH = _UNIT_DIR / _UNIT_NAME

_UNIT_TEMPLATE = """\
[Unit]
Description=yagents — agent framework daemon
After=docker.service

[Service]
Type=simple
ExecStart={exec_start}
Restart=on-failure
RestartSec=5
Environment=PATH={path_env}

[Install]
WantedBy=default.target
"""


def install() -> str:
    """Generate and install a systemd user service unit.

    Returns the path to the installed unit file.

    Raises
    ------
    FileNotFoundError
        If the ``yagents`` executable cannot be found on ``$PATH``.
    RuntimeError
        If ``systemctl --user`` commands fail.
    """
    yagents_bin = shutil.which("yagents")
    if not yagents_bin:
        raise FileNotFoundError(
            "Cannot find 'yagents' on $PATH. "
            "Make sure the package is installed (e.g. `uv pip install -e .`)."
        )

    import os

    path_env = os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")

    unit_content = _UNIT_TEMPLATE.format(
        exec_start=f"{yagents_bin} start",
        path_env=path_env,
    )

    _UNIT_DIR.mkdir(parents=True, exist_ok=True)
    _UNIT_PATH.write_text(unit_content, encoding="utf-8")

    # Reload systemd and enable the service
    _systemctl("daemon-reload")
    _systemctl("enable", _UNIT_NAME)
    _systemctl("start", _UNIT_NAME)

    return str(_UNIT_PATH)


def uninstall() -> None:
    """Stop and disable the systemd user service, then remove the unit file.

    Raises
    ------
    RuntimeError
        If ``systemctl --user`` commands fail (other than "not loaded").
    """
    # Stop and disable (ignore errors if not loaded)
    _systemctl("stop", _UNIT_NAME, ignore_errors=True)
    _systemctl("disable", _UNIT_NAME, ignore_errors=True)

    if _UNIT_PATH.exists():
        _UNIT_PATH.unlink()

    _systemctl("daemon-reload", ignore_errors=True)


def status() -> str:
    """Return the current service status as a string."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "status", _UNIT_NAME],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout or result.stderr or "unknown"
    except FileNotFoundError, subprocess.TimeoutExpired:
        return "systemctl not available"


def _systemctl(*args: str, ignore_errors: bool = False) -> None:
    """Run ``systemctl --user <args>``."""
    cmd = ["systemctl", "--user", *args]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 and not ignore_errors:
            raise RuntimeError(
                f"systemctl failed: {' '.join(cmd)}\nstderr: {result.stderr.strip()}"
            )
    except FileNotFoundError:
        if not ignore_errors:
            raise RuntimeError(
                "systemctl not found. "
                "systemd user services require a Linux system with systemd."
            )
