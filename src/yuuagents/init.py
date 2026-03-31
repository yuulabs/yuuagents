"""Daemon bootstrap helper — programmatic equivalent of ``yagents install`` + ``yagents up``.

Usage::

    from yuuagents.init import setup

    await setup("/path/to/config.yaml")
    # or
    await setup(my_config_object)

After ``setup()`` returns the daemon is running and the database is ready.
CLI commands like ``yagents list`` / ``yagents status`` work immediately.
"""

from __future__ import annotations

from importlib import import_module
import importlib.metadata
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from yuuagents.service_requirements import service_dependency_message

if TYPE_CHECKING:
    from yuuagents.config import Config


def _load_service_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        config_module = import_module("yuuagents.config")
        yaml_module = import_module("yaml")
        logger = import_module("loguru").logger
        task_persistence = import_module("yuuagents.persistence").SQLitePersistence
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            service_dependency_message("yuuagents.init.setup()", exc)
        ) from exc
    return config_module, yaml_module, logger, task_persistence


# ---------------------------------------------------------------------------
# Docker image helpers (shared with CLI)
# ---------------------------------------------------------------------------


def runtime_image_tag() -> str:
    """Return the versioned runtime image tag, e.g. ``yuuagents-runtime:0.1.0``."""
    try:
        v = importlib.metadata.version("yuuagents")
    except Exception:
        v = "latest"
    return f"yuuagents-runtime:{v}"


def _runtime_dockerfile_text() -> str:
    dockerfile_path = Path(__file__).resolve().parent / "daemon" / "runtime.Dockerfile"
    return dockerfile_path.read_text(encoding="utf-8")


def build_runtime_image(tag: str | None = None) -> bool:
    """Build the yuuagents runtime Docker image.

    Returns ``True`` on success, ``False`` on failure.
    """
    tag = tag or runtime_image_tag()
    dockerfile = _runtime_dockerfile_text()
    with tempfile.TemporaryDirectory(prefix="yagents-runtime-") as td:
        p = Path(td) / "Dockerfile"
        p.write_text(dockerfile, encoding="utf-8")
        result = subprocess.run(
            ["docker", "build", "-t", tag, td],
            capture_output=False,
        )
        if result.returncode != 0:
            return False

        if tag != "yuuagents-runtime:latest":
            subprocess.run(
                ["docker", "tag", tag, "yuuagents-runtime:latest"],
                capture_output=False,
            )

        return True


def _image_exists(image: str) -> bool:
    """Check if a Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _config_uses_docker_tools(cfg: Config) -> bool:
    docker_tools = {"execute_bash", "read_file", "edit_file", "delete_file"}
    return any(
        any(tool in docker_tools for tool in agent.tools)
        for agent in cfg.agents.values()
    )


# ---------------------------------------------------------------------------
# Daemon helpers
# ---------------------------------------------------------------------------


def _daemon_is_running(socket_path: Path) -> bool:
    """Return ``True`` if a daemon is already listening on *socket_path*."""
    import errno
    import socket

    if not socket_path.exists():
        return False
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(0.5)
    try:
        sock.connect(str(socket_path))
    except FileNotFoundError:
        return False
    except OSError as e:
        if e.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.ENOTSOCK, errno.EINVAL):
            return False
        raise
    else:
        return True
    finally:
        sock.close()


def _start_daemon(config_path: Path | None = None) -> None:
    """Start the daemon in the background via ``yagents up -d``.

    If the daemon is already running this is a no-op.
    """
    yagents_bin = shutil.which("yagents")
    if yagents_bin:
        cmd: list[str] = [yagents_bin, "up", "-d"]
    else:
        cmd = [sys.executable, "-m", "yuuagents.cli.main", "up", "-d"]

    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def setup(config: str | Path | Config) -> Config:
    """One-shot daemon/bootstrap initialization.

    Equivalent to running ``yagents install`` followed by ``yagents up -d``.

    Parameters
    ----------
    config:
        **Required.**  One of:

        - A :class:`~yuuagents.config.Config` object.
        - A path (``str`` or ``Path``) to a YAML configuration file.

    Returns
    -------
    Config
        The resolved configuration object.

    The function is **idempotent**: calling it multiple times with the same
    config is safe.  It will skip steps that are already done (directories
    exist, database tables exist, image already built, daemon already
    running).
    """
    config_module, yaml_module, logger, task_persistence = _load_service_runtime()

    # -- resolve config --
    config_file_path: Path | None = None
    if isinstance(config, (str, Path)):
        config_file_path = Path(config).expanduser().resolve()
        if config_file_path.exists():
            cfg = config_module.load(config_file_path)
        else:
            cfg = config_module.load_packaged_default()
            config_file_path = None
    else:
        cfg = config

    errors = cfg.validate()
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # -- Step 1: create directories --
    logger.info("Creating directories ...")
    config_module.YAGENTS_HOME.mkdir(parents=True, exist_ok=True)
    (config_module.YAGENTS_HOME / "dockers").mkdir(exist_ok=True)

    db_path = cfg.sqlite_path
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    # -- Step 2: write config to ~/.yagents/config.yaml --
    import msgspec
    import json

    config_data = json.loads(msgspec.json.encode(cfg))
    config_module.DEFAULT_CONFIG_PATH.write_text(
        yaml_module.dump(config_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    logger.info("Config written to {}", config_module.DEFAULT_CONFIG_PATH)

    # -- Step 3: initialize database --
    logger.info("Initializing database ...")
    persistence = task_persistence(db_url=cfg.db_url)
    await persistence.start()
    await persistence.stop()
    logger.info("Database ready.")

    # -- Step 4: ensure Docker image (best-effort) --
    image = cfg.docker.image
    if _config_uses_docker_tools(cfg):
        if image.startswith("yuuagents-runtime:") or image == "yuuagents-runtime":
            if not _image_exists(image):
                logger.info("Building runtime image {} ...", image)
                ok = build_runtime_image(image)
                if not ok:
                    logger.warning(
                        "Failed to build Docker image {}. "
                        "Daemon startup will continue, but Docker-based tools "
                        "will fail until Docker is installed and reachable.",
                        image,
                    )
                else:
                    logger.info("Image {} built successfully.", image)
            else:
                logger.debug("Image {} already exists.", image)
        else:
            if not _image_exists(image):
                logger.info("Pulling image {} ...", image)
                result = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=True,
                )
                if result.returncode != 0:
                    logger.warning(
                        "Failed to pull Docker image {}. "
                        "Daemon startup will continue, but Docker-based tools "
                        "will fail until Docker is installed and reachable.",
                        image,
                    )
                else:
                    logger.info("Image {} pulled.", image)
    else:
        logger.debug("No configured agents require Docker tools; skipping image setup.")

    # -- Step 5: start daemon if not running --
    socket_path = cfg.socket_path
    if _daemon_is_running(socket_path):
        logger.info("Daemon already running on {}", socket_path)
    else:
        logger.info("Starting daemon ...")
        _start_daemon(config_path=config_file_path)
        # Wait briefly for the daemon to become reachable
        import asyncio

        for _ in range(20):
            await asyncio.sleep(0.5)
            if _daemon_is_running(socket_path):
                break
        if _daemon_is_running(socket_path):
            logger.info("Daemon started on {}", socket_path)
        else:
            logger.warning(
                "Daemon may still be starting up. "
                "Check with `yagents list` in a few seconds."
            )

    return cfg
