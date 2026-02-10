"""CLI entry point — ``yagents`` command."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
import shutil

import click
import yaml

from yuuagents.config import (
    DEFAULT_CONFIG_PATH,
    YAGENTS_HOME,
    Config,
    _PROJECT_CONFIG_NAME,
    _PROJECT_OVERRIDES_NAME,
    _deep_merge,
    find_project_root,
    load as load_config,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from yuuagents.cli.client import YAgentsClient

from yuuagents.cli.client import DaemonNotRunningError


_DOTENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Local config file names (looked up in cwd only).
_LOCAL_CONFIG_NAME = "config.yaml"
_LOCAL_OVERRIDES_NAME = "config.overrides.yaml"


def _local_config() -> Path | None:
    """Return ``cwd/config.yaml`` if it exists, else ``None``."""
    p = Path.cwd() / _LOCAL_CONFIG_NAME
    return p if p.is_file() else None


def _local_overrides() -> Path | None:
    """Return ``cwd/config.overrides.yaml`` if it exists, else ``None``."""
    p = Path.cwd() / _LOCAL_OVERRIDES_NAME
    return p if p.is_file() else None


def _find_dotenv(start_dir: Path) -> Path | None:
    home = Path.home().resolve()
    cur = start_dir.resolve()
    while True:
        candidate = cur / ".env"
        if candidate.exists() and candidate.is_file():
            return candidate
        if cur == home or cur.parent == cur:
            return None
        cur = cur.parent


def _load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        assert sep == "=", f"Invalid .env line (missing '='): {raw_line!r}"
        key = key.strip()
        assert key, f"Invalid .env line (empty key): {raw_line!r}"
        assert _DOTENV_KEY_RE.match(key), f"Invalid .env key: {key!r}"
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def _socket(ctx: click.Context) -> str:
    return str(Path(ctx.obj).expanduser())


def _client(ctx: click.Context) -> YAgentsClient:
    from yuuagents.cli.client import YAgentsClient

    return YAgentsClient(_socket(ctx))


@click.group()
@click.option(
    "--socket",
    default=None,
    help="Path to daemon Unix socket (default: from config)",
)
@click.pass_context
def cli(ctx: click.Context, socket: str | None) -> None:
    """yagents — minimal agent framework."""
    if socket:
        ctx.obj = socket
    else:
        cfg = load_config()
        ctx.obj = str(cfg.socket_path)


# ── Install ──


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml (fully replaces defaults — requires confirmation)",
)
@click.option(
    "--overrides",
    "overrides_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.overrides.yaml (deep-merged on top)",
)
@click.option(
    "--project-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Project root containing config.example.yaml (default: auto-detect)",
)
@click.option(
    "-s",
    "--systemd",
    is_flag=True,
    default=False,
    help="Register yagents as a systemd user service",
)
@click.pass_context
def install(
    ctx: click.Context,
    config_path: str | None,
    overrides_path: str | None,
    project_dir: str | None,
    systemd: bool,
) -> None:
    """One-time install: install config, create directories, init database, pull Docker.

    \b
    Config resolution order:
      1. Start with defaults (config.example.yaml from project root).
      2. If --overrides is given, deep-merge it on top.
      3. If --config is given, it FULLY REPLACES the result (with confirmation).

    \b
    This command will:
      1. Write merged config to ~/.yagents/config.yaml.
      2. Create required directories (~/.yagents/skills, ~/.yagents/dockers, db parent).
      3. Initialize the database (create tables).
      4. Pull the Docker image specified in the config.

    If ``--systemd`` is provided, it will also:
      5. Register yagents as a systemd user service.
    """
    total_steps = 5 if systemd else 4
    runtime_tag = _runtime_image_tag()
    # -- Step 0: resolve config sources --
    if config_path:
        # User provided a full config — requires confirmation
        click.echo(f"Config file:   {config_path}")
        click.echo("WARNING: --config fully replaces the default configuration.")
        if not click.confirm("Are you sure you want to proceed?"):
            click.echo("Aborted.")
            sys.exit(0)

        user_config_p = Path(config_path)
        user_data = yaml.safe_load(user_config_p.read_text(encoding="utf-8")) or {}

        # Apply overrides on top if given
        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            user_data = _deep_merge(user_data, over_data)
            click.echo(f"Overrides:     {overrides_path}")

        merged_data = user_data
    else:
        # --- Try cwd first: config.yaml / config.overrides.yaml ---
        local_cfg = _local_config()
        local_over = _local_overrides()

        if local_cfg is not None:
            click.echo(f"Found local:   {local_cfg}")
            base_data = yaml.safe_load(local_cfg.read_text(encoding="utf-8")) or {}

            if local_over is not None:
                click.echo(f"Found local:   {local_over}")
                over_data = yaml.safe_load(local_over.read_text(encoding="utf-8")) or {}
                base_data = _deep_merge(base_data, over_data)

            # Apply CLI --overrides on top
            if overrides_path:
                over_p = Path(overrides_path)
                over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
                base_data = _deep_merge(base_data, over_data)
                click.echo(f"CLI overrides: {overrides_path}")

            merged_data = base_data
        else:
            # Fallback: auto-detect project root for config.example.yaml
            root: Path | None
            if project_dir:
                root = Path(project_dir)
            else:
                root = find_project_root()

            if root is None or not (root / _PROJECT_CONFIG_NAME).exists():
                click.echo(
                    f"Error: cannot find {_LOCAL_CONFIG_NAME} in current directory "
                    f"or {_PROJECT_CONFIG_NAME} in project root. "
                    "Run this command from the project directory or pass --project-dir.",
                    err=True,
                )
                sys.exit(1)
            assert root is not None

            base_path = root / _PROJECT_CONFIG_NAME
            click.echo(f"Project root:  {root}")
            click.echo(f"Base config:   {base_path}")

            base_data = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}

            # Check for project-level overrides
            proj_overrides = root / _PROJECT_OVERRIDES_NAME
            if proj_overrides.exists():
                over_data = (
                    yaml.safe_load(proj_overrides.read_text(encoding="utf-8")) or {}
                )
                base_data = _deep_merge(base_data, over_data)
                click.echo(f"Overrides:     {proj_overrides}")
            else:
                click.echo(
                    f"Overrides:     (none — {_PROJECT_OVERRIDES_NAME} not found)"
                )

            # Apply CLI --overrides on top
            if overrides_path:
                over_p = Path(overrides_path)
                over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
                base_data = _deep_merge(base_data, over_data)
                click.echo(f"CLI overrides: {overrides_path}")

            merged_data = base_data

    if not config_path:
        merged_data = _deep_merge(merged_data, {"docker": {"image": runtime_tag}})

    # -- Step 1: write config --
    click.echo()
    click.echo(f"[1/{total_steps}] Installing configuration ...")

    YAGENTS_HOME.mkdir(parents=True, exist_ok=True)
    DEFAULT_CONFIG_PATH.write_text(
        yaml.dump(merged_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    click.echo(f"  -> {DEFAULT_CONFIG_PATH}")

    # Load the config for subsequent steps
    cfg = load_config()

    # Validate
    errors = cfg.validate()
    if errors:
        click.echo("  WARNING: configuration has validation errors:", err=True)
        for err in errors:
            click.echo(f"    - {err}", err=True)

    # -- Step 2: create directories --
    click.echo()
    click.echo(f"[2/{total_steps}] Creating directories ...")

    dirs_to_create = [
        YAGENTS_HOME / "skills",
        YAGENTS_HOME / "dockers",
    ]
    for sp in cfg.skills.paths:
        dirs_to_create.append(Path(sp).expanduser())

    db_path = cfg.sqlite_path
    if db_path is not None:
        dirs_to_create.append(db_path.parent)

    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)
        click.echo(f"  -> {d}")

    # -- Step 3: initialize database --
    click.echo()
    click.echo(f"[3/{total_steps}] Initializing database ...")
    click.echo(f"  URL: {cfg.db_url}")
    _init_database(cfg.db_url)
    click.echo("  -> Database tables created.")

    # -- Step 4: Docker --
    click.echo()
    click.echo(f"[4/{total_steps}] Setting up Docker ...")

    image = cfg.docker.image
    click.echo(f"  Image: {image}")

    docker_ok, docker_detail = _docker_check()
    if not docker_ok:
        click.echo(
            "  WARNING: Docker is not usable from this environment.\n"
            f"  Details: {docker_detail}\n"
            "  yagents requires a reachable Docker daemon.\n"
            "  Install Docker Engine: https://docs.docker.com/engine/install/\n"
            "  Common fixes:\n"
            "    - Start the daemon: sudo systemctl start docker\n"
            "    - Fix permissions: sudo usermod -aG docker $USER (then re-login)\n"
            "    - If you use rootless Docker, ensure DOCKER_HOST is set correctly\n"
            f"  Then run `docker pull {image}` manually if needed.",
            err=True,
        )
    else:
        if (
            image == runtime_tag
            or image.startswith("yuuagents-runtime:")
            or image == "yuuagents-runtime"
        ):
            if _image_exists(image):
                click.echo(f"  Image {image} already available locally.")
            else:
                click.echo(f"  Building {image} (this may take a minute) ...")
                ok = _build_runtime_image(image)
                if not ok:
                    click.echo(
                        f"  WARNING: Failed to build {image}. "
                        "You can retry by re-running `yagents install`.",
                        err=True,
                    )
                else:
                    click.echo(f"  Image {image} built successfully.")
        else:
            if _image_exists(image):
                click.echo(f"  Image {image} already available locally.")
            else:
                click.echo(f"  Pulling {image} (this may take a minute) ...")
                result = subprocess.run(
                    ["docker", "pull", image],
                    capture_output=False,
                )
                if result.returncode != 0:
                    click.echo(
                        f"  WARNING: Failed to pull {image}. "
                        f"You can retry with `docker pull {image}`.",
                        err=True,
                    )
                else:
                    click.echo(f"  Image {image} pulled successfully.")

    if systemd:
        click.echo()
        click.echo(f"[5/{total_steps}] Registering systemd user service ...")

        try:
            from yuuagents.cli.service import install as install_service

            unit_path = install_service()
            click.echo(f"  Service installed: {unit_path}")
            click.echo("  Service enabled and started.")
        except FileNotFoundError as e:
            click.echo(f"  WARNING: {e}", err=True)
            click.echo("  You can start the daemon manually: yagents up", err=True)
        except RuntimeError as e:
            click.echo(f"  WARNING: {e}", err=True)
            click.echo("  You can start the daemon manually: yagents up", err=True)

    # -- Done --
    click.echo()
    click.echo("Install complete!")
    click.echo()

    # Show next steps based on first provider
    if cfg.providers:
        first_provider = next(iter(cfg.providers.values()))
        click.echo("Next steps:")
        click.echo(
            f"  1. Set your LLM API key:  export {first_provider.api_key_env}=sk-..."
        )
        click.echo(
            '  2. Run an agent:          yagents run --agent main --task "hello world"'
        )
    else:
        click.echo("Next steps:")
        click.echo("  1. Configure at least one provider in ~/.yagents/config.yaml")
        click.echo("  2. Set your LLM API key")
        click.echo(
            '  3. Run an agent:          yagents run --agent main --task "hello world"'
        )


def _init_database(db_url: str) -> None:
    """Create database tables (idempotent).

    Uses :class:`TaskPersistence` to run ``CREATE TABLE IF NOT EXISTS``
    via SQLAlchemy's ``metadata.create_all``.
    """
    from yuuagents.persistence import TaskPersistence

    async def _run() -> None:
        p = TaskPersistence(db_url=db_url)
        await p.start()
        await p.stop()

    asyncio.run(_run())


def _check_db_path_unchanged(new_cfg: Config) -> None:
    """Abort if the database path in *new_cfg* differs from the installed config.

    When the user changes ``db.url`` to point to a different file, the old
    database is silently orphaned on disk.  To prevent this kind of disk
    leak we refuse to start and ask the user to ``uninstall`` + ``install``
    so that the old data is cleaned up first.
    """
    if not DEFAULT_CONFIG_PATH.exists():
        # No installed config yet — nothing to compare against.
        return

    installed_cfg = load_config()  # loads from DEFAULT_CONFIG_PATH
    old_path = installed_cfg.sqlite_path
    new_path = new_cfg.sqlite_path

    # Both non-SQLite (or identical) — fine.
    if old_path == new_path:
        return

    # One is SQLite and the other is not, or they point to different files.
    click.echo(
        "ERROR: Database path has changed since last install.\n"
        f"  Installed: {installed_cfg.db.url}\n"
        f"  Requested: {new_cfg.db.url}\n"
        "\n"
        "Starting with a different database path would orphan the old\n"
        "database file on disk.  To switch database paths safely:\n"
        "\n"
        "  1. yagents uninstall   (removes old data)\n"
        "  2. yagents install     (with the new config)\n"
        "  3. yagents up\n",
        err=True,
    )
    sys.exit(1)


@cli.command()
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt",
)
@click.pass_context
def uninstall(ctx: click.Context, yes: bool) -> None:
    """Uninstall yagents: stop daemon, remove ALL persistent data.

    \b
    This command will:
      1. Stop the running daemon.
      2. Unregister the systemd user service.
      3. Stop and remove all yagents Docker containers.
      4. Remove the SQLite database file.
      5. Remove the entire ~/.yagents directory (config, skills, dockers, socket).

    After uninstall, the system is as if yagents was never installed.
    """
    # -- Collect what will be removed so we can show the user --
    cfg: Config | None = None
    try:
        cfg = load_config()
    except Exception:
        pass

    items_to_remove: list[str] = []
    db_path: Path | None = None
    if cfg is not None:
        db_path = cfg.sqlite_path
        if db_path is not None and db_path.exists():
            # If the db lives outside YAGENTS_HOME, list it explicitly
            try:
                db_path.relative_to(YAGENTS_HOME)
            except ValueError:
                items_to_remove.append(f"Database:       {db_path}")
    if YAGENTS_HOME.exists():
        items_to_remove.append(f"Data directory: {YAGENTS_HOME}/")

    if not items_to_remove:
        click.echo("Nothing to uninstall (no data found).")
        return

    click.echo("The following will be PERMANENTLY deleted:")
    for item in items_to_remove:
        click.echo(f"  {item}")
    click.echo()

    if not yes:
        if not click.confirm("Are you sure you want to proceed?"):
            click.echo("Aborted.")
            return

    total_steps = 5
    # -- Step 1: stop daemon --
    click.echo()
    click.echo(f"[1/{total_steps}] Stopping daemon ...")
    c = _client(ctx)
    try:
        c.health()
        c.shutdown()
        click.echo("  -> Shutdown requested.")
    except DaemonNotRunningError:
        click.echo("  -> Daemon is not running.")
    except Exception:
        click.echo("  -> Daemon is not running.")
    finally:
        c.close()

    # -- Step 2: unregister systemd service --
    click.echo()
    click.echo(f"[2/{total_steps}] Unregistering systemd user service ...")
    try:
        from yuuagents.cli.service import uninstall as uninstall_service

        uninstall_service()
        click.echo("  -> Service unregistered.")
    except RuntimeError as e:
        click.echo(f"  -> Skipped: {e}", err=True)

    # -- Step 3: stop & remove yagents Docker containers --
    click.echo()
    click.echo(f"[3/{total_steps}] Removing yagents Docker containers ...")
    _remove_yagents_containers()

    # -- Step 4: remove database (if outside YAGENTS_HOME) --
    click.echo()
    click.echo(f"[4/{total_steps}] Removing database ...")
    if db_path is not None and db_path.exists():
        try:
            db_path.relative_to(YAGENTS_HOME)
            click.echo(f"  -> {db_path} (will be removed with data directory)")
        except ValueError:
            db_path.unlink()
            click.echo(f"  -> Removed {db_path}")
    else:
        click.echo("  -> No database file found.")

    # -- Step 5: remove ~/.yagents entirely --
    click.echo()
    click.echo(f"[5/{total_steps}] Removing data directory ...")
    if YAGENTS_HOME.exists():
        shutil.rmtree(YAGENTS_HOME)
        click.echo(f"  -> Removed {YAGENTS_HOME}/")
    else:
        click.echo(f"  -> {YAGENTS_HOME}/ does not exist.")

    click.echo()
    click.echo("Uninstall complete. All yagents data has been removed.")


def _remove_yagents_containers() -> None:
    """Stop and remove all Docker containers whose name starts with ``yagents-``."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=^yagents-",
                "--format",
                "{{.ID}} {{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            click.echo("  -> WARNING: could not list Docker containers.", err=True)
            return
    except FileNotFoundError, subprocess.TimeoutExpired:
        click.echo("  -> Skipped (Docker not available).")
        return

    lines = [
        line.strip() for line in result.stdout.strip().splitlines() if line.strip()
    ]
    if not lines:
        click.echo("  -> No yagents containers found.")
        return

    for line in lines:
        parts = line.split(None, 1)
        cid = parts[0]
        name = parts[1] if len(parts) > 1 else cid[:12]
        subprocess.run(
            ["docker", "rm", "-f", cid],
            capture_output=True,
            timeout=15,
        )
        click.echo(f"  -> Removed container {name} ({cid[:12]})")


def _docker_available() -> bool:
    ok, _detail = _docker_check()
    return ok


def _runtime_image_tag() -> str:
    try:
        v = importlib.metadata.version("yuuagents")
    except Exception:
        v = "latest"
    return f"yuuagents-runtime:{v}"


def _runtime_dockerfile_text() -> str:
    dockerfile_path = (
        Path(__file__).resolve().parents[1] / "daemon" / "runtime.Dockerfile"
    )
    try:
        return dockerfile_path.read_text(encoding="utf-8")
    except Exception:
        return _FALLBACK_RUNTIME_DOCKERFILE


def _build_runtime_image(tag: str) -> bool:
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


_FALLBACK_RUNTIME_DOCKERFILE = """\
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN set -eux; \\
    for i in 1 2 3; do \\
        apt-get update -y \\
        && apt-get install -y --no-install-recommends --fix-missing \\
            bash \\
            ca-certificates \\
            coreutils \\
            diffutils \\
            findutils \\
            gawk \\
            patch \\
            sed \\
            tmux \\
        && break; \\
        sleep 2; \\
    done; \\
    rm -rf /var/lib/apt/lists/*

RUN cat > /usr/local/bin/yagents-apply-patch <<'EOF'
#!/usr/bin/env bash
set +e

target="$1"
if [ -z "$target" ]; then
  echo "missing target path"
  echo "__YAGENTS_EXIT_CODE__=2"
  exit 0
fi
case "$target" in
  /*) ;;
  *)
    echo "target path must be absolute: $target"
    echo "__YAGENTS_EXIT_CODE__=2"
    exit 0
    ;;
esac

before="$(mktemp)"
after="$(mktemp)"
patch_raw="$(mktemp)"
patch_rewritten="$(mktemp)"

cleanup() {
  rm -f "$before" "$after" "$patch_raw" "$patch_rewritten"
}
trap cleanup EXIT

mkdir -p "$(dirname "$target")"

if [ -f "$target" ]; then
  cat "$target" > "$before"
else
  : > "$before"
fi

cat > "$patch_raw"
if [ "$?" -ne 0 ]; then
  echo "failed to read patch"
  echo "__YAGENTS_EXIT_CODE__=3"
  exit 0
fi

awk -v tgt="$target" '
BEGIN { done_old=0; done_new=0 }
{
  if (!done_old && $0 ~ /^--- /) { print "--- " tgt; done_old=1; next }
  if (!done_new && $0 ~ /^\\+\\+\\+ /) { print "+++ " tgt; done_new=1; next }
  print $0
}
' "$patch_raw" > "$patch_rewritten"

patch -p0 -u -N --silent < "$patch_rewritten"
code=$?
if [ "$code" -ne 0 ]; then
  echo "patch failed (exit=$code)"
  echo "__YAGENTS_EXIT_CODE__=$code"
  exit 0
fi

if [ -f "$target" ]; then
  cat "$target" > "$after"
else
  : > "$after"
fi

if cmp -s "$before" "$after"; then
  echo "$target"
  echo "no-op"
  echo "hunks: 0"
  echo "lines: +0 -0"
  echo "__YAGENTS_EXIT_CODE__=0"
  exit 0
fi

read hunks added removed <<EOF2
$(diff -u "$before" "$after" | awk '
BEGIN { h=0; a=0; r=0 }
/^@@ / { h++ }
/^\\+\\+\\+ / { next }
/^--- / { next }
/^\\+/ { a++; next }
/^-/ { r++; next }
END { printf "%d %d %d\\n", h, a, r }
')
EOF2

echo "$target"
echo "hunks: $hunks"
echo "lines: +$added -$removed"
echo "__YAGENTS_EXIT_CODE__=0"
exit 0
EOF

RUN chmod +x /usr/local/bin/yagents-apply-patch
"""


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
    except FileNotFoundError, subprocess.TimeoutExpired:
        return False


def _docker_check() -> tuple[bool, str]:
    timeout_s = _docker_timeout_seconds()
    try:
        version = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        return False, "docker CLI not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "docker --version timed out"

    try:
        server = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        v = (version.stdout or version.stderr or "").strip()
        return (
            False,
            f"{v or 'docker'}; docker daemon check timed out after {timeout_s}s",
        )

    if server.returncode == 0:
        return True, ""

    v = (version.stdout or version.stderr or "docker").strip()
    raw = (server.stderr or server.stdout or "").strip()
    first_line = (
        raw.splitlines()[0].strip()
        if raw
        else f"docker version exited {server.returncode}"
    )
    lower = raw.lower()
    if "permission denied" in lower:
        return False, f"{v}; permission denied connecting to daemon ({first_line})"
    if (
        "cannot connect to the docker daemon" in lower
        or "is the docker daemon running" in lower
    ):
        return False, f"{v}; daemon not reachable ({first_line})"
    if "context" in lower and "not found" in lower:
        return False, f"{v}; docker context error ({first_line})"
    return False, f"{v}; docker daemon check failed ({first_line})"


def _docker_timeout_seconds() -> int:
    raw = os.environ.get("YAGENTS_DOCKER_TIMEOUT", "").strip()
    if raw.isdigit():
        return max(5, min(int(raw), 120))
    return 30


# ── Configuration management ──


@cli.command()
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.yaml (fully replaces current config)",
)
@click.option(
    "--overrides",
    "overrides_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to config.overrides.yaml (deep-merged on top)",
)
@click.pass_context
def config(
    ctx: click.Context,
    config_path: str | None,
    overrides_path: str | None,
) -> None:
    """View or update configuration (hot-reload without daemon restart).

    \b
    Usage:
      yagents config                    # Print current config (from memory)
      yagents config --overrides FILE   # Update config with overrides
      yagents config --config FILE      # Replace config entirely

    \b
    This command updates both disk (~/.yagents/config.yaml) and the running
    daemon's in-memory configuration without requiring a restart.
    """
    # No arguments: print current config from daemon
    if not config_path and not overrides_path:
        try:
            cfg = load_config()
            config_dict = {
                "db": {
                    "url": cfg.db.url,
                },
                "yuutrace": {
                    "db_path": cfg.yuutrace.db_path,
                    "ui_port": cfg.yuutrace.ui_port,
                    "server_port": cfg.yuutrace.server_port,
                },
                "daemon": {
                    "socket": cfg.daemon.socket,
                    "log_level": cfg.daemon.log_level,
                },
                "docker": {
                    "image": cfg.docker.image,
                },
                "skills": {
                    "paths": cfg.skills.paths,
                },
                "tavily": {
                    "api_key_env": cfg.tavily.api_key_env,
                },
                "providers": {
                    name: {
                        "api_type": p.api_type,
                        "api_key_env": p.api_key_env,
                        "default_model": p.default_model,
                        "base_url": p.base_url,
                        "organization": p.organization,
                    }
                    for name, p in cfg.providers.items()
                },
                "agents": {
                    name: {
                        "provider": a.provider,
                        "model": a.model,
                        "persona": a.persona,
                        "tools": a.tools,
                        "skills": a.skills,
                    }
                    for name, a in cfg.agents.items()
                },
            }
            click.echo(
                yaml.dump(config_dict, default_flow_style=False, allow_unicode=True)
            )
        except Exception as e:
            click.echo(f"Error loading config: {e}", err=True)
            sys.exit(1)
        return

    # Update config: similar logic to install command
    if config_path:
        click.echo(f"Config file:   {config_path}")
        click.echo("WARNING: --config fully replaces the current configuration.")
        if not click.confirm("Are you sure you want to proceed?"):
            click.echo("Aborted.")
            sys.exit(0)

        user_config_p = Path(config_path)
        user_data = yaml.safe_load(user_config_p.read_text(encoding="utf-8")) or {}

        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            user_data = _deep_merge(user_data, over_data)
            click.echo(f"Overrides:     {overrides_path}")

        merged_data = user_data
    else:
        # Load current config and apply overrides
        if not DEFAULT_CONFIG_PATH.exists():
            click.echo(
                f"Error: {DEFAULT_CONFIG_PATH} not found. Run `yagents install` first.",
                err=True,
            )
            sys.exit(1)

        base_data = (
            yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
        )
        click.echo(f"Base config:   {DEFAULT_CONFIG_PATH}")

        if overrides_path:
            over_p = Path(overrides_path)
            over_data = yaml.safe_load(over_p.read_text(encoding="utf-8")) or {}
            base_data = _deep_merge(base_data, over_data)
            click.echo(f"Overrides:     {overrides_path}")

        merged_data = base_data

    # Write to disk
    click.echo()
    click.echo("Updating configuration ...")
    DEFAULT_CONFIG_PATH.write_text(
        yaml.dump(merged_data, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    click.echo(f"  -> Written to {DEFAULT_CONFIG_PATH}")

    # Validate
    cfg = load_config()
    errors = cfg.validate()
    if errors:
        click.echo("  WARNING: configuration has validation errors:", err=True)
        for err in errors:
            click.echo(f"    - {err}", err=True)

    # Hot-reload daemon if running
    click.echo()
    click.echo("Reloading daemon configuration ...")
    c = _client(ctx)
    try:
        c.health()
        result = c.reload_config()
        if result.get("ok"):
            click.echo("  -> Daemon configuration reloaded successfully")
        else:
            click.echo(f"  WARNING: {result.get('error', 'Unknown error')}", err=True)
    except DaemonNotRunningError:
        click.echo(
            "  WARNING: Daemon is not running. "
            "The config file has been updated; changes will take effect on next start.",
            err=True,
        )
    except Exception as e:
        click.echo(
            f"  WARNING: Could not reload daemon (is it running?): {e}", err=True
        )
        click.echo(
            "  The config file has been updated, but you may need to restart the daemon.",
            err=True,
        )
    finally:
        c.close()

    click.echo()
    click.echo("Configuration update complete!")


# ── Daemon management ──


@cli.command()
@click.option("--config", "config_path", default=None, help="Config file path")
@click.option("-d", "--daemon", is_flag=True, default=False, help="Run in background")
@click.option(
    "--dot-env",
    "dot_env_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Load environment variables from a .env file",
)
@click.pass_context
def up(
    ctx: click.Context,
    config_path: str | None,
    daemon: bool,
    dot_env_path: str | None,
) -> None:
    """Start the yagents daemon.

    Before starting, the resolved configuration is compared against the
    installed config (``~/.yagents/config.yaml``).  If the database path
    has changed, the command refuses to start and asks the user to run
    ``yagents uninstall`` + ``yagents install`` instead.  This prevents
    orphaned database files from leaking on disk.

    The installed config file is then updated with the latest resolved
    configuration so that subsequent ``up`` invocations pick up non-db
    changes automatically.
    """
    used_dotenv: Path | None = None
    if dot_env_path:
        used_dotenv = Path(dot_env_path).expanduser()
        _load_dotenv_file(used_dotenv)
    else:
        auto = _find_dotenv(Path.cwd())
        if auto is not None:
            used_dotenv = auto
            _load_dotenv_file(auto)

    if daemon:
        if used_dotenv is None:
            try:
                from yuuagents.cli.service import start as start_service

                start_service()
                click.echo("Started via systemd user service.")
                return
            except RuntimeError:
                pass

        yagents_bin = shutil.which("yagents")
        if yagents_bin:
            cmd = [yagents_bin, "up"]
        else:
            cmd = [sys.executable, "-m", "yuuagents.cli.main", "up"]

        if config_path:
            cmd.extend(["--config", config_path])

        subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        click.echo("Daemon started in background.")
        return

    from yuuagents.daemon.server import serve
    import msgspec

    # -- Resolve the "new" config that the user wants to run with --
    if config_path:
        cfg = load_config(config_path)
    else:
        local_cfg = _local_config()
        local_over = _local_overrides()

        if local_cfg is not None:
            click.echo(f"Found local:   {local_cfg}")
            base_data = yaml.safe_load(local_cfg.read_text(encoding="utf-8")) or {}
        elif DEFAULT_CONFIG_PATH.exists():
            base_data = (
                yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}
            )
        else:
            base_data = {}

        if local_over is not None:
            click.echo(f"Found local:   {local_over}")
            over_data = yaml.safe_load(local_over.read_text(encoding="utf-8")) or {}
            base_data = _deep_merge(base_data, over_data)

        cfg = msgspec.convert(base_data, Config)

    # -- Guard: detect database path change --
    _check_db_path_unchanged(cfg)

    # -- Persist the latest config so it becomes the new "installed" baseline --
    if DEFAULT_CONFIG_PATH.exists():
        new_data = json.loads(msgspec.json.encode(cfg))
        DEFAULT_CONFIG_PATH.write_text(
            yaml.dump(new_data, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )

    click.echo(f"Starting daemon on {cfg.socket_path} ...")
    asyncio.run(serve(cfg))


@cli.command()
@click.pass_context
def down(ctx: click.Context) -> None:
    """Stop the daemon."""
    try:
        from yuuagents.cli.service import stop as stop_service

        stop_service(ignore_errors=True)
    except RuntimeError:
        pass

    c = _client(ctx)
    try:
        c.health()
        c.shutdown()
        click.echo("Shutdown requested.")
    except DaemonNotRunningError:
        click.echo("Daemon is not running.")
    except Exception:
        click.echo("Daemon is not running.")
    finally:
        c.close()


@cli.command()
@click.argument("task_id")
@click.pass_context
def stop(ctx: click.Context, task_id: str) -> None:
    """Cancel a running task."""
    c = _client(ctx)
    try:
        c.cancel(task_id)
        click.echo(f"Task {task_id} cancelled.")
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


# ── Agent management ──


@cli.command()
@click.option(
    "--agent",
    "agent_name",
    default="main",
    help="Agent config name (default: main)",
)
@click.option(
    "--persona",
    default="",
    help="Override the system prompt (persona) from agent config",
)
@click.option("--task", required=True, help="Task description")
@click.option("--tools", default=None, help="Comma-separated tool names")
@click.option(
    "--skills", "skill_names", default=None, help="Comma-separated skill names"
)
@click.option("--model", default="", help="LLM model override")
@click.option("--container", default="", help="Existing Docker container ID to use")
@click.option("--image", default="", help="Docker image to create a new container from")
@click.pass_context
def run(
    ctx: click.Context,
    agent_name: str,
    persona: str,
    task: str,
    tools: str | None,
    skill_names: str | None,
    model: str,
    container: str,
    image: str,
) -> None:
    """Submit a new agent task."""
    c = _client(ctx)
    payload: dict[str, object] = {
        "agent": agent_name,
        "persona": persona,
        "task": task,
        "model": model,
        "container": container,
        "image": image,
    }
    if tools:
        payload["tools"] = [t.strip() for t in tools.split(",")]
    if skill_names:
        payload["skills"] = [s.strip() for s in skill_names.split(",")]
    try:
        result = c.submit(payload)
        click.echo(f"Task started: {result['task_id']}  agent={result['agent_id']}")
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command("list")
@click.pass_context
def list_agents(ctx: click.Context) -> None:
    """List all agents."""
    c = _client(ctx)
    try:
        agents = c.list_agents()
        if not agents:
            click.echo("No agents.")
            return
        for a in agents:
            status = a.get("status", "?")
            tid = a.get("task_id", "?")
            aid = a.get("agent_id", "?")
            task = a.get("task", "")[:60]
            steps = a.get("steps", 0)
            cost = a.get("total_cost_usd", 0)
            click.echo(
                f"  [{status:>8}] {tid}  agent={aid}  steps={steps}  ${cost:.4f}  {task}"
            )
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command()
@click.argument("task_id")
@click.pass_context
def status(ctx: click.Context, task_id: str) -> None:
    """Show task status."""
    c = _client(ctx)
    try:
        info = c.status(task_id)
        click.echo(json.dumps(info, indent=2, ensure_ascii=False))
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command()
@click.argument("task_id")
@click.pass_context
def logs(ctx: click.Context, task_id: str) -> None:
    """Show task conversation history."""
    c = _client(ctx)
    try:
        hist = c.history(task_id)
        for msg in hist:
            role: str = "?"
            items: list[object] = []
            if isinstance(msg, dict):
                role = str(msg.get("role", "?"))
                raw_items = msg.get("items", [])
                items = raw_items if isinstance(raw_items, list) else []
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                role = str(msg[0])
                raw_items = msg[1]
                items = raw_items if isinstance(raw_items, list) else []

            click.echo(f"\n--- {role} ---")
            for item in items:
                if isinstance(item, str):
                    click.echo(item)
                else:
                    click.echo(json.dumps(item, indent=2, ensure_ascii=False))
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


@cli.command()
@click.argument("task_id")
@click.argument("message")
@click.pass_context
def input(ctx: click.Context, task_id: str, message: str) -> None:
    """Send input to a task.

    If the agent is blocked on user_input, this replies to that request.
    If the agent is done/cancelled, this appends a new user message and resumes it.
    """
    c = _client(ctx)
    try:
        c.respond(task_id, message)
        click.echo("Input sent.")
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        c.close()


# ── Trace ──


@cli.group()
def trace() -> None:
    """Manage tracing."""


@trace.command("ui")
def trace_ui() -> None:
    """Start yuutrace WebUI."""
    cfg = load_config()
    ytrace_bin = shutil.which("ytrace")
    if ytrace_bin is None:
        click.echo("Error: ytrace not found in PATH", err=True)
        sys.exit(1)
    db_path = str(Path(cfg.yuutrace.db_path).expanduser())
    port = str(cfg.yuutrace.ui_port)
    cmd = [ytrace_bin, "ui", "--db", db_path, "--port", port]
    click.echo(f"Starting trace UI on http://127.0.0.1:{port} ...")
    raise SystemExit(subprocess.call(cmd))


# ── Skills ──


@cli.group()
def skills() -> None:
    """Manage Agent Skills."""


@skills.command("list")
@click.pass_context
def skills_list(ctx: click.Context) -> None:
    """List discovered skills."""
    c = _client(ctx)
    try:
        sk = c.skills()
        if not sk:
            click.echo("No skills found.")
            return
        for s in sk:
            click.echo(f"  {s['name']:20s}  {s.get('description', '')}")
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    finally:
        c.close()


@skills.command("scan")
@click.pass_context
def skills_scan(ctx: click.Context) -> None:
    """Rescan skill directories."""
    c = _client(ctx)
    try:
        sk = c.scan_skills()
        click.echo(f"Found {len(sk)} skill(s).")
        for s in sk:
            click.echo(f"  {s['name']:20s}  {s.get('description', '')}")
    except DaemonNotRunningError as e:
        click.echo(str(e), err=True)
        sys.exit(1)
    finally:
        c.close()
