"""DockerManager — async Docker container lifecycle."""

from __future__ import annotations

import asyncio
import base64
import importlib.metadata
import os
import re
import shlex
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

import aiodocker
from aiodocker.exceptions import DockerError
from attrs import define, field
from loguru import logger

if TYPE_CHECKING:
    from yuuagents.flow import OutputBuffer

_DOCKERS_ROOT = Path("~/.yagents/dockers").expanduser()

_PROXY_KEYS = ("http_proxy", "https_proxy", "no_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY")


def _proxy_env() -> dict[str, str]:
    """Collect proxy env vars from the host process."""
    return {k: v for k in _PROXY_KEYS if (v := os.environ.get(k))}


# Injected into every agent's system prompt so it knows the mount layout.
DOCKER_SYSTEM_PROMPT = """\
<docker_environment>
You are running commands inside a Docker container as root.
- The host filesystem root is mounted read-only at /mnt/host
- Your home directory /home/yuu is a persistent read-write workspace
  (backed by a host directory specific to this container)
- You can read any host file via /mnt/host/... but cannot modify the host directly.
- You have full root access: install packages with apt, npm, etc. directly.
</docker_environment>"""


@define
class DockerManager:
    """Manage Docker containers for agent bash/file tools."""

    image: str = "yuuagents-runtime:latest"
    container_home: str = "/home/yuu"
    uid: int | None = None
    gid: int | None = None
    default_container: str = ""
    _containers: dict[str, str] = field(factory=dict)
    _client: aiodocker.Docker | None = field(default=None, repr=False)
    _start_lock: asyncio.Lock = field(factory=asyncio.Lock, repr=False)
    _terminal_locks: dict[tuple[str, str], asyncio.Lock] = field(
        factory=dict, repr=False
    )
    _tooling_ready: set[str] = field(factory=set, repr=False)
    _tooling_locks: dict[str, asyncio.Lock] = field(factory=dict, repr=False)

    @property
    def workdir(self) -> str:
        return self.container_home

    def _user_spec(self) -> str:
        uid = self.uid if self.uid is not None else os.getuid()
        gid = self.gid if self.gid is not None else os.getgid()
        return f"{uid}:{gid}"

    async def _ensure_started(self) -> None:
        if self._client is not None and self.default_container:
            return

        async with self._start_lock:
            if self._client is not None and self.default_container:
                return
            self._client = aiodocker.Docker()
            if not self.default_container:
                self.default_container = await self._ensure_default()
            logger.info("Docker ready, default container: {}", self.default_container)

    async def start(self) -> None:
        await self._ensure_started()

    async def stop(self) -> None:
        # Clean up non-default containers we created
        for task_id, cid in list(self._containers.items()):
            if cid != self.default_container:
                await self._remove(cid)
        self._containers.clear()
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def default_container_id(self) -> str:
        return self.default_container

    async def host_home_dir(self, container_id: str) -> str:
        """Return the host directory bound to ``container_home``."""
        await self._ensure_started()
        assert self._client is not None

        container = self._client.containers.container(container_id)
        info = await container.show()
        for mount in info.get("Mounts", []):
            if mount.get("Destination") == self.container_home:
                source = mount.get("Source", "")
                if source:
                    return source
                break
        raise ValueError(
            f"container {container_id!r} has no mount for {self.container_home!r}"
        )

    async def resolve(
        self,
        *,
        task_id: str = "",
        container: str = "",
        image: str = "",
    ) -> str:
        """Resolve a container for a task.

        - ``container`` given → verify it exists and is running, then use it.
        - ``image`` given (without container) → create a new container from that image.
        - No args → use the shared default container.
        """
        await self._ensure_started()

        if container and image:
            raise ValueError("Provide either container or image, not both")

        if container:
            assert self._client is not None
            try:
                c = self._client.containers.container(container)
                info = await c.show()
                if not info["State"]["Running"]:
                    await c.start()
                    info = await c.show()
                cid = info["Id"]
                await self._ensure_required_tooling(cid)
                return cid
            except Exception as exc:
                raise ValueError(f"container not found: {container}") from exc

        if image:
            cid = await self._create(image=image, task_id=task_id)
            if task_id:
                self._containers[task_id] = cid
            await self._ensure_required_tooling(cid)
            return cid

        assert self.default_container
        return self.default_container

    async def exec(self, container_id: str, command: str, timeout: int) -> str:
        """Run *command* in *container_id*, return stdout+stderr."""

        await self._ensure_started()

        # Prefer bash for the default Ubuntu container, but fall back to sh for
        # minimal images (e.g. alpine) that don't ship bash.
        result = await self._exec_with_shell(container_id, "bash", command, timeout)
        if "executable file not found" in result and '"bash"' in result:
            return await self._exec_with_shell(container_id, "sh", command, timeout)
        return result

    async def exec_terminal(
        self,
        container_id: str,
        session_id: str,
        command: str,
        timeout: int,
        output_buffer: OutputBuffer | None = None,
    ) -> str:
        await self._ensure_started()
        assert isinstance(session_id, str) and session_id

        session_name = self._terminal_session_name(session_id)
        key = (container_id, session_name)
        lock = self._terminal_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._terminal_locks[key] = lock

        async with lock:
            await self._ensure_required_tooling(container_id)
            await self._ensure_tmux_session(container_id, session_name)
            return await self._exec_tmux_command(
                container_id=container_id,
                session_name=session_name,
                command=command,
                timeout=timeout,
                output_buffer=output_buffer,
            )

    async def _exec_with_shell(
        self,
        container_id: str,
        shell: str,
        command: str,
        timeout: int,
        *,
        user: str | None = None,
        workdir: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> str:
        assert self._client is not None
        container = self._client.containers.container(container_id)
        if environment is None:
            environment = {"HOME": self.container_home, **_proxy_env()}
        kwargs: dict = dict(
            cmd=[shell, "-c", command],
            stdout=True,
            stderr=True,
            workdir=workdir or self.workdir,
            environment=environment,
            user=user if user is not None else "root",
        )
        exe = await container.exec(**kwargs)

        started = exe.start()
        if asyncio.iscoroutine(started):  # pragma: no cover
            stream = await started
        else:
            stream = started

        async def _close_stream() -> None:
            try:
                closed = stream.close()  # type: ignore
                if asyncio.iscoroutine(closed):
                    await closed
            except Exception:
                pass

        async def _read_all(
            output_buffer: OutputBuffer | None = None,
        ) -> bytes:
            chunks: list[bytes] = []
            try:
                while True:
                    msg = await stream.read_out()  # type:ignore
                    if msg is None:
                        break
                    data = getattr(msg, "data", b"")
                    if data:
                        chunks.append(data)
                        if output_buffer is not None:
                            output_buffer.write(data)
            finally:
                await _close_stream()
            return b"".join(chunks)

        try:
            output = await asyncio.wait_for(_read_all(), timeout=timeout)
        except asyncio.TimeoutError:
            await _close_stream()
            return f"[ERROR] Command timed out after {timeout}s"

        return output.decode("utf-8", errors="replace")

    @staticmethod
    def _terminal_session_name(session_id: str) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_-]+", "_", session_id).strip("_")
        safe = safe[:48] if len(safe) > 48 else safe
        assert safe
        return f"yag_{safe}"

    @staticmethod
    def _parse_exit_code_marker(output: str) -> int:
        lines = output.splitlines()
        assert lines and lines[-1].startswith("__YAGENTS_EXIT_CODE__=")
        code_raw = lines[-1].removeprefix("__YAGENTS_EXIT_CODE__=").strip()
        assert code_raw.isdigit()
        return int(code_raw)

    async def _ensure_required_tooling(self, container_id: str) -> None:
        if container_id in self._tooling_ready:
            return

        lock = self._tooling_locks.get(container_id)
        if lock is None:
            lock = asyncio.Lock()
            self._tooling_locks[container_id] = lock

        async with lock:
            if container_id in self._tooling_ready:
                return

            missing = await self._missing_required_tooling(container_id)
            if missing:
                raise ValueError(self._missing_tooling_error(container_id, missing))
            self._tooling_ready.add(container_id)

    async def _missing_required_tooling(self, container_id: str) -> list[str]:
        cmd = r"""\
missing=""

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || missing="$missing $1"
}

need_cmd bash
need_cmd tmux
need_cmd patch
need_cmd diff
need_cmd base64
need_cmd awk
need_cmd mktemp
need_cmd cmp
need_cmd cat
need_cmd rm
need_cmd mkdir
need_cmd dirname

test -x /usr/local/bin/yagents-apply-patch || missing="$missing yagents-apply-patch"

echo "$missing"
"""
        try:
            out = await self._exec_with_shell(container_id, "sh", cmd, 30)
        except Exception as exc:
            raise ValueError(
                f"container tooling check failed (container={container_id[:12]}): {exc}"
            ) from exc

        raw = out.strip()
        if not raw:
            return []
        return [p for p in raw.split() if p]

    @staticmethod
    def _missing_tooling_error(container_id: str, missing: list[str]) -> str:
        try:
            pkg_version = importlib.metadata.version("yuuagents")
        except Exception:
            pkg_version = ""

        runtime_hint = (
            f"yuuagents-runtime:{pkg_version}"
            if pkg_version
            else "yuuagents-runtime:<version>"
        )
        tools = ", ".join(missing)
        return (
            "container image does not satisfy yagents runtime requirements\n"
            f"- container: {container_id}\n"
            f"- missing: {tools}\n"
            "\n"
            "If you use a custom image/container, you must preinstall all required tools.\n"
            f"If you use the default install flow, build the runtime image (tag: {runtime_hint})."
        )

    async def _ensure_tmux_session(self, container_id: str, session_name: str) -> None:
        q = shlex.quote(session_name)
        probe = await self._exec_with_shell(
            container_id,
            "bash",
            f"tmux has-session -t {q} >/dev/null 2>&1; echo __YAGENTS_EXIT_CODE__=$?",
            30,
        )
        if self._parse_exit_code_marker(probe) == 0:
            return

        create = (
            f"tmux new-session -d -s {q} -c {shlex.quote(self.workdir)} bash"
            f" >/dev/null 2>&1; echo __YAGENTS_EXIT_CODE__=$?"
        )
        out = await self._exec_with_shell(container_id, "bash", create, 30)
        if self._parse_exit_code_marker(out) == 0:
            return

        create = (
            f"tmux new-session -d -s {q} -c {shlex.quote(self.workdir)} sh"
            f" >/dev/null 2>&1; echo __YAGENTS_EXIT_CODE__=$?"
        )
        out = await self._exec_with_shell(container_id, "bash", create, 30)
        assert self._parse_exit_code_marker(out) == 0, out

    async def _exec_tmux_command(
        self,
        *,
        container_id: str,
        session_name: str,
        command: str,
        timeout: int,
        output_buffer: OutputBuffer | None = None,
    ) -> str:
        token = uuid.uuid4().hex
        cmd_b64 = base64.b64encode(command.encode("utf-8")).decode("ascii")
        begin = f"__YAGENTS_BEGIN__={token}"
        end_prefix = f"__YAGENTS_END__={token}"

        payload = (
            f"printf '\\n{begin}\\n'; "
            f"__y_cmd=$(printf %s {cmd_b64} | base64 -d); "
            f'eval "$__y_cmd"; __y_code=$?; '
            f"printf '\\n{end_prefix} __YAGENTS_EXIT_CODE__=%s\\n' \"$__y_code\""
        )

        q_sess = shlex.quote(session_name)
        send = (
            f"tmux send-keys -t {q_sess} -l {shlex.quote(payload)};"
            f" tmux send-keys -t {q_sess} C-m"
        )
        await self._exec_with_shell(container_id, "bash", send, 30)

        deadline = asyncio.get_running_loop().time() + max(1, timeout)
        last_capture = ""
        streamed_body = ""
        try:
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    await self._interrupt_tmux_command(container_id, session_name)
                    return f"[ERROR] Command timed out after {timeout}s"

                cap = await self._exec_with_shell(
                    container_id,
                    "bash",
                    f"tmux capture-pane -p -t {q_sess} -S -2000",
                    int(min(10, max(1, remaining))),
                )
                last_capture = cap
                body = self._extract_tmux_body(
                    capture=cap,
                    begin=begin,
                    end_prefix=end_prefix,
                )
                if body is not None and output_buffer is not None:
                    delta = body[len(streamed_body):]
                    if delta:
                        output_buffer.write(delta.encode("utf-8", errors="replace"))
                    streamed_body = body
                if end_prefix in cap:
                    break
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            await self._interrupt_tmux_command(container_id, session_name)
            raise

        lines = last_capture.splitlines()
        begin_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == begin:
                begin_idx = i
            if line.strip().startswith(end_prefix):
                end_idx = i

        assert begin_idx != -1 and end_idx != -1 and end_idx >= begin_idx, last_capture

        end_line = lines[end_idx].strip()
        code = 0
        if "__YAGENTS_EXIT_CODE__=" in end_line:
            code_raw = end_line.split("__YAGENTS_EXIT_CODE__=", 1)[1].strip()
            if code_raw.isdigit():
                code = int(code_raw)

        body = "\n".join(lines[begin_idx + 1 : end_idx]).strip()
        if code != 0:
            return f"{body}\n[exit code: {code}]".strip()
        return body

    @staticmethod
    def _extract_tmux_body(
        *,
        capture: str,
        begin: str,
        end_prefix: str,
    ) -> str | None:
        lines = capture.splitlines()
        begin_idx = -1
        end_idx = -1
        for i, line in enumerate(lines):
            if line.strip() == begin:
                begin_idx = i
            if line.strip().startswith(end_prefix):
                end_idx = i
                break
        if begin_idx == -1:
            return None
        if end_idx == -1:
            return "\n".join(lines[begin_idx + 1 :]).strip()
        return "\n".join(lines[begin_idx + 1 : end_idx]).strip()

    async def _interrupt_tmux_command(self, container_id: str, session_name: str) -> None:
        q_sess = shlex.quote(session_name)
        try:
            await self._exec_with_shell(
                container_id,
                "bash",
                f"tmux send-keys -t {q_sess} C-c",
                5,
            )
        except Exception:
            logger.debug("Failed to interrupt tmux session {}", session_name)

    async def cleanup(self, task_id: str) -> None:
        """Remove a per-task container (if we created one)."""
        cid = self._containers.pop(task_id, None)
        if cid and cid != self.default_container:
            await self._remove(cid)
        if cid:
            self._tooling_ready.discard(cid)
            self._tooling_locks.pop(cid, None)
            for key in list(self._terminal_locks):
                if key[0] == cid:
                    self._terminal_locks.pop(key, None)

    # -- private --

    async def _ensure_default(self) -> str:
        """Create or reuse the shared default container."""
        name = "yagents-default"
        assert self._client is not None

        # Try to reuse existing
        try:
            container = self._client.containers.container(name)
            info = await container.show()
            if info["State"]["Running"]:
                cid = info["Id"]
                await self._ensure_required_tooling(cid)
                return cid
            await container.start()
            info = await container.show()
            cid = info["Id"]
            await self._ensure_required_tooling(cid)
            return cid
        except Exception:
            pass

        cid = await self._create(image=self.image, name=name)
        await self._ensure_required_tooling(cid)
        return cid

    async def _create(
        self,
        *,
        image: str = "",
        name: str = "",
        task_id: str = "",
    ) -> str:
        assert self._client is not None
        image = image or self.image

        await self._ensure_image(image)
        if not name:
            import uuid

            name = f"yagents-{uuid.uuid4().hex[:12]}"

        # Prepare host home directory for this container
        home_dir = _DOCKERS_ROOT / name
        home_dir.mkdir(parents=True, exist_ok=True)

        container_home = self.container_home

        env_list = [f"HOME={container_home}"]
        env_list.extend(f"{k}={v}" for k, v in _proxy_env().items())

        config: dict = {
            "Image": image,
            "Cmd": ["sleep", "infinity"],
            "Tty": False,
            "Env": env_list,
            "WorkingDir": container_home,
            "User": "0:0",
            "HostConfig": {
                "Binds": [
                    "/:/mnt/host:ro",
                    f"{home_dir}:{container_home}:rw",
                ],
            },
        }
        if task_id:
            config["Labels"] = {"yagents.task_id": task_id}

        container = await self._client.containers.create_or_replace(
            name=name,
            config=config,
        )
        await container.start()
        info = await container.show()
        cid = info["Id"]
        logger.info("Created container {} ({})", name, cid[:12])
        return cid

    async def _ensure_image(self, image: str) -> None:
        """Ensure *image* exists locally (pull if missing)."""
        assert self._client is not None

        try:
            await self._client.images.inspect(image)
            return
        except DockerError as exc:
            if getattr(exc, "status", None) != 404:
                raise

        if image.startswith("yuuagents-runtime:") or image == "yuuagents-runtime":
            raise RuntimeError(
                f"missing runtime image {image!r}; run `yagents install` to build it"
            )

        # Pull missing image. Support "repo:tag" while keeping registry ports intact.
        from_image = image
        tag: str | None = None
        if "@" not in image and ":" in image:
            from_image, tag = image.rsplit(":", 1)

        logger.info("Pulling Docker image {}", image)
        await self._client.images.pull(from_image, tag=tag)

    async def _remove(self, container_id: str) -> None:
        assert self._client is not None
        try:
            container = self._client.containers.container(container_id)
            await container.kill()
        except Exception:
            pass
        try:
            container = self._client.containers.container(container_id)
            await container.delete(force=True)
        except Exception:
            pass
