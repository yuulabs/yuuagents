"""DockerManager — async Docker container lifecycle."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import aiodocker
from aiodocker.exceptions import DockerError
from attrs import define, field
from loguru import logger

_DOCKERS_ROOT = Path("~/.yagents/dockers").expanduser()

# Injected into every agent's system prompt so it knows the mount layout.
DOCKER_SYSTEM_PROMPT = """\
<docker_environment>
You are running commands inside a Docker container.
- The host filesystem root is mounted read-only at /mnt/host
- Your home directory /home/yuu is a persistent read-write workspace
  (backed by a host directory specific to this container)
- You can read any host file via /mnt/host/... but cannot modify the host directly.
</docker_environment>"""


@define
class DockerManager:
    """Manage Docker containers for agent bash/file tools."""

    image: str = "ubuntu:24.04"
    container_home: str = "/home/yuu"
    uid: int | None = None
    gid: int | None = None
    default_container: str = ""
    _containers: dict[str, str] = field(factory=dict)
    _client: aiodocker.Docker | None = field(default=None, repr=False)
    _start_lock: asyncio.Lock = field(factory=asyncio.Lock, repr=False)

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
                return info["Id"]
            except Exception as exc:
                raise ValueError(f"container not found: {container}") from exc

        if image:
            cid = await self._create(image=image, task_id=task_id)
            if task_id:
                self._containers[task_id] = cid
            return cid

        assert self.default_container
        return self.default_container

    async def exec(self, container_id: str, command: str, timeout: int) -> str:
        """Run *command* in *container_id*, return stdout+stderr."""

        await self._ensure_started()

        async def _exec_with_shell(shell: str) -> str:
            assert self._client is not None
            container = self._client.containers.container(container_id)
            exe = await container.exec(
                cmd=[shell, "-c", command],
                stdout=True,
                stderr=True,
                workdir=self.workdir,
                environment={"HOME": self.container_home},
            )

            # aiodocker 0.25.0: Exec.start() is synchronous and returns a Stream.
            # Older versions returned an awaitable; support both to keep behavior stable.
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

            async def _read_all() -> bytes:
                chunks: list[bytes] = []
                try:
                    while True:
                        msg = await stream.read_out()  # type:ignore
                        if msg is None:
                            break
                        data = getattr(msg, "data", b"")
                        if data:
                            chunks.append(data)
                finally:
                    await _close_stream()
                return b"".join(chunks)

            try:
                output = await asyncio.wait_for(_read_all(), timeout=timeout)
            except asyncio.TimeoutError:
                await _close_stream()
                return f"[ERROR] Command timed out after {timeout}s"

            return output.decode("utf-8", errors="replace")

        # Prefer bash for the default Ubuntu container, but fall back to sh for
        # minimal images (e.g. alpine) that don't ship bash.
        result = await _exec_with_shell("bash")
        if "executable file not found" in result and '"bash"' in result:
            return await _exec_with_shell("sh")
        return result

    async def cleanup(self, task_id: str) -> None:
        """Remove a per-task container (if we created one)."""
        cid = self._containers.pop(task_id, None)
        if cid and cid != self.default_container:
            await self._remove(cid)

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
                return info["Id"]
            await container.start()
            info = await container.show()
            return info["Id"]
        except Exception:
            pass

        return await self._create(image=self.image, name=name)

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

        config: dict = {
            "Image": image,
            "Cmd": ["sleep", "infinity"],
            "Tty": False,
            "Env": [f"HOME={container_home}"],
            "WorkingDir": container_home,
            "User": self._user_spec(),
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
