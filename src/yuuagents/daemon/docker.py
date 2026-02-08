"""DockerManager — async Docker container lifecycle."""

from __future__ import annotations

import asyncio
from pathlib import Path

import aiodocker
from attrs import define, field
from loguru import logger

_DOCKERS_ROOT = Path("~/.yagents/dockers").expanduser()

# Injected into every agent's system prompt so it knows the mount layout.
DOCKER_SYSTEM_PROMPT = """\
<docker_environment>
You are running commands inside a Docker container.
- The host filesystem root is mounted read-only at /mnt/host
- Your home directory /root is a persistent read-write workspace
  (backed by a host directory specific to this container)
- You can read any host file via /mnt/host/... but cannot modify the host directly.
</docker_environment>"""


@define
class DockerManager:
    """Manage Docker containers for agent bash/file tools."""

    image: str = "ubuntu:24.04"
    default_container: str = ""
    _containers: dict[str, str] = field(factory=dict)
    _client: aiodocker.Docker | None = field(default=None, repr=False)

    async def start(self) -> None:
        self._client = aiodocker.Docker()
        if not self.default_container:
            self.default_container = await self._ensure_default()
        logger.info("Docker ready, default container: {}", self.default_container)

    async def stop(self) -> None:
        # Clean up non-default containers we created
        for agent_id, cid in list(self._containers.items()):
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
        container: str = "",
        image: str = "",
    ) -> str:
        """Resolve a container for an agent.

        - No args → use the shared default container.
        - ``container`` given → use that existing container (fail if not found).
        - ``image`` given → create a new container from that image.
        - Both given → error.
        """
        if container and image:
            raise ValueError("Specify either --container or --image, not both")

        if container:
            # Fail-fast: verify it exists
            assert self._client is not None
            try:
                c = self._client.containers.container(container)
                await c.show()
            except Exception as exc:
                raise ValueError(
                    f"Container {container!r} not found or not accessible"
                ) from exc
            return container

        if image:
            # Create a new container specifically for this agent
            agent_id = ""
            return await self._create(image=image, name="", agent_id=agent_id)

        # No args → use default container
        assert self.default_container
        return self.default_container

    async def exec(self, container_id: str, command: str, timeout: int) -> str:
        """Run *command* in *container_id*, return stdout+stderr."""
        assert self._client is not None
        container = self._client.containers.container(container_id)
        exe = await container.exec(
            cmd=["bash", "-c", command],
            stdout=True,
            stderr=True,
        )
        try:
            output = await asyncio.wait_for(exe.start(), timeout=timeout)  # type: ignore
        except asyncio.TimeoutError:
            return f"[ERROR] Command timed out after {timeout}s"
        if isinstance(output, bytes):
            return output.decode("utf-8", errors="replace")
        return str(output) if output else ""

    async def cleanup(self, agent_id: str) -> None:
        """Remove a per-agent container (if we created one)."""
        cid = self._containers.pop(agent_id, None)
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
        agent_id: str = "",
    ) -> str:
        assert self._client is not None
        image = image or self.image
        if not name:
            import uuid

            name = f"yagents-{uuid.uuid4().hex[:12]}"

        # Prepare host home directory for this container
        home_dir = _DOCKERS_ROOT / name
        home_dir.mkdir(parents=True, exist_ok=True)

        config: dict = {
            "Image": image,
            "Cmd": ["sleep", "infinity"],
            "Tty": False,
            "HostConfig": {
                "Binds": [
                    "/:/mnt/host:ro",
                    f"{home_dir}:/root:rw",
                ],
            },
        }
        if agent_id:
            config["Labels"] = {"yagents.agent_id": agent_id}

        container = await self._client.containers.create_or_replace(
            name=name,
            config=config,
        )
        await container.start()
        info = await container.show()
        cid = info["Id"]
        logger.info("Created container {} ({})", name, cid[:12])
        return cid

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
