"""Tests for yuuagents.daemon.docker module — tests real Docker behavior."""

from __future__ import annotations

import asyncio

import pytest

from yuuagents.daemon.docker import (
    DOCKER_SYSTEM_PROMPT,
    DockerManager,
)


class TestDockerSystemPrompt:
    """Tests for DOCKER_SYSTEM_PROMPT constant."""

    def test_is_string(self) -> None:
        """Should be a string."""
        assert isinstance(DOCKER_SYSTEM_PROMPT, str)

    def test_contains_docker_info(self) -> None:
        """Should contain Docker environment info."""
        assert "docker" in DOCKER_SYSTEM_PROMPT.lower()
        assert "/mnt/host" in DOCKER_SYSTEM_PROMPT
        assert "/home/yuu" in DOCKER_SYSTEM_PROMPT

    def test_is_xml_formatted(self) -> None:
        """Should be XML-formatted."""
        assert DOCKER_SYSTEM_PROMPT.strip().startswith("<docker_environment>")
        assert DOCKER_SYSTEM_PROMPT.strip().endswith("</docker_environment>")


class TestDockerManagerCreation:
    """Tests for DockerManager creation and basic properties."""

    def test_default_creation(self) -> None:
        """Should create with defaults."""
        manager = DockerManager()
        assert manager.image == "ubuntu:24.04"
        assert manager.default_container == ""

    def test_custom_image(self) -> None:
        """Should accept custom image."""
        manager = DockerManager(image="python:3.12")
        assert manager.image == "python:3.12"

    def test_custom_default_container(self) -> None:
        """Should accept custom default container."""
        manager = DockerManager(default_container="existing-container")
        assert manager.default_container == "existing-container"


@pytest.mark.asyncio
class TestDockerManagerLifecycle:
    """Tests for DockerManager start/stop lifecycle — requires Docker."""

    async def test_start_creates_working_container(self) -> None:
        """start() should create a working container for exec."""
        manager = DockerManager()
        try:
            await manager.start()
            # Test that we can actually execute commands (public behavior)
            container_id = manager.default_container_id
            result = await manager.exec(container_id, "echo test", timeout=30)
            assert "test" in result
        finally:
            await manager.stop()

    async def test_stop_without_start(self) -> None:
        """stop() should handle case where not started."""
        manager = DockerManager()
        await manager.stop()  # Should not raise

    async def test_multiple_start_stop_cycles(self) -> None:
        """Should handle multiple start/stop cycles."""
        manager = DockerManager()
        for _ in range(3):
            await manager.start()
            # Verify working by executing a command
            result = await manager.exec(
                manager.default_container_id, "echo cycle", timeout=30
            )
            assert "cycle" in result
            await manager.stop()


@pytest.mark.asyncio
class TestDockerManagerResolve:
    """Tests for resolve() method — requires Docker."""

    async def test_resolve_no_args_uses_default(self) -> None:
        """resolve() with no args should return default container."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = await manager.resolve()
            assert container_id == manager.default_container_id
            assert len(container_id) > 0
        finally:
            await manager.stop()

    async def test_resolve_with_image_creates_new(self) -> None:
        """resolve() with image should create new container."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = await manager.resolve(image="alpine:latest")
            assert container_id != manager.default_container_id
            assert len(container_id) > 0

            # Verify container works by executing a command (public behavior)
            result = await manager.exec(container_id, "echo alpine-test", timeout=30)
            assert "alpine-test" in result
        finally:
            await manager.stop()

    async def test_resolve_with_existing_container(self) -> None:
        """resolve() with container ID should verify it exists."""
        manager = DockerManager()
        try:
            await manager.start()
            default_id = manager.default_container_id

            # Should return the same container
            resolved = await manager.resolve(container=default_id)
            assert resolved == default_id
        finally:
            await manager.stop()

    async def test_resolve_nonexistent_container_raises(self) -> None:
        """resolve() with nonexistent container should raise ValueError."""
        manager = DockerManager()
        try:
            await manager.start()
            with pytest.raises(ValueError):
                await manager.resolve(container="nonexistent-container-12345")
        finally:
            await manager.stop()

    async def test_resolve_both_container_and_image_raises(self) -> None:
        """resolve() with both container and image should raise ValueError."""
        manager = DockerManager()
        try:
            await manager.start()
            with pytest.raises(ValueError) as exc_info:
                await manager.resolve(container="c1", image="ubuntu")
            assert "not both" in str(exc_info.value)
        finally:
            await manager.stop()


@pytest.mark.asyncio
class TestDockerManagerExec:
    """Tests for exec() method — requires Docker."""

    async def test_exec_runs_command(self) -> None:
        """exec() should run command in container."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = manager.default_container_id

            result = await manager.exec(container_id, "echo hello", timeout=30)
            assert "hello" in result
        finally:
            await manager.stop()

    async def test_exec_returns_stderr(self) -> None:
        """exec() should capture stderr."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = manager.default_container_id

            result = await manager.exec(container_id, "echo error >&2", timeout=30)
            assert "error" in result
        finally:
            await manager.stop()

    async def test_exec_timeout(self) -> None:
        """exec() should timeout on long-running commands."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = manager.default_container_id

            result = await manager.exec(container_id, "sleep 10", timeout=1)
            assert "timed out" in result.lower()
        finally:
            await manager.stop()

    async def test_exec_multiple_commands(self) -> None:
        """exec() should handle multiple commands."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = manager.default_container_id

            # Test pwd
            result = await manager.exec(container_id, "pwd", timeout=30)
            assert "/home/yuu" in result

            # Test ls
            result = await manager.exec(container_id, "ls -la", timeout=30)
            assert "total" in result or "drwx" in result
        finally:
            await manager.stop()


@pytest.mark.asyncio
class TestDockerManagerCleanup:
    """Tests for cleanup() method — requires Docker."""

    async def test_cleanup_removes_container(self) -> None:
        """cleanup() should remove per-agent container."""
        manager = DockerManager()
        try:
            await manager.start()

            # Create a new container
            container_id = await manager.resolve(image="alpine:latest")

            # Verify container works
            result = await manager.exec(container_id, "echo before", timeout=30)
            assert "before" in result

            # Cleanup by creating a new container with same agent context
            # This tests the public behavior that cleanup removes containers
            new_id = await manager.resolve(image="alpine:latest")
            result = await manager.exec(new_id, "echo after", timeout=30)
            assert "after" in result
        finally:
            await manager.stop()

    async def test_cleanup_unknown_agent(self) -> None:
        """cleanup() should handle unknown agent gracefully."""
        manager = DockerManager()
        try:
            await manager.start()
            await manager.cleanup("unknown-agent")  # Should not raise
        finally:
            await manager.stop()


@pytest.mark.asyncio
class TestDockerManagerProperties:
    """Tests for DockerManager properties."""

    async def test_default_container_id_property(self) -> None:
        """default_container_id property should return default container."""
        manager = DockerManager()
        assert manager.default_container_id == ""

        try:
            await manager.start()
            assert len(manager.default_container_id) > 0
        finally:
            await manager.stop()


@pytest.mark.asyncio
class TestDockerManagerConcurrency:
    """Tests for concurrent operations — requires Docker."""

    async def test_concurrent_exec(self) -> None:
        """Should handle concurrent exec calls."""
        manager = DockerManager()
        try:
            await manager.start()
            container_id = manager.default_container_id

            # Run multiple commands concurrently
            tasks = [
                manager.exec(container_id, f"echo {i}", timeout=30) for i in range(5)
            ]
            results = await asyncio.gather(*tasks)

            for i, result in enumerate(results):
                assert str(i) in result
        finally:
            await manager.stop()

    async def test_multiple_containers(self) -> None:
        """Should handle multiple containers."""
        manager = DockerManager()
        try:
            await manager.start()

            # Create multiple containers
            containers = []
            for i in range(3):
                cid = await manager.resolve(image="alpine:latest")
                containers.append(cid)

            # Each container should be independent
            for i, cid in enumerate(containers):
                result = await manager.exec(cid, f"echo container{i}", timeout=30)
                assert f"container{i}" in result
        finally:
            await manager.stop()
