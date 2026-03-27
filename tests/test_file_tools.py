from __future__ import annotations

import json

import pytest

from yuuagents.context import AgentContext, DockerExecutor
from yuuagents.tools.file import edit_file, read_file


class StubDocker:
    def __init__(self, output: str) -> None:
        self.output = output
        self.calls: list[tuple[str, str, int]] = []

    async def exec(self, container_id: str, command: str, timeout: int) -> str:
        self.calls.append((container_id, command, timeout))
        return self.output

    async def exec_terminal(
        self,
        container_id: str,
        session_id: str,
        command: str,
        timeout: int,
        *,
        soft_timeout: int | None = None,
    ) -> str:
        del container_id, session_id, command, timeout, soft_timeout
        raise NotImplementedError

    def get_pending(self, container_id: str, session_id: str) -> None:
        del container_id, session_id
        return None

    async def resume_pending(
        self,
        container_id: str,
        session_id: str,
        timeout: int,
    ) -> str:
        del container_id, session_id, timeout
        raise NotImplementedError

    async def write_terminal(
        self,
        container_id: str,
        session_id: str,
        data: str,
        *,
        append_newline: bool = True,
    ) -> str:
        del container_id, session_id, data, append_newline
        raise NotImplementedError

    async def capture_terminal(self, container_id: str, session_id: str) -> str:
        del container_id, session_id
        raise NotImplementedError


def make_ctx(docker: DockerExecutor) -> AgentContext:
    return AgentContext(
        task_id="task-1",
        agent_id="agent-1",
        workdir="/tmp",
        docker_container="cid",
        docker=docker,
    )


@pytest.mark.asyncio
async def test_read_file_returns_text_payload() -> None:
    docker = StubDocker(
        json.dumps(
            {
                "kind": "text",
                "payload": "hello\nworld\n",
                "total_lines": 2,
                "start_line": 1,
                "end_line": 2,
                "returned_lines": 2,
            }
        )
    )
    ctx = make_ctx(docker)

    result = await read_file.bind(ctx).run(path="/tmp/demo.txt")

    assert result == (
        "[read_file] path=/tmp/demo.txt lines=1-2 returned_lines=2 total_lines=2\n"
        "hello\nworld\n"
    )
    assert docker.calls
    container_id, command, timeout = docker.calls[0]
    assert container_id == "cid"
    assert timeout == 30
    assert "start_line = 1" in command
    assert "max_lines = 200" in command


@pytest.mark.asyncio
async def test_read_file_supports_line_ranges() -> None:
    docker = StubDocker(
        json.dumps(
            {
                "kind": "text",
                "payload": "b\nc\n",
                "total_lines": 4,
                "start_line": 2,
                "end_line": 3,
                "returned_lines": 2,
            }
        )
    )
    ctx = make_ctx(docker)

    result = await read_file.bind(ctx).run(path="/tmp/demo.txt", start_line=2, end_line=3)

    assert result == (
        "[read_file] path=/tmp/demo.txt lines=2-3 returned_lines=2 total_lines=4\n"
        "b\nc\n"
    )
    _, command, _ = docker.calls[0]
    assert "start_line = 2" in command
    assert "end_line = 3" in command


@pytest.mark.asyncio
async def test_read_file_surfaces_container_error_output() -> None:
    docker = StubDocker(
        "Traceback (most recent call last):\n"
        '  File "<stdin>", line 19, in <module>\n'
        "RuntimeError: something went wrong\n"
    )
    ctx = make_ctx(docker)

    result = await read_file.bind(ctx).run(path="/tmp/large.txt")

    assert result.startswith("[read_file error]")
    assert "RuntimeError: something went wrong" in result


@pytest.mark.asyncio
async def test_edit_file_supports_exact_string_replacement() -> None:
    docker = StubDocker("Edited /tmp/demo.txt\n")

    result = await edit_file.bind(make_ctx(docker)).run(
        path="/tmp/demo.txt",
        old_string="before",
        new_string="after",
    )

    assert result == "Edited /tmp/demo.txt\n"
    _, command, timeout = docker.calls[0]
    assert timeout == 30
    assert "count = text.count(old)" in command
    assert "start_line = None" in command


@pytest.mark.asyncio
async def test_edit_file_supports_line_range_replacement() -> None:
    docker = StubDocker("Edited /tmp/demo.txt\n")

    result = await edit_file.bind(make_ctx(docker)).run(
        path="/tmp/demo.txt",
        start_line=2,
        end_line=3,
        new_string="replacement\n",
    )

    assert result == "Edited /tmp/demo.txt\n"
    _, command, _ = docker.calls[0]
    assert "lines = text.splitlines(keepends=True)" in command
    assert "start_line = 2" in command
    assert "end_line = 3" in command


@pytest.mark.asyncio
async def test_edit_file_rejects_mixed_selectors() -> None:
    docker = StubDocker("unused")

    with pytest.raises(
        ValueError,
        match="Provide exactly one edit selector: either old_string or start_line/end_line",
    ):
        await edit_file.bind(make_ctx(docker)).run(
            path="/tmp/demo.txt",
            old_string="before",
            start_line=1,
            end_line=1,
            new_string="after",
        )


@pytest.mark.asyncio
async def test_edit_file_surfaces_line_range_errors() -> None:
    docker = StubDocker(
        "Traceback (most recent call last):\n"
        '  File "<stdin>", line 21, in <module>\n'
        "RuntimeError: Line range out of bounds: lines=10-12, total_lines=8\n"
    )

    result = None
    with pytest.raises(RuntimeError, match=r"Line range out of bounds: lines=10-12, total_lines=8"):
        result = await edit_file.bind(make_ctx(docker)).run(
            path="/tmp/demo.txt",
            start_line=10,
            end_line=12,
            new_string="after\n",
        )
    assert result is None
