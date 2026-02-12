"""Tests for yuuagents.tools module."""

from __future__ import annotations

import base64
import difflib
import re

import pytest
import yuutools as yt

from yuuagents.context import AgentContext, DelegateDepthExceededError
from yuuagents.tools import BUILTIN_TOOLS, get
from yuuagents.tools.bash import execute_bash
from yuuagents.tools.delegate import delegate
from yuuagents.tools.file import _write_file_patch, delete_file, read_file, write_file
from yuuagents.tools.skill_cli import execute_skill_cli
from yuuagents.tools.user_input import user_input
from yuuagents.tools.web import web_search


class TestBuiltinToolsRegistry:
    """Tests for BUILTIN_TOOLS dictionary."""

    def test_registry_has_all_tools(self) -> None:
        """Registry should contain all expected tools."""
        expected = {
            "execute_bash",
            "execute_skill_cli",
            "delegate",
            "read_file",
            "write_file",
            "delete_file",
            "user_input",
            "web_search",
        }
        actual = set(BUILTIN_TOOLS.keys())
        assert actual == expected

    def test_registry_values_are_tools(self) -> None:
        """All values should be Tool objects."""
        for name, tool in BUILTIN_TOOLS.items():
            assert isinstance(tool, yt.Tool), f"{name} is not a Tool"

    def test_execute_bash_in_registry(self) -> None:
        """execute_bash should be in registry."""
        assert "execute_bash" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["execute_bash"] is execute_bash

    def test_execute_skill_cli_in_registry(self) -> None:
        """execute_skill_cli should be in registry."""
        assert "execute_skill_cli" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["execute_skill_cli"] is execute_skill_cli

    def test_read_file_in_registry(self) -> None:
        """read_file should be in registry."""
        assert "read_file" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["read_file"] is read_file

    def test_write_file_in_registry(self) -> None:
        """write_file should be in registry."""
        assert "write_file" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["write_file"] is write_file

    def test_delete_file_in_registry(self) -> None:
        """delete_file should be in registry."""
        assert "delete_file" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["delete_file"] is delete_file

    def test_web_search_in_registry(self) -> None:
        """web_search should be in registry."""
        assert "web_search" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["web_search"] is web_search

    def test_user_input_in_registry(self) -> None:
        """user_input should be in registry."""
        assert "user_input" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["user_input"] is user_input

    def test_delegate_in_registry(self) -> None:
        """delegate should be in registry."""
        assert "delegate" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["delegate"] is delegate


class TestGetFunction:
    """Tests for get() function."""

    def test_get_single_tool(self) -> None:
        """Should return single tool."""
        tools = get(["execute_bash"])
        assert len(tools) == 1
        assert tools[0] is execute_bash

    def test_get_multiple_tools(self) -> None:
        """Should return multiple tools in order."""
        tools = get(["read_file", "write_file", "execute_bash"])
        assert len(tools) == 3
        assert tools[0] is read_file
        assert tools[1] is write_file
        assert tools[2] is execute_bash

    def test_get_empty_list(self) -> None:
        """Should return empty list for empty input."""
        tools = get([])
        assert tools == []

    def test_get_unknown_tool_raises_keyerror(self) -> None:
        """Should raise KeyError for unknown tool."""
        with pytest.raises(KeyError) as exc_info:
            get(["unknown_tool"])
        assert "unknown" in str(exc_info.value).lower()

    def test_get_partial_unknown_raises_keyerror(self) -> None:
        """Should raise KeyError if any tool is unknown."""
        with pytest.raises(KeyError):
            get(["execute_bash", "unknown_tool", "read_file"])

    def test_get_error_message_includes_available(self) -> None:
        """Error message should list available tools."""
        with pytest.raises(KeyError) as exc_info:
            get(["nonexistent"])
        error_msg = str(exc_info.value)
        assert "available:" in error_msg
        assert "execute_bash" in error_msg
        assert "read_file" in error_msg

    def test_get_duplicate_names(self) -> None:
        """Should handle duplicate names (returns duplicates)."""
        tools = get(["execute_bash", "execute_bash"])
        assert len(tools) == 2
        assert tools[0] is tools[1]


class TestExecuteBashTool:
    """Tests for execute_bash tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(execute_bash, yt.Tool)


class TestExecuteSkillCliTool:
    """Tests for execute_skill_cli tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(execute_skill_cli, yt.Tool)


@pytest.mark.asyncio
async def test_execute_skill_cli_allows_simple_command(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ctx = AgentContext(
        task_id="t1",
        agent_id="main",
        workdir=str(tmp_path),
        docker_container="c1",
    )
    bound = execute_skill_cli.bind(ctx)
    out = await bound.run(command="/usr/bin/echo hello", timeout=10)
    assert "hello" in out
    assert "__YAGENTS_EXIT_CODE__" not in out


@pytest.mark.asyncio
async def test_execute_skill_cli_inherits_process_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("YAGENTS_SKILL_CLI_ENV_TEST", "hello123")
    ctx = AgentContext(
        task_id="t1",
        agent_id="main",
        workdir=str(tmp_path),
        docker_container="c1",
    )
    bound = execute_skill_cli.bind(ctx)
    out = await bound.run(command="/usr/bin/env", timeout=10)
    assert "YAGENTS_SKILL_CLI_ENV_TEST=hello123" in out
    assert "__YAGENTS_EXIT_CODE__" not in out


@pytest.mark.asyncio
async def test_execute_skill_cli_blocks_rm(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ctx = AgentContext(
        task_id="t1",
        agent_id="main",
        workdir=str(tmp_path),
        docker_container="c1",
    )
    bound = execute_skill_cli.bind(ctx)
    with pytest.raises(ValueError, match="dangerous command"):
        await bound.run(command="rm -rf .", timeout=10)


@pytest.mark.asyncio
async def test_execute_skill_cli_blocks_shell_operators(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    ctx = AgentContext(
        task_id="t1",
        agent_id="main",
        workdir=str(tmp_path),
        docker_container="c1",
    )
    bound = execute_skill_cli.bind(ctx)
    with pytest.raises(ValueError, match="operators"):
        await bound.run(command="git status && echo hi", timeout=10)


class TestReadFileTool:
    """Tests for read_file tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(read_file, yt.Tool)


class TestWriteFileTool:
    """Tests for write_file tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(write_file, yt.Tool)


class TestDeleteFileTool:
    """Tests for delete_file tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(delete_file, yt.Tool)


class TestWebSearchTool:
    """Tests for web_search tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(web_search, yt.Tool)


@pytest.mark.asyncio
async def test_delegate_depth_limit_raises_custom_error() -> None:
    class FakeManager:
        async def delegate(
            self,
            *,
            agent: str,
            first_user_message: str,
            tools: list[str] | None,
            delegate_depth: int,
        ) -> str:
            return "ok"

    ctx = AgentContext(
        task_id="t1",
        agent_id="main",
        workdir="/tmp",
        docker_container="c1",
        delegate_depth=3,
        manager=FakeManager(),
    )
    bound = delegate.bind(ctx)
    with pytest.raises(DelegateDepthExceededError) as exc_info:
        await bound.run(agent="coder", context="x", task="y")
    msg = str(exc_info.value)
    assert "delegate depth limit exceeded" in msg
    assert "max_depth=3" in msg


class _FakeDocker:
    def __init__(self, initial: str) -> None:
        self.content = initial

    async def exec(self, container_id: str, command: str, timeout: int) -> str:
        m = re.search(
            r"^printf %s (?P<b64>'[^']*'|[A-Za-z0-9+/=]+) \| base64 -d \| yagents-apply-patch (?P<path>.+)$",
            command,
        )
        if m is not None:
            b64 = m.group("b64").strip("'")
            patch_text = base64.b64decode(b64.encode("ascii")).decode("utf-8")
            before = self.content
            after = _apply_unified_diff(before, patch_text)
            self.content = after
            summary = _diff_summary(
                path=m.group("path").strip(),
                before=before,
                after=after,
            )
            return f"{summary}\n__YAGENTS_EXIT_CODE__=0"

        raise AssertionError(f"unexpected command: {command!r}")


def _apply_unified_diff(before: str, patch_text: str) -> str:
    before_lines = before.splitlines(keepends=True)
    i = 0
    out: list[str] = []

    lines = patch_text.splitlines(keepends=False)
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("@@ "):
            m = re.match(
                r"^@@ -(?P<a>\d+)(?:,(?P<b>\d+))? \+(?P<c>\d+)(?:,(?P<d>\d+))? @@",
                line,
            )
            assert m is not None
            old_start = int(m.group("a"))
            out.extend(before_lines[i : old_start - 1])
            i = old_start - 1
            idx += 1
            while idx < len(lines):
                h = lines[idx]
                if h.startswith("@@ "):
                    break
                if h.startswith("--- ") or h.startswith("+++ "):
                    idx += 1
                    continue
                if not h:
                    idx += 1
                    continue
                tag = h[0]
                text = h[1:] + "\n"
                if tag == " ":
                    assert i < len(before_lines)
                    assert before_lines[i] == text
                    out.append(text)
                    i += 1
                elif tag == "-":
                    assert i < len(before_lines)
                    assert before_lines[i] == text
                    i += 1
                elif tag == "+":
                    out.append(text)
                else:
                    raise AssertionError(f"unexpected hunk line: {h!r}")
                idx += 1
            continue
        idx += 1
    out.extend(before_lines[i:])
    return "".join(out)


def _diff_summary(*, path: str, before: str, after: str) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a{path}",
            tofile=f"b{path}",
            lineterm="",
        )
    )
    hunks = sum(1 for line in diff if line.startswith("@@ "))
    added = sum(
        1 for line in diff if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in diff if line.startswith("-") and not line.startswith("---")
    )
    if before == after:
        return f"{path}\nno-op\nhunks: 0\nlines: +0 -0"
    return f"{path}\nhunks: {hunks}\nlines: +{added} -{removed}"


@pytest.mark.asyncio
async def test_write_file_applies_patch_and_returns_diff_summary() -> None:
    docker = _FakeDocker("a\nb\n")
    patch = "\n".join(
        [
            "--- a/x.txt",
            "+++ b/x.txt",
            "@@ -1,2 +1,3 @@",
            " a",
            "+b2",
            " b",
            "",
        ]
    )
    summary = await _write_file_patch(
        path="/x.txt",
        patch=patch,
        container="c1",
        docker=docker,
    )
    assert docker.content == "a\nb2\nb\n"
    assert "/x.txt" in summary
    assert "hunks: 1" in summary
    assert "lines: +1 -0" in summary


@pytest.mark.asyncio
async def test_write_file_noop_patch_returns_noop_summary() -> None:
    docker = _FakeDocker("a\nb\n")
    patch = "\n".join(
        [
            "--- a/x.txt",
            "+++ b/x.txt",
            "@@ -1,2 +1,2 @@",
            " a",
            " b",
            "",
        ]
    )
    summary = await _write_file_patch(
        path="/x.txt",
        patch=patch,
        container="c1",
        docker=docker,
    )
    assert docker.content == "a\nb\n"
    assert "no-op" in summary


# ── cli_guard integration tests ──────────────────────────────────────────────


@pytest.fixture()
def recording_guard():
    """Guard that records calls and allows everything."""
    calls: list[list[str]] = []

    def _guard(argv: list[str]) -> None:
        calls.append(argv)

    return _guard, calls


@pytest.fixture()
def blocking_guard():
    """Guard that always rejects."""

    def _guard(argv: list[str]) -> None:
        raise ValueError("blocked by guard")

    return _guard


class TestCliGuard:
    @pytest.mark.asyncio
    async def test_guard_receives_correct_argv(
        self, tmp_path, monkeypatch, recording_guard
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        guard, calls = recording_guard
        ctx = AgentContext(
            task_id="t1",
            agent_id="main",
            workdir=str(tmp_path),
            docker_container="c1",
            cli_guard=guard,
        )
        bound = execute_skill_cli.bind(ctx)
        await bound.run(command="cat SKILL.md", timeout=5)
        assert len(calls) == 1
        assert calls[0] == ["cat", "SKILL.md"]

    @pytest.mark.asyncio
    async def test_guard_blocks_execution(
        self, tmp_path, monkeypatch, blocking_guard
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        ctx = AgentContext(
            task_id="t1",
            agent_id="main",
            workdir=str(tmp_path),
            docker_container="c1",
            cli_guard=blocking_guard,
        )
        bound = execute_skill_cli.bind(ctx)
        with pytest.raises(ValueError, match="blocked by guard"):
            await bound.run(command="cat SKILL.md", timeout=5)

    @pytest.mark.asyncio
    async def test_no_guard_does_not_block(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        ctx = AgentContext(
            task_id="t1",
            agent_id="main",
            workdir=str(tmp_path),
            docker_container="c1",
            cli_guard=None,
        )
        bound = execute_skill_cli.bind(ctx)
        result = await bound.run(command="/usr/bin/echo hello", timeout=5)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_blacklist_before_guard(
        self, tmp_path, monkeypatch, recording_guard
    ):
        monkeypatch.setenv("HOME", str(tmp_path))
        guard, calls = recording_guard
        ctx = AgentContext(
            task_id="t1",
            agent_id="main",
            workdir=str(tmp_path),
            docker_container="c1",
            cli_guard=guard,
        )
        bound = execute_skill_cli.bind(ctx)
        with pytest.raises(ValueError, match="dangerous command"):
            await bound.run(command="rm -rf /tmp/foo", timeout=5)
        assert len(calls) == 0
