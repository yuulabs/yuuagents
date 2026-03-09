"""Tests for yuuagents.tools module."""

from __future__ import annotations

import pytest
import yuutools as yt

from yuuagents.context import AgentContext, DelegateDepthExceededError
from yuuagents.tools import BUILTIN_TOOLS, get
from yuuagents.tools.bash import execute_bash
from yuuagents.tools.delegate import delegate
from yuuagents.tools.file import delete_file, edit_file, read_file, write_file
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
            "edit_file",
            "delete_file",
            "read_skill",
            "user_input",
            "web_search",
            "launch_agent",
            "session_poll",
            "session_interrupt",
            "session_result",
            "sleep",
            "view_image",
            "check_running_tool",
            "cancel_running_tool",
            "update_todo",
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
async def test_execute_skill_cli_chain_operators(tmp_path, monkeypatch) -> None:
    """Chain operators (&&, ||, |) are allowed and executed via shell."""
    monkeypatch.setenv("HOME", str(tmp_path))
    ctx = AgentContext(
        task_id="t1",
        agent_id="main",
        workdir=str(tmp_path),
        docker_container="c1",
    )
    bound = execute_skill_cli.bind(ctx)
    out = await bound.run(command="/usr/bin/echo a && /usr/bin/echo b", timeout=10)
    assert "a" in out and "b" in out


class TestHeredocValidation:
    """Heredoc support in _validate_cli_command."""

    def test_heredoc_basic(self):
        """Simple heredoc should pass validation."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        cmd = "ybot im send --ctx 3 << 'EOF'\n[{\"type\":\"text\",\"text\":\"hello\"}]\nEOF"
        result = _validate_cli_command(cmd)
        assert result is not None

    def test_heredoc_with_inner_quotes(self):
        """Heredoc body with quotes should not be rejected."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        cmd = 'ybot im send --ctx 3 << \'EOF\'\n[{"type":"text","text":"他说\\"你好\\""}]\nEOF'
        result = _validate_cli_command(cmd)
        assert result is not None

    def test_heredoc_body_not_validated_as_command(self):
        """Heredoc body can contain forbidden programs/tokens — it's data."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        # Body contains "rm", ";", "sudo" — all forbidden as commands but fine as data
        cmd = "ybot im send --ctx 3 << 'EOF'\nrm -rf /; sudo reboot\nEOF"
        result = _validate_cli_command(cmd)
        assert result is not None

    def test_heredoc_command_after_delimiter_rejected(self):
        """No commands allowed after heredoc closing delimiter."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        cmd = "ybot im send --ctx 3 << 'EOF'\nhello\nEOF\nrm -rf /"
        with pytest.raises(ValueError, match="after.*heredoc"):
            _validate_cli_command(cmd)

    def test_heredoc_forbidden_program_in_command_rejected(self):
        """The command part (before <<) is still validated."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        cmd = "rm -rf / << 'EOF'\ndata\nEOF"
        with pytest.raises(ValueError, match="dangerous command"):
            _validate_cli_command(cmd)

    def test_heredoc_unclosed_rejected(self):
        """Heredoc without closing delimiter should be rejected."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        cmd = "ybot im send --ctx 3 << 'EOF'\nhello world"
        with pytest.raises(ValueError, match="unclosed.*heredoc"):
            _validate_cli_command(cmd)

    def test_heredoc_empty_delimiter_rejected(self):
        """Heredoc with empty delimiter should be rejected."""
        from yuuagents.tools.skill_cli import _validate_cli_command

        cmd = "ybot im send --ctx 3 <<\nhello\n"
        with pytest.raises(ValueError):
            _validate_cli_command(cmd)

    @pytest.mark.asyncio
    async def test_heredoc_e2e_cat(self, tmp_path, monkeypatch):
        """Heredoc actually pipes data to stdin of the command."""
        monkeypatch.setenv("HOME", str(tmp_path))
        out = await execute_skill_cli.fn(
            command="cat << 'EOF'\nhello heredoc\nEOF",
            timeout=5,
            cli_guard=None,
        )
        assert "hello heredoc" in out

    @pytest.mark.asyncio
    async def test_heredoc_preserves_quotes(self, tmp_path, monkeypatch):
        """Heredoc body preserves quotes intact through shell execution."""
        monkeypatch.setenv("HOME", str(tmp_path))
        out = await execute_skill_cli.fn(
            command='cat << \'EOF\'\n{"key": "value with \\"quotes\\""}\nEOF',
            timeout=5,
            cli_guard=None,
        )
        assert '"key"' in out
        assert "value with" in out


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


class TestEditFileTool:
    """Tests for edit_file tool."""

    def test_is_tool_instance(self) -> None:
        """Should be a Tool instance."""
        assert isinstance(edit_file, yt.Tool)

    def test_edit_file_in_registry(self) -> None:
        """edit_file should be in registry."""
        assert "edit_file" in BUILTIN_TOOLS
        assert BUILTIN_TOOLS["edit_file"] is edit_file


class TestEditFileE2E:
    """End-to-end tests for edit_file — runs the generated Python script locally."""

    @staticmethod
    def _make_ctx(tmp_path):
        import asyncio
        import subprocess

        class LocalExecutor:
            async def exec(self, container_id: str, command: str, timeout: int) -> str:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(stderr.decode())
                return stdout.decode()

        return AgentContext(
            task_id="t1",
            agent_id="main",
            workdir=str(tmp_path),
            docker_container="ignored",
            docker=LocalExecutor(),
        )

    @pytest.mark.asyncio
    async def test_basic_replace(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("hello world")
        ctx = self._make_ctx(tmp_path)
        bound = edit_file.bind(ctx)
        result = await bound.run(
            path=str(target), old_string="hello", new_string="goodbye"
        )
        assert "Edited" in result
        assert target.read_text() == "goodbye world"

    @pytest.mark.asyncio
    async def test_path_with_spaces(self, tmp_path):
        target = tmp_path / "my file.txt"
        target.write_text("aaa bbb ccc")
        ctx = self._make_ctx(tmp_path)
        bound = edit_file.bind(ctx)
        await bound.run(path=str(target), old_string="bbb", new_string="BBB")
        assert target.read_text() == "aaa BBB ccc"

    @pytest.mark.asyncio
    async def test_multiline_strings(self, tmp_path):
        target = tmp_path / "multi.txt"
        target.write_text("line1\nline2\nline3\n")
        ctx = self._make_ctx(tmp_path)
        bound = edit_file.bind(ctx)
        await bound.run(
            path=str(target), old_string="line2\nline3", new_string="replaced"
        )
        assert target.read_text() == "line1\nreplaced\n"

    @pytest.mark.asyncio
    async def test_rejects_zero_occurrences(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("hello world")
        ctx = self._make_ctx(tmp_path)
        bound = edit_file.bind(ctx)
        with pytest.raises(RuntimeError, match="Expected 1 occurrence, found 0"):
            await bound.run(
                path=str(target), old_string="missing", new_string="x"
            )

    @pytest.mark.asyncio
    async def test_rejects_multiple_occurrences(self, tmp_path):
        target = tmp_path / "test.txt"
        target.write_text("aaa bbb aaa")
        ctx = self._make_ctx(tmp_path)
        bound = edit_file.bind(ctx)
        with pytest.raises(RuntimeError, match="Expected 1 occurrence, found 2"):
            await bound.run(
                path=str(target), old_string="aaa", new_string="x"
            )

    @pytest.mark.asyncio
    async def test_special_chars(self, tmp_path):
        target = tmp_path / "special.txt"
        target.write_text('say "hello" & <world>')
        ctx = self._make_ctx(tmp_path)
        bound = edit_file.bind(ctx)
        await bound.run(
            path=str(target),
            old_string='"hello" & <world>',
            new_string='"bye" | {earth}',
        )
        assert target.read_text() == 'say "bye" | {earth}'


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


class TestReadFileE2E:
    """End-to-end tests for read_file — runs via LocalExecutor."""

    @staticmethod
    def _make_ctx(tmp_path):
        import asyncio
        import subprocess

        class LocalExecutor:
            async def exec(self, container_id: str, command: str, timeout: int) -> str:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                stdout, stderr = await proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(stderr.decode())
                return stdout.decode()

        return AgentContext(
            task_id="t1",
            agent_id="main",
            workdir=str(tmp_path),
            docker_container="ignored",
            docker=LocalExecutor(),
        )

    @pytest.mark.asyncio
    async def test_read_text_file(self, tmp_path):
        target = tmp_path / "hello.txt"
        target.write_text("hello world\n")
        ctx = self._make_ctx(tmp_path)
        bound = read_file.bind(ctx)
        result = await bound.run(path=str(target))
        assert result == "hello world\n"

    @pytest.mark.asyncio
    async def test_read_large_text_file_rejected(self, tmp_path):
        target = tmp_path / "big.txt"
        target.write_text("line\n" * 700)
        ctx = self._make_ctx(tmp_path)
        bound = read_file.bind(ctx)
        with pytest.raises(RuntimeError, match="too large"):
            await bound.run(path=str(target))

    @pytest.mark.asyncio
    async def test_read_binary_file_rejected(self, tmp_path):
        target = tmp_path / "blob.bin"
        target.write_bytes(b"\x00\x01\x02\x03" * 100)
        ctx = self._make_ctx(tmp_path)
        bound = read_file.bind(ctx)
        with pytest.raises(RuntimeError, match="Binary file"):
            await bound.run(path=str(target))

    @pytest.mark.asyncio
    async def test_read_image_returns_multimodal(self, tmp_path):
        # Create a minimal valid PNG (1x1 pixel)
        import base64

        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        target = tmp_path / "pixel.png"
        target.write_bytes(base64.b64decode(png_b64))
        ctx = self._make_ctx(tmp_path)
        bound = read_file.bind(ctx)
        result = await bound.run(path=str(target))
        assert isinstance(result, list)
        assert result[0]["type"] == "image_url"
        assert result[0]["image_url"]["url"].startswith("data:image/")
