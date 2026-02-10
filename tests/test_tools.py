"""Tests for yuuagents.tools module."""

from __future__ import annotations

import pytest
import yuutools as yt

from yuuagents.tools import BUILTIN_TOOLS, get
from yuuagents.tools.bash import execute_bash
from yuuagents.tools.file import delete_file, read_file, write_file
from yuuagents.tools.user_input import user_input
from yuuagents.tools.web import web_search


class TestBuiltinToolsRegistry:
    """Tests for BUILTIN_TOOLS dictionary."""

    def test_registry_has_all_tools(self) -> None:
        """Registry should contain all expected tools."""
        expected = {
            "execute_bash",
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
