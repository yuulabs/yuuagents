import pytest

import yuullm

from yuuagents.loop import _trace_llm_gen_items


class TestTraceLlmGenItems:
    def test_joins_text_fragments(self):
        items = [
            yuullm.Response(item="hel"),
            yuullm.Response(item="lo"),
        ]
        assert _trace_llm_gen_items(items) == [{"type": "text", "text": "hello"}]

    def test_groups_tool_calls_and_parses_arguments(self):
        items = [
            yuullm.ToolCall(id="tc_1", name="search", arguments='{"q":"x"}'),
            yuullm.ToolCall(id="tc_2", name="calc", arguments='{"x":1}'),
        ]
        assert _trace_llm_gen_items(items) == [
            {
                "type": "tool_calls",
                "tool_calls": [
                    {"id": "tc_1", "function": "search", "arguments": {"q": "x"}},
                    {"id": "tc_2", "function": "calc", "arguments": {"x": 1}},
                ],
            }
        ]

    def test_keeps_invalid_tool_call_arguments_as_string(self):
        items = [yuullm.ToolCall(id="tc_1", name="search", arguments="{")]
        assert _trace_llm_gen_items(items) == [
            {
                "type": "tool_calls",
                "tool_calls": [{"id": "tc_1", "function": "search", "arguments": "{"}],
            }
        ]

    def test_flushes_text_around_tool_calls(self):
        items = [
            yuullm.Response(item="a"),
            yuullm.Response(item="b"),
            yuullm.ToolCall(id="tc_1", name="t", arguments="{}"),
            yuullm.Response(item="c"),
        ]
        assert _trace_llm_gen_items(items) == [
            {"type": "text", "text": "ab"},
            {
                "type": "tool_calls",
                "tool_calls": [{"id": "tc_1", "function": "t", "arguments": {}}],
            },
            {"type": "text", "text": "c"},
        ]

    def test_passes_through_typed_dict_items(self):
        items = [yuullm.Response(item={"type": "text", "text": "hi"})]
        assert _trace_llm_gen_items(items) == [{"type": "text", "text": "hi"}]


@pytest.mark.parametrize(
    "stream_item",
    [
        yuullm.Reasoning(item="think"),
        object(),
    ],
)
def test_does_not_crash_on_non_standard_items(stream_item):
    assert isinstance(_trace_llm_gen_items([stream_item]), list)
