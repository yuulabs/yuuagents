"""Structured runtime inputs for yuuagents."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, cast

from attrs import define, field
import yuullm
from yuullm.types import (
    is_audio_item,
    is_file_item,
    is_image_item,
    is_text_item,
    is_tool_call_item,
    is_tool_result_item,
)


@define(frozen=True, kw_only=True)
class ConversationInput:
    kind: Literal["conversation"] = "conversation"
    messages: list[yuullm.Message] = field(factory=list)


@define(frozen=True, kw_only=True)
class HandoffInput:
    kind: Literal["handoff"] = "handoff"
    context: list[yuullm.Message] = field(factory=list)
    task: list[yuullm.Message] = field(factory=list)


@define(frozen=True, kw_only=True)
class RolloverInput:
    kind: Literal["rollover"] = "rollover"
    context: list[yuullm.Message] = field(factory=list)
    summary: list[yuullm.Message] = field(factory=list)
    task: list[yuullm.Message] = field(factory=list)


@define(frozen=True, kw_only=True)
class ScheduledInput:
    kind: Literal["scheduled"] = "scheduled"
    context: list[yuullm.Message] = field(factory=list)
    task: list[yuullm.Message] = field(factory=list)
    trigger: list[yuullm.Message] = field(factory=list)


AgentInput: TypeAlias = (
    ConversationInput
    | HandoffInput
    | RolloverInput
    | ScheduledInput
)


def conversation_input_from_text(text: str) -> ConversationInput:
    return ConversationInput(messages=[yuullm.user(text)])


def iter_input_fields(agent_input: AgentInput) -> list[tuple[str, list[yuullm.Message]]]:
    match agent_input:
        case ConversationInput(messages=messages):
            return [("messages", messages)]
        case HandoffInput(context=context, task=task):
            return [("context", context), ("task", task)]
        case RolloverInput(context=context, summary=summary, task=task):
            return [("context", context), ("summary", summary), ("task", task)]
        case ScheduledInput(context=context, task=task, trigger=trigger):
            return [("context", context), ("task", task), ("trigger", trigger)]


def flatten_input_messages(agent_input: AgentInput) -> list[yuullm.Message]:
    messages: list[yuullm.Message] = []
    for _, field_messages in iter_input_fields(agent_input):
        messages.extend(field_messages)
    return messages


def agent_input_preview(agent_input: AgentInput, *, max_chars: int = 160) -> str:
    parts: list[str] = []
    for name, messages in iter_input_fields(agent_input):
        text = " ".join(render_message_text(msg) for msg in messages).strip()
        if not text:
            continue
        parts.append(f"{name}={text}")
    preview = " | ".join(parts) if parts else agent_input.kind
    return _truncate_text(preview, max_chars)


def agent_input_field_previews(
    agent_input: AgentInput,
    *,
    max_chars: int = 160,
) -> dict[str, str]:
    previews: dict[str, str] = {}
    for name, messages in iter_input_fields(agent_input):
        text = " ".join(render_message_text(msg) for msg in messages).strip()
        previews[name] = _truncate_text(text, max_chars) if text else ""
    return previews


def agent_input_to_jsonable(agent_input: AgentInput) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": agent_input.kind}
    for name, messages in iter_input_fields(agent_input):
        payload[name] = [message_to_jsonable(message) for message in messages]
    return payload


def agent_input_from_jsonable(value: Any) -> AgentInput:
    if not isinstance(value, dict):
        raise TypeError("input must be an object")

    kind = value.get("kind")
    if not isinstance(kind, str):
        raise TypeError("input.kind must be a string")

    match kind:
        case "conversation":
            return ConversationInput(
                messages=_message_list_from_jsonable(value.get("messages", []), "messages"),
            )
        case "handoff":
            return HandoffInput(
                context=_message_list_from_jsonable(value.get("context", []), "context"),
                task=_message_list_from_jsonable(value.get("task", []), "task"),
            )
        case "rollover":
            return RolloverInput(
                context=_message_list_from_jsonable(value.get("context", []), "context"),
                summary=_message_list_from_jsonable(value.get("summary", []), "summary"),
                task=_message_list_from_jsonable(value.get("task", []), "task"),
            )
        case "scheduled":
            return ScheduledInput(
                context=_message_list_from_jsonable(value.get("context", []), "context"),
                task=_message_list_from_jsonable(value.get("task", []), "task"),
                trigger=_message_list_from_jsonable(value.get("trigger", []), "trigger"),
            )
        case _:
            raise ValueError(f"unsupported input.kind {kind!r}")


def message_to_jsonable(message: yuullm.Message) -> list[Any]:
    role, items = message
    return [role, [dict(item) for item in items]]


def message_from_jsonable(value: Any) -> yuullm.Message:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError("message must be a [role, items] pair")

    role = value[0]
    if role not in {"system", "user", "assistant", "tool"}:
        raise TypeError(f"unsupported message role {role!r}")

    raw_items = value[1]
    if not isinstance(raw_items, list):
        raise TypeError("message items must be a list")

    items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise TypeError("message items must be objects")
        item = dict(raw_item)
        item_type = item.get("type")
        if not isinstance(item_type, str):
            raise TypeError("message item.type must be a string")
        items.append(item)

    _validate_message_items(cast(str, role), items)
    return cast(yuullm.Message, (role, items))


def render_message_text(message: yuullm.Message) -> str:
    role, items = message
    parts: list[str] = []
    for item in items:
        if is_text_item(item):
            parts.append(item["text"])
        elif is_image_item(item):
            url = item["image_url"]["url"]
            parts.append("<base64 image>" if url.startswith("data:") else f"<image {url}>")
        elif is_audio_item(item):
            parts.append("<audio>")
        elif is_file_item(item):
            parts.append("<file>")
        elif is_tool_call_item(item):
            parts.append(f"{item['name']}({item['arguments']})")
        elif is_tool_result_item(item):
            parts.append(f"[tool_result {item['tool_call_id']}]")
        else:
            parts.append(f"<{role}>")
    return "".join(parts)


def _message_list_from_jsonable(value: Any, field_name: str) -> list[yuullm.Message]:
    if not isinstance(value, list):
        raise TypeError(f"input.{field_name} must be a list")
    return [message_from_jsonable(message) for message in value]


def _validate_message_items(role: str, items: list[dict[str, Any]]) -> None:
    allowed_by_role = {
        "system": {"text"},
        "user": {"text", "image_url", "input_audio", "file"},
        "assistant": {"text", "tool_call"},
        "tool": {"tool_result"},
    }
    allowed = allowed_by_role[role]
    for item in items:
        item_type = item["type"]
        if item_type not in allowed:
            raise TypeError(
                f"message role {role!r} does not accept item type {item_type!r}"
            )


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
