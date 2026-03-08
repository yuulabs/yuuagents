"""Todo tool — LLM self-tracks task progress for user reporting."""

from __future__ import annotations

import json

import yuutools as yt


@yt.tool(
    params={
        "items": (
            'JSON array of task items, e.g. '
            '[{"task": "搜索资料", "status": "done"}, {"task": "写代码", "status": "in_progress"}]. '
            'Valid statuses: pending, in_progress, done, skipped'
        ),
    },
    description=(
        "更新你的任务清单。用它追踪自己的工作进度，方便给用户汇报。"
        "每次调用会替换整个清单。"
    ),
)
async def update_todo(items: str) -> str:
    parsed = json.loads(items) if isinstance(items, str) else items
    if not isinstance(parsed, list):
        return "[ERROR] items must be a JSON array"
    lines = []
    for item in parsed:
        task = item.get("task", "?")
        status = item.get("status", "pending")
        icon = {"done": "✓", "in_progress": "▶", "skipped": "—"}.get(status, "○")
        lines.append(f"  {icon} {task}")
    return "Todo updated:\n" + "\n".join(lines)
