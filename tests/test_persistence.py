from __future__ import annotations

from datetime import datetime, timezone

import pytest
import yuullm

from yuuagents.persistence import TaskPersistence, TaskRecorder, TaskWriter, ToolCallDTO
from yuuagents.types import AgentStatus


@pytest.mark.asyncio
async def test_recover_pending_tools_writes_interrupted_once(tmp_path) -> None:
    db_url = f"sqlite+aiosqlite:///{tmp_path / 'tasks.sqlite3'}"

    p = TaskPersistence(db_url=db_url)
    await p.start()
    writer = TaskWriter(persistence=p)
    await writer.start()

    created_at = datetime.now(timezone.utc)
    await p.create_task(
        task_id="t1",
        agent_id="main",
        persona="persona",
        task="do something",
        system_prompt="system",
        model="test-model",
        tools=["execute_bash"],
        docker_container="dummy",
        created_at=created_at,
    )

    recorder = TaskRecorder(task_id="t1", writer=writer)
    assistant_msg = yuullm.assistant(
        "hi",
        {
            "type": "tool_call",
            "id": "c1",
            "name": "execute_bash",
            "arguments": "{}",
        },
    )
    await recorder.record_llm(
        turn=1,
        history_append=assistant_msg,
        tool_calls=[ToolCallDTO(call_id="c1", name="execute_bash", args_json="{}")],
    )

    await writer.flush()
    await writer.stop()
    await p.stop()

    p2 = TaskPersistence(db_url=db_url)
    await p2.start()
    try:
        assert await p2.recover_pending_tools("t1") is True
        assert await p2.recover_pending_tools("t1") is False

        row = await p2.get_task_row("t1")
        assert row is not None
        assert row.status == AgentStatus.RUNNING.value
        assert row.head_turn == 1

        history = await p2.load_history("t1")
        assert len(history) >= 3
        last = history[-1]
        assert isinstance(last, tuple)
        assert last[0] == "tool"
        assert "interrupted" in str(last)
    finally:
        await p2.stop()
