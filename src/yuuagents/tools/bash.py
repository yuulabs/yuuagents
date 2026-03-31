"""execute_bash — run commands in a Docker container."""

from __future__ import annotations

import asyncio
from typing import Any

import yuutools as yt
from loguru import logger

from yuuagents.capabilities import DockerCapability, require_docker
from yuuagents.core.flow import Flow

_SOFT_TIMEOUT_PREFIX = "[SOFT_TIMEOUT] "


@yt.tool(
    params={
        "command": "The bash command to execute",
        "timeout": "Timeout in seconds (default 120, max 600)",
        "soft_timeout": (
            "Optional soft timeout in seconds. If the command has not "
            "finished after this many seconds, the tool is moved to the "
            "background and partial output is returned together with a "
            "run id. The command keeps running — use input_background to "
            "send stdin, and wait_background / inspect_background to "
            "collect the final result."
        ),
    },
    description=(
        "Execute a bash command in the Docker container(work dir is `/home/yuu`). Commands run in a persistent "
        "terminal-like session: working directory and environment variables persist "
        "across calls."
    ),
)
async def execute_bash(
    command: str,
    timeout: int = 120,
    soft_timeout: int | None = None,
    docker: DockerCapability = yt.depends(require_docker),
    flow: Flow[Any, str] | None = yt.depends(lambda ctx: ctx.current_flow),
) -> str:
    if flow is None:
        raise RuntimeError("execute_bash requires an active flow")
    timeout = max(1, min(timeout, 600))
    container = docker.container_id
    executor = docker.executor
    session_id = flow.id

    if soft_timeout is None:
        return await executor.exec_terminal(
            container, session_id, command, timeout,
        )

    soft_timeout = max(1, min(soft_timeout, timeout))
    result = await executor.exec_terminal(
        container, session_id, command, timeout, soft_timeout=soft_timeout,
    )

    if not result.startswith(_SOFT_TIMEOUT_PREFIX):
        # Completed within soft_timeout — return normally.
        return result

    # Command is still running.  Ask the agent to defer us to background
    # and continue waiting for the final result in the background task.
    partial = result.removeprefix(_SOFT_TIMEOUT_PREFIX).removeprefix(
        "Command is still running.\n"
    )
    flow.request_defer(partial)

    # Phase 2: background — poll PendingCommand for partial output, then await.
    pending = executor.get_pending(container, session_id)
    assert pending is not None

    while not pending.done:
        while not flow.mailbox.empty():
            try:
                data = flow.mailbox.get_nowait()
            except asyncio.QueueEmpty:
                break
            try:
                await executor.write_terminal(
                    container,
                    session_id,
                    data,
                    append_newline=False,
                )
            except Exception:
                logger.debug("mailbox input failed during bg wait")
        await asyncio.sleep(3)
        try:
            cap = await executor.capture_terminal(container, session_id)
            part = pending.partial(cap)
            if part:
                flow.emit(part)
        except Exception:
            logger.debug("periodic capture failed during bg wait")

    return await pending.wait()
