"""Control tools for waiting on and interacting with deferred runs."""

from __future__ import annotations

import asyncio

import yuutools as yt

from yuuagents.capabilities import DelegateManager, require_delegate_manager
from yuuagents.runtime_session import Session


@yt.tool(
    params={
        "seconds": "Seconds to wait (1-600, default 300). This blocks the current flow until done.",
    },
    description=(
        "Sleep for a while in the current flow. "
        "This is a normal long-running tool, so the host may still defer it to the background."
    ),
)
async def sleep(seconds: int = 300) -> str:
    seconds = max(1, min(600, int(seconds)))
    await asyncio.sleep(seconds)
    return f"等待 {seconds}s 完成。"


@yt.tool(
    params={
        "run_id": "Background run id returned by a deferred tool.",
        "limit": "Maximum number of recent stem events to show.",
        "max_chars": "Maximum total output length in characters after rendering and truncation.",
    },
    description=(
        "Inspect the current stem of a deferred/background run. "
        "Use this to check progress before deciding whether to wait, send input, defer a delegate, or cancel it."
    ),
)
async def inspect_background(
    run_id: str,
    limit: int = 200,
    max_chars: int = 4000,
    manager: DelegateManager = yt.depends(require_delegate_manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
) -> str:
    if parent is None:
        raise RuntimeError("background tools require an active session")
    return manager.inspect_run(
        parent=parent,
        run_id=run_id,
        limit=limit,
        max_chars=max_chars,
    )


@yt.tool(
    params={
        "run_id": "Background run id returned by a deferred tool.",
    },
    description="Cancel a deferred/background run immediately.",
)
async def cancel_background(
    run_id: str,
    manager: DelegateManager = yt.depends(require_delegate_manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
) -> str:
    if parent is None:
        raise RuntimeError("background tools require an active session")
    return manager.cancel_run(parent=parent, run_id=run_id)


@yt.tool(
    params={
        "run_id": "Background run id of a deferred run.",
        "data": (
            "Input to send into the background run. "
            "For bash this behaves like stdin/terminal input. For delegate it sends a normal message."
        ),
        "append_newline": "Whether to append Enter/newline after the input. Default true.",
    },
    description=(
        "Send input to a deferred/background run. "
        "Use this for interactive bash commands or to message a delegated agent running in the background."
    ),
)
async def input_background(
    run_id: str,
    data: str,
    append_newline: bool = True,
    manager: DelegateManager = yt.depends(require_delegate_manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
) -> str:
    if parent is None:
        raise RuntimeError("background tools require an active session")
    return await manager.input_run(
        parent=parent,
        run_id=run_id,
        data=data,
        append_newline=append_newline,
    )


@yt.tool(
    params={
        "run_id": "Background run id of a deferred delegate tool call.",
        "message": (
            "Optional prompt sent to the delegated agent. "
            "Use this to ask it to defer its own foreground tool and report progress."
        ),
    },
    description=(
        "Send a defer signal to a delegated agent that is currently running in the background. "
        "This is useful for nudging the delegated agent to stop waiting on its own tool and report progress."
    ),
)
async def defer_background(
    run_id: str,
    message: str = "",
    manager: DelegateManager = yt.depends(require_delegate_manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
) -> str:
    if parent is None:
        raise RuntimeError("background tools require an active session")
    return manager.defer_run(parent=parent, run_id=run_id, message=message)


@yt.tool(
    params={
        "run_ids": "List of one or more background run ids to wait for.",
    },
    description=(
        "Wait until one or more deferred/background runs finish. "
        "This blocks the current flow, but the host may still defer this wait tool itself to the background."
    ),
)
async def wait_background(
    run_ids: list[str],
    manager: DelegateManager = yt.depends(require_delegate_manager),
    parent: Session | None = yt.depends(lambda ctx: ctx.session),
) -> str:
    if parent is None:
        raise RuntimeError("background tools require an active session")
    return await manager.wait_runs(parent=parent, run_ids=run_ids)
