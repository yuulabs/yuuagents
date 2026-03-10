"""sleep — pause execution for a specified number of seconds."""

from __future__ import annotations

import asyncio

import yuutools as yt


@yt.tool(
    params={
        "seconds": "Number of seconds to wait (1-300)",
    },
    description="Wait for a specified number of seconds. Useful for polling patterns.",
)
async def sleep(seconds: int) -> str:
    seconds = max(1, min(300, int(seconds)))
    await asyncio.sleep(seconds)
    return f"Slept for {seconds} seconds."
