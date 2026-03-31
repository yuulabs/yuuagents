#!/usr/bin/env python3
"""Minimal SDK quick start for local agent development."""

from __future__ import annotations

import asyncio
import os

import yuullm
import yuutools as yt

from yuuagents import run_once


@yt.tool(description="Return the current working directory", params={})
async def current_directory() -> str:
    return os.getcwd()


async def main() -> None:
    llm = yuullm.YLLMClient(
        provider="openai",
        api_key_env="OPENAI_API_KEY",
        default_model="gpt-4o-mini",
    )

    result = await run_once(
        "Say hello, call current_directory, then report the directory.",
        llm=llm,
        tools=[current_directory],
        system="You are a concise coding assistant.",
    )

    print(result.output_text)
    print("steps:", result.steps)


if __name__ == "__main__":
    asyncio.run(main())
