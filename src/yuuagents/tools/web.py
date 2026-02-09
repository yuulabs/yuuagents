"""web_search — Tavily API search returning LLM-friendly text."""

from __future__ import annotations

import importlib

import yuutools as yt


@yt.tool(
    params={
        "query": "Search query string",
        "max_results": "Maximum number of results (default 5, max 10)",
    },
    description="Search the web using Tavily API. Returns LLM-friendly text results.",
)
async def web_search(
    query: str,
    max_results: int = 5,
    api_key: str = yt.depends(lambda ctx: ctx.tavily_api_key),
) -> str:
    max_results = max(1, min(max_results, 10))
    tavily = importlib.import_module("tavily")
    client = tavily.AsyncTavilyClient(api_key=api_key)
    resp = await client.search(query=query, max_results=max_results)

    parts: list[str] = []
    for i, r in enumerate(resp.get("results", []), 1):
        title = r.get("title", "")
        url = r.get("url", "")
        snippet = r.get("content", "")
        parts.append(f"[{i}] {title}\n    {url}\n    {snippet}")
    return "\n\n".join(parts) if parts else "No results found."
