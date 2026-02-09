"""Thin HTTP client that talks to the daemon over Unix Domain Socket."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class YAgentsClient:
    """Synchronous wrapper around httpx for CLI use."""

    def __init__(self, socket_path: str | Path) -> None:
        self._transport = httpx.HTTPTransport(uds=str(socket_path))
        self._client = httpx.Client(
            transport=self._transport,
            base_url="http://yagents",
            timeout=30.0,
        )

    def close(self) -> None:
        self._client.close()

    # -- health --

    def health(self) -> dict[str, Any]:
        return self._client.get("/health").json()

    # -- agents --

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._client.post("/api/agents", json=payload)
        resp.raise_for_status()
        return resp.json()

    def list_agents(self) -> list[dict[str, Any]]:
        return self._client.get("/api/agents").json()

    def status(self, task_id: str) -> dict[str, Any]:
        resp = self._client.get(f"/api/agents/{task_id}")
        resp.raise_for_status()
        return resp.json()

    def history(self, task_id: str) -> list[dict[str, Any]]:
        resp = self._client.get(f"/api/agents/{task_id}/history")
        resp.raise_for_status()
        return resp.json()

    def respond(self, task_id: str, content: str) -> dict[str, Any]:
        resp = self._client.post(
            f"/api/agents/{task_id}/input",
            json={"content": content},
        )
        resp.raise_for_status()
        return resp.json()

    def cancel(self, task_id: str) -> dict[str, Any]:
        resp = self._client.delete(f"/api/agents/{task_id}")
        resp.raise_for_status()
        return resp.json()

    # -- skills --

    def skills(self) -> list[dict[str, Any]]:
        return self._client.get("/api/skills").json()

    def scan_skills(self) -> list[dict[str, Any]]:
        resp = self._client.post("/api/skills/scan")
        return resp.json()

    # -- config --

    def get_config(self) -> dict[str, Any]:
        return self._client.get("/api/config").json()

    def reload_config(self) -> dict[str, Any]:
        resp = self._client.post("/api/config/reload")
        resp.raise_for_status()
        return resp.json()
