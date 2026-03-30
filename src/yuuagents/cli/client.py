"""Thin HTTP client that talks to the daemon over Unix Domain Socket."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx


class DaemonNotRunningError(Exception):
    """Raised when the CLI cannot connect to the daemon."""

    def __init__(self, socket_path: str) -> None:
        self.socket_path = socket_path
        super().__init__(
            f"Cannot connect to daemon (socket: {socket_path}).\n"
            "Is the daemon running? Start it with:\n"
            "\n"
            "    yagents up\n"
        )


class YAgentsClient:
    """Synchronous wrapper around httpx for CLI use."""

    def __init__(self, socket_path: str | Path) -> None:
        self._socket_path = str(socket_path)
        self._transport = httpx.HTTPTransport(uds=self._socket_path)
        self._client = httpx.Client(
            transport=self._transport,
            base_url="http://yagents",
            timeout=30.0,
        )

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send an HTTP request, translating connection errors."""
        try:
            return self._client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise DaemonNotRunningError(self._socket_path) from exc

    def close(self) -> None:
        self._client.close()

    # -- health --

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health").json()

    def shutdown(self) -> dict[str, Any]:
        resp = self._request("POST", "/api/shutdown")
        resp.raise_for_status()
        return resp.json()

    # -- agents --

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._request("POST", "/api/agents", json=payload)
        resp.raise_for_status()
        return resp.json()

    def list_agents(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/agents").json()

    def status(self, task_id: str) -> dict[str, Any]:
        resp = self._request("GET", f"/api/agents/{task_id}")
        resp.raise_for_status()
        return resp.json()

    def history(self, task_id: str) -> list[dict[str, Any]]:
        resp = self._request("GET", f"/api/agents/{task_id}/history")
        resp.raise_for_status()
        return resp.json()

    def respond(self, task_id: str, message: list[Any]) -> dict[str, Any]:
        resp = self._request(
            "POST",
            f"/api/agents/{task_id}/input",
            json={"message": message},
        )
        resp.raise_for_status()
        return resp.json()

    def cancel(self, task_id: str) -> dict[str, Any]:
        resp = self._request("DELETE", f"/api/agents/{task_id}")
        resp.raise_for_status()
        return resp.json()

    # -- config --

    def get_config(self) -> dict[str, Any]:
        return self._request("GET", "/api/config").json()

    def reload_config(self) -> dict[str, Any]:
        resp = self._request("POST", "/api/config/reload")
        resp.raise_for_status()
        return resp.json()
