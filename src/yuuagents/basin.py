"""Basin — index of all live flows in the current process."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from attrs import define, field

from yuuagents.core.flow import Flow


@define
class Basin:
    """Process-local registry of live flows.

    Basin is intentionally dumb: it indexes flows by ID and offers direct
    lookup. It does not understand tool semantics.
    """

    _flows: dict[str, Flow[Any, Any]] = field(factory=dict, init=False)

    def register(self, flow: Flow[Any, Any]) -> Flow[Any, Any]:
        self._flows[flow.id] = flow
        return flow

    def forget(self, flow_id: str) -> None:
        self._flows.pop(flow_id, None)

    def get(self, flow_id: str) -> Flow[Any, Any] | None:
        return self._flows.get(flow_id)

    def require(self, flow_id: str) -> Flow[Any, Any]:
        flow = self.get(flow_id)
        if flow is None:
            raise KeyError(flow_id)
        return flow

    def __contains__(self, flow_id: str) -> bool:
        return flow_id in self._flows

    def iter_flows(self) -> Iterable[Flow[Any, Any]]:
        return self._flows.values()
