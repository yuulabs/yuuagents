"""Helpers for reporting missing service-mode optional dependencies."""

from __future__ import annotations


def service_dependency_message(entrypoint: str, exc: ModuleNotFoundError) -> str:
    missing = exc.name or "an optional dependency"
    return (
        f"{entrypoint} requires service-mode optional dependencies "
        f"(missing: {missing}). Install `yuuagents[daemon]` "
        "and add `[docker]` / `[web]` as needed."
    )
