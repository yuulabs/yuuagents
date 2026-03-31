"""Helpers for reporting missing bundled runtime dependencies."""

from __future__ import annotations


def service_dependency_message(entrypoint: str, exc: ModuleNotFoundError) -> str:
    missing = exc.name or "an optional dependency"
    return (
        f"{entrypoint} requires bundled runtime dependencies "
        f"(missing: {missing}). Reinstall `yuuagents` "
        "and add `[docker]` / `[web]` as needed."
    )
