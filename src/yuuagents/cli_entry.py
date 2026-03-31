"""Lazy console entrypoint for service-mode CLI."""

from __future__ import annotations

from importlib import import_module

from yuuagents.service_requirements import service_dependency_message


def main() -> None:
    try:
        cli = import_module("yuuagents.cli.main").cli
    except ModuleNotFoundError as exc:
        raise SystemExit(service_dependency_message("yagents", exc)) from exc
    cli()
