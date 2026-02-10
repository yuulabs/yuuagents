"""Tests for yagents CLI dotenv support."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from yuuagents.cli.main import _find_dotenv, _load_dotenv_file


def test_load_dotenv_file_sets_only_missing_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "EXISTING=from_file",
                "NEW=value",
                "QUOTED='a b'",
                "export EXPORTED=ok",
                "# COMMENTED=skip",
                "",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("EXISTING", "from_env")
    monkeypatch.delenv("NEW", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    monkeypatch.delenv("EXPORTED", raising=False)

    _load_dotenv_file(env_file)

    assert os.environ["EXISTING"] == "from_env"
    assert os.environ["NEW"] == "value"
    assert os.environ["QUOTED"] == "a b"
    assert os.environ["EXPORTED"] == "ok"


def test_load_dotenv_file_rejects_invalid_lines(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("NOT_AN_ASSIGNMENT\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        _load_dotenv_file(env_file)


def test_find_dotenv_searches_up_to_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    project = home / "proj"
    nested = project / "a" / "b"
    nested.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))

    env_file = project / ".env"
    env_file.write_text("X=1\n", encoding="utf-8")

    found = _find_dotenv(nested)
    assert found == env_file.resolve()


def test_find_dotenv_does_not_search_above_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    nested = home / "proj"
    nested.mkdir(parents=True)

    monkeypatch.setenv("HOME", str(home))

    (tmp_path / ".env").write_text("X=1\n", encoding="utf-8")

    assert _find_dotenv(nested) is None
