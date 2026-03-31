from __future__ import annotations

from pathlib import Path


def test_readme_centers_flow_basin_and_taskhost() -> None:
    text = Path("README.md").read_text(encoding="utf-8")

    assert "Flow" in text
    assert "Basin" in text
    assert "TaskHost" in text

    for legacy_name in ("Session", "Pool", "LocalRun", "run_once"):
        assert legacy_name not in text

