"""Tests for yuuagents.skills module."""

from __future__ import annotations

import tempfile
from pathlib import Path

from yuuagents.skills.discovery import render, scan
from yuuagents.types import SkillInfo


class TestScanFunction:
    """Tests for scan() function."""

    def test_scan_empty_list(self) -> None:
        """Should return empty list for empty paths."""
        skills = scan([])
        assert skills == []

    def test_scan_nonexistent_path(self) -> None:
        """Should return empty list for nonexistent paths."""
        skills = scan(["/nonexistent/path"])
        assert skills == []

    def test_scan_file_not_directory(self) -> None:
        """Should skip files, only process directories."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create a file instead of directory
            (tmp_path / "not-a-dir").write_text("content")
            skills = scan([tmp])
            assert skills == []

    def test_scan_directory_without_skill_md(self) -> None:
        """Should skip directories without SKILL.md."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Create directory without SKILL.md
            (tmp_path / "empty-skill").mkdir()
            skills = scan([tmp])
            assert skills == []

    def test_scan_single_skill(self) -> None:
        """Should find single skill."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            skill_dir = tmp_path / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: my-skill
description: A test skill
---

# My Skill

Content here.
""")
            skills = scan([tmp])
            assert len(skills) == 1
            assert skills[0].name == "my-skill"
            assert skills[0].description == "A test skill"

    def test_scan_multiple_skills(self) -> None:
        """Should find multiple skills."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # First skill
            skill1 = tmp_path / "skill-one"
            skill1.mkdir()
            (skill1 / "SKILL.md").write_text("""---
name: skill-one
description: First skill
---
Content
""")

            # Second skill
            skill2 = tmp_path / "skill-two"
            skill2.mkdir()
            (skill2 / "SKILL.md").write_text("""---
name: skill-two
description: Second skill
---
Content
""")

            skills = scan([tmp])
            assert len(skills) == 2
            names = {s.name for s in skills}
            assert names == {"skill-one", "skill-two"}

    def test_scan_finds_all_skills(self) -> None:
        """Should find all skills regardless of order."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create multiple skills
            for name in ["zebra", "alpha", "mike"]:
                skill_dir = tmp_path / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(f"""---
name: {name}
description: {name} skill
---
""")

            skills = scan([tmp])
            names = {s.name for s in skills}
            assert names == {"alpha", "mike", "zebra"}

    def test_scan_multiple_paths(self) -> None:
        """Should scan multiple paths."""
        with (
            tempfile.TemporaryDirectory() as tmp1,
            tempfile.TemporaryDirectory() as tmp2,
        ):
            path1 = Path(tmp1)
            path2 = Path(tmp2)

            # Skill in first path
            (path1 / "skill-a").mkdir()
            (path1 / "skill-a" / "SKILL.md").write_text(
                "---\nname: skill-a\ndescription: Skill A\n---\n"
            )

            # Skill in second path
            (path2 / "skill-b").mkdir()
            (path2 / "skill-b" / "SKILL.md").write_text(
                "---\nname: skill-b\ndescription: Skill B\n---\n"
            )

            skills = scan([tmp1, tmp2])
            assert len(skills) == 2
            names = {s.name for s in skills}
            assert names == {"skill-a", "skill-b"}

    def test_scan_expands_user_path(self) -> None:
        """Should expand user home directory."""
        # We can't fully test this without creating files in home,
        # but we can test it doesn't crash on ~ paths
        skills = scan(["~/.nonexistent_skills_dir"])
        assert skills == []

    def test_scan_skips_subdirectories_without_skill_md(self) -> None:
        """Should skip nested directories without SKILL.md."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create nested structure
            (tmp_path / "parent" / "child").mkdir(parents=True)
            (tmp_path / "parent" / "SKILL.md").write_text(
                "---\nname: parent\ndescription: Parent skill\n---\n"
            )

            skills = scan([tmp])
            # Should only find parent, not child
            assert len(skills) == 1
            assert skills[0].name == "parent"


class TestRenderFunction:
    """Tests for render() function."""

    def test_render_empty_list(self) -> None:
        """Should return empty string for empty list."""
        xml = render([])
        assert xml == ""

    def test_render_single_skill(self) -> None:
        """Should render single skill."""
        skills = [
            SkillInfo(
                name="git-worktree",
                description="Git worktrees",
                location="/path/to/skill",
            )
        ]
        xml = render(skills)
        assert "<available_skills>" in xml
        assert "</available_skills>" in xml
        assert "<skill>" in xml
        assert "</skill>" in xml
        assert "<name>git-worktree</name>" in xml
        assert "<description>Git worktrees</description>" in xml
        assert "<location>/path/to/skill</location>" in xml

    def test_render_multiple_skills(self) -> None:
        """Should render multiple skills."""
        skills = [
            SkillInfo(name="skill-a", description="First", location="/a"),
            SkillInfo(name="skill-b", description="Second", location="/b"),
        ]
        xml = render(skills)
        assert xml.count("<skill>") == 2
        assert xml.count("</skill>") == 2
        assert "skill-a" in xml
        assert "skill-b" in xml

    def test_render_empty_description(self) -> None:
        """Should handle empty description."""
        skills = [SkillInfo(name="test", description="", location="/path")]
        xml = render(skills)
        assert "<description></description>" in xml

    def test_render_special_characters(self) -> None:
        """Should handle special characters in fields."""
        skills = [
            SkillInfo(
                name="test-skill",
                description="A <complex> skill",
                location="/path/with spaces",
            )
        ]
        xml = render(skills)
        assert "A <complex> skill" in xml
        assert "/path/with spaces" in xml

    def test_render_structure(self) -> None:
        """Should produce valid XML structure containing skill data."""
        skills = [SkillInfo(name="test", description="Test skill", location="/test")]
        xml = render(skills)

        # Test structure without exact whitespace/line formatting
        assert "<available_skills>" in xml
        assert "</available_skills>" in xml
        assert "<skill>" in xml
        assert "</skill>" in xml
        assert "<name>test</name>" in xml
        assert "<description>Test skill</description>" in xml
        assert "<location>/test</location>" in xml


class TestSkillIntegration:
    """Integration tests for skills module."""

    def test_scan_and_render_roundtrip(self) -> None:
        """Scan followed by render should produce valid XML."""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create a skill
            skill_dir = tmp_path / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("""---
name: test-skill
description: A test skill for integration
---

# Test Skill

This is the content.
""")

            # Scan
            skills = scan([tmp])
            assert len(skills) == 1

            # Render
            xml = render(skills)
            assert "<name>test-skill</name>" in xml
            assert "A test skill for integration" in xml

    def test_skills_sorted_in_render(self) -> None:
        """Skills should maintain order from scan in render."""
        skills = [
            SkillInfo(name="zebra", description="Z", location="/z"),
            SkillInfo(name="alpha", description="A", location="/a"),
            SkillInfo(name="beta", description="B", location="/b"),
        ]
        xml = render(skills)

        # Order should be preserved
        zebra_pos = xml.find("zebra")
        alpha_pos = xml.find("alpha")
        beta_pos = xml.find("beta")

        assert zebra_pos < alpha_pos < beta_pos
