"""Tests for the comprehensive feature runner and documentation generator."""
from __future__ import annotations

from pathlib import Path

from datetime import datetime, timezone

from hydrosis.testing.full_feature_runner import run_full_feature_checks


def test_full_feature_runner_generates_markdown(tmp_path: Path) -> None:
    """Ensure the feature runner exercises the workflow and emits documentation."""

    output_path = tmp_path / "feature_report.md"
    results, documentation_path = run_full_feature_checks(output_path)

    assert documentation_path.exists(), "Markdown documentation was not created"
    content = documentation_path.read_text(encoding="utf-8")

    assert "HydroSIS 功能测试报告" in content
    assert results, "Feature results should not be empty"
    section_titles = [result.name for result in results]

    assert any("模型配置" in title for title in section_titles)
    assert any("情景" in title for title in section_titles)
    assert any("报告生成" in title for title in section_titles)

    for required_phrase in ["断言结论", "测试输入", "关键输出与校验"]:
        assert required_phrase in content

def test_committed_feature_report_is_current(tmp_path: Path) -> None:
    """Regenerated feature report should match the committed documentation."""

    fixed_timestamp = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    output_path = tmp_path / "feature_report.md"
    _, generated_path = run_full_feature_checks(
        output_path, generated_at=fixed_timestamp
    )

    generated = generated_path.read_text(encoding="utf-8")
    normalised = generated.replace(
        generated_path.as_posix(), "docs/test_documentation.md"
    )

    expected = Path("docs/test_documentation.md").read_text(encoding="utf-8")
    assert normalised == expected