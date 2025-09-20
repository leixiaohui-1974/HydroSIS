"""Tests ensuring example documentation is generated and validated."""
from __future__ import annotations

from pathlib import Path

from hydrosis.testing.example_documenter import generate_example_documentation


def test_generate_example_documentation(tmp_path: Path) -> None:
    """Running the generator should validate examples and emit Markdown docs."""

    paths = generate_example_documentation(tmp_path)

    expected_slugs = {
        "watershed_partition",
        "parameter_zone_assignment",
        "hand_calculated_run",
        "scenario_modification",
        "extended_runoff_models",
        "multi_model_comparison",
    }

    generated = {path.name for path in paths}
    assert generated == {f"{slug}.md" for slug in expected_slugs}

    for slug in expected_slugs:
        document = tmp_path / f"{slug}.md"
        assert document.exists(), f"Markdown document for {slug} was not created"
        content = document.read_text(encoding="utf-8")
        assert content.startswith("# "), "Markdown output should begin with a heading"
        assert "断言结论" in content
        assert "测试输入" in content
        assert "该文档由自动化示例验证程序生成" in content

def test_committed_example_docs_are_current(tmp_path: Path) -> None:
    """Ensure committed example Markdown matches regenerated output."""

    generated_paths = generate_example_documentation(tmp_path)
    for generated_path in generated_paths:
        committed_path = Path("docs/examples") / generated_path.name
        assert committed_path.exists(), f"Missing committed document for {generated_path.name}"

        generated = generated_path.read_text(encoding="utf-8")
        committed = committed_path.read_text(encoding="utf-8")
        assert generated == committed

