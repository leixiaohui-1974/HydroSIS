"""Tests for reporting utilities."""
from __future__ import annotations

import tempfile
from pathlib import Path

import unittest
from unittest import mock

from hydrosis import (
    MarkdownReportBuilder,
    ModelScore,
    SimulationEvaluator,
    generate_evaluation_report,
    plot_hydrograph,
    plot_metric_bars,
    summarise_aggregated_metrics,
)
from hydrosis.evaluation import DEFAULT_METRICS, DEFAULT_ORIENTATION


class ReportingTests(unittest.TestCase):
    """Validate Markdown report generation and plotting helpers."""

    def setUp(self) -> None:
        metrics = {name: func for name, func in DEFAULT_METRICS.items()}
        orientations = dict(DEFAULT_ORIENTATION)
        self.evaluator = SimulationEvaluator(metrics=metrics, orientations=orientations)

        self.scores = [
            ModelScore(
                model_id="baseline",
                per_subbasin={
                    "S1": {"rmse": 0.2, "mae": 0.1, "pbias": 1.5, "nse": 0.9},
                    "S2": {"rmse": 0.3, "mae": 0.15, "pbias": -2.0, "nse": 0.85},
                },
                aggregated={"rmse": 0.25, "mae": 0.12, "pbias": -0.5, "nse": 0.88},
            ),
            ModelScore(
                model_id="scenario",
                per_subbasin={
                    "S1": {"rmse": 0.18, "mae": 0.09, "pbias": -0.8, "nse": 0.92},
                    "S2": {"rmse": 0.4, "mae": 0.2, "pbias": -3.0, "nse": 0.8},
                },
                aggregated={"rmse": 0.29, "mae": 0.14, "pbias": -1.9, "nse": 0.86},
            ),
        ]

    def test_markdown_builder_outputs_expected_format(self) -> None:
        builder = MarkdownReportBuilder(title="Report")
        table = summarise_aggregated_metrics(self.scores, self.evaluator)
        builder.add_table(table)
        content = builder.to_markdown()

        self.assertIn("# Report", content)
        self.assertIn("| Model | RMSE | MAE", content)
        self.assertIn("baseline", content)

    def test_generate_evaluation_report_without_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.md"
            generate_evaluation_report(
                report_path,
                self.scores,
                self.evaluator,
                simulations=None,
                observations=None,
                description="测试报告",
                figures_directory=Path(tmpdir) / "figures",
                ranking_metric="rmse",
            )
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("测试报告", content)
            self.assertIn("总体评价指标", content)
            self.assertIn("基于 RMSE 的模型排序", content)

    def test_plot_helpers_create_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            hydro_path = Path(tmpdir) / "hydro.png"
            metric_path = Path(tmpdir) / "metric.png"

            created_hydro = plot_hydrograph(
                hydro_path,
                simulations={"baseline": [1.0, 2.0, 1.5]},
                observed=[0.9, 2.1, 1.4],
                title="Hydrograph",
            )
            self.assertTrue(created_hydro.exists())
            self.assertIn(created_hydro.suffix, {".png", ".svg"})

            created_metric = plot_metric_bars(
                metric_path,
                self.scores,
                self.evaluator,
                metric="rmse",
                title="RMSE",
            )
            self.assertTrue(created_metric.exists())
            self.assertIn(created_metric.suffix, {".png", ".svg"})

    def test_generate_report_uses_relative_paths_for_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "reports"
            figures_dir = Path(tmpdir) / "figures"
            figures_dir.mkdir(parents=True, exist_ok=True)
            metric_figure = figures_dir / "metric_rmse.svg"
            hydro_figure = figures_dir / "hydrograph_S1.svg"
            metric_figure.write_text("<svg></svg>", encoding="utf-8")
            hydro_figure.write_text("<svg></svg>", encoding="utf-8")

            with mock.patch(
                "hydrosis.reporting.markdown._generate_metric_figures",
                return_value=[metric_figure],
            ), mock.patch(
                "hydrosis.reporting.markdown._generate_hydrograph_figures",
                return_value=[hydro_figure],
            ):
                report_path = report_dir / "evaluation.md"
                generate_evaluation_report(
                    report_path,
                    self.scores,
                    self.evaluator,
                    simulations={"baseline": {"S1": [1.0, 1.1, 1.2]}},
                    observations={"S1": [0.9, 1.0, 1.05]},
                    figures_directory=figures_dir,
                )
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("![metric_rmse](../figures/metric_rmse.svg)", content)
            self.assertIn("![hydrograph_S1](../figures/hydrograph_S1.svg)", content)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
