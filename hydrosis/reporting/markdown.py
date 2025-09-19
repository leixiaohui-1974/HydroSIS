"""Markdown report generation for HydroSIS evaluations."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Mapping, MutableSequence, Sequence

from hydrosis.evaluation import ModelComparator, ModelScore, SimulationEvaluator

from .charts import plot_hydrograph, plot_metric_bars
from .templates import EvaluationReportTemplate, render_template, default_evaluation_template


@dataclass
class TableData:
    """Internal representation of a markdown table."""

    headers: Sequence[str]
    rows: Sequence[Sequence[str]]


class MarkdownReportBuilder:
    """Utility class for constructing markdown documents programmatically."""

    def __init__(self, title: str | None = None) -> None:
        self._lines: MutableSequence[str] = []
        if title:
            self.add_heading(title, level=1)

    def add_heading(self, text: str, level: int = 1) -> None:
        self._lines.append(f"{'#' * max(level, 1)} {text}")
        self._lines.append("")

    def add_paragraph(self, text: str) -> None:
        self._lines.append(text)
        self._lines.append("")

    def add_list(self, items: Iterable[str], ordered: bool = False) -> None:
        for idx, item in enumerate(items, start=1):
            prefix = f"{idx}. " if ordered else "- "
            self._lines.append(f"{prefix}{item}")
        self._lines.append("")

    def add_table(self, table: TableData) -> None:
        headers = " | ".join(table.headers)
        separator = " | ".join(["---"] * len(table.headers))
        self._lines.append(f"| {headers} |")
        self._lines.append(f"| {separator} |")
        for row in table.rows:
            values = " | ".join(row)
            self._lines.append(f"| {values} |")
        self._lines.append("")

    def add_image(self, path: Path, alt_text: str) -> None:
        self._lines.append(f"![{alt_text}]({Path(path).as_posix()})")
        self._lines.append("")

    def add_horizontal_rule(self) -> None:
        self._lines.append("---")
        self._lines.append("")

    def extend(self, lines: Iterable[str]) -> None:
        for line in lines:
            self._lines.append(line)
        self._lines.append("")

    def to_markdown(self) -> str:
        return "\n".join(self._lines).strip() + "\n"

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_markdown(), encoding="utf-8")


def summarise_aggregated_metrics(
    scores: Sequence[ModelScore], evaluator: SimulationEvaluator
) -> TableData:
    """Construct a table summarising aggregated metrics by model."""

    metrics = list(evaluator.metric_names())
    headers = ["Model"] + [metric.upper() for metric in metrics]
    rows: List[List[str]] = []
    for score in scores:
        row = [score.model_id]
        for metric in metrics:
            if metric not in score.aggregated:
                row.append("-")
                continue
            value = score.aggregated[metric]
            if evaluator.metric_orientation(metric) == "minabs":
                value = abs(value)
            row.append(f"{value:.4f}")
        rows.append(row)
    return TableData(headers=headers, rows=rows)


def _generate_metric_figures(
    scores: Sequence[ModelScore],
    evaluator: SimulationEvaluator,
    figures_directory: Path,
) -> List[Path]:
    figures: List[Path] = []
    for metric in evaluator.metric_names():
        try:
            figure = plot_metric_bars(
                figures_directory / f"metric_{metric}.png",
                scores,
                evaluator,
                metric,
                title=f"Aggregated {metric.upper()} by Model",
            )
            figures.append(figure)
        except RuntimeError:
            break
    return figures


def _generate_hydrograph_figures(
    simulations: Mapping[str, Mapping[str, Sequence[float]]],
    observations: Mapping[str, Sequence[float]] | None,
    figures_directory: Path,
) -> List[Path]:
    if observations is None:
        return []

    figures: List[Path] = []
    for subbasin_id in sorted(observations):
        observed = observations[subbasin_id]
        relevant_simulations = {
            model_id: hydrographs[subbasin_id]
            for model_id, hydrographs in simulations.items()
            if subbasin_id in hydrographs
        }
        if not relevant_simulations:
            continue
        try:
            figure = plot_hydrograph(
                figures_directory / f"hydrograph_{subbasin_id}.png",
                relevant_simulations,
                observed,
                title=f"Hydrograph at {subbasin_id}",
            )
            figures.append(figure)
        except RuntimeError:
            break
    return figures


def generate_evaluation_report(
    output_path: Path,
    scores: Sequence[ModelScore],
    evaluator: SimulationEvaluator,
    simulations: Mapping[str, Mapping[str, Sequence[float]]] | None = None,
    observations: Mapping[str, Sequence[float]] | None = None,
    description: str | None = None,
    figures_directory: Path | None = None,
    ranking_metric: str | None = None,
    template: EvaluationReportTemplate | None = None,
    narrative_callback: Callable[[str], str] | None = None,
    template_context: Mapping[str, str] | None = None,
) -> Path:
    """Create a markdown report summarising model evaluation results."""

    output_path = Path(output_path)
    if figures_directory is None:
        figures_directory = output_path.parent / "figures"
    figures_directory.mkdir(parents=True, exist_ok=True)

    report_directory = output_path.parent

    def _relative_figure_path(path: Path) -> Path:
        try:
            return path.relative_to(report_directory)
        except ValueError:
            return Path(os.path.relpath(path, report_directory))

    builder = MarkdownReportBuilder(title="HydroSIS 模型评估报告")
    if description:
        builder.add_paragraph(description)

    table = summarise_aggregated_metrics(scores, evaluator)
    builder.add_heading("总体评价指标", level=2)
    builder.add_table(table)

    comparator = ModelComparator(evaluator)
    metrics = list(evaluator.metric_names())
    if not ranking_metric and metrics:
        ranking_metric = metrics[0]
    if ranking_metric and metrics:
        try:
            ranking = comparator.rank(list(scores), ranking_metric)
            builder.add_heading(
                f"基于 {ranking_metric.upper()} 的模型排序", level=2
            )
            builder.add_list(
                [
                    f"{score.model_id}: {score.aggregated.get(ranking_metric, 'N/A')}"
                    for score in ranking
                ],
                ordered=True,
            )
        except KeyError:
            builder.add_paragraph(
                f"指标 {ranking_metric} 未在评估指标中定义，无法生成排序。"
            )

    metric_figures = _generate_metric_figures(scores, evaluator, figures_directory)
    if metric_figures:
        builder.add_heading("指标图表", level=2)
        for figure in metric_figures:
            builder.add_image(_relative_figure_path(figure), alt_text=figure.stem)
    else:
        builder.add_paragraph("未生成指标图表（可能缺少 matplotlib 依赖）。")

    if simulations:
        hydrograph_figures = _generate_hydrograph_figures(
            simulations, observations, figures_directory
        )
        if hydrograph_figures:
            builder.add_heading("子流域径流过程对比", level=2)
            for figure in hydrograph_figures:
                builder.add_image(_relative_figure_path(figure), alt_text=figure.stem)
        elif observations is not None:
            builder.add_paragraph("未生成径流对比图（可能缺少 matplotlib 依赖）。")

    if template is None:
        template = default_evaluation_template()
    render_template(
        builder,
        template,
        template_context or {},
        narrator=narrative_callback,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    builder.write(output_path)
    return output_path


__all__ = [
    "MarkdownReportBuilder",
    "generate_evaluation_report",
    "summarise_aggregated_metrics",
]
