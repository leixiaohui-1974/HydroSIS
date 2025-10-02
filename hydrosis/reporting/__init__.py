"""Reporting utilities for HydroSIS outputs."""

from .charts import plot_hydrograph, plot_metric_bars
from .markdown import (
    MarkdownReportBuilder,
    generate_evaluation_report,
    summarise_aggregated_metrics,
)
from .narratives import qwen_narrative
from .templates import (
    EvaluationReportTemplate,
    ReportSection,
    default_evaluation_template,
    render_template,
)

__all__ = [
    "MarkdownReportBuilder",
    "generate_evaluation_report",
    "plot_hydrograph",
    "plot_metric_bars",
    "summarise_aggregated_metrics",
    "EvaluationReportTemplate",
    "ReportSection",
    "default_evaluation_template",
    "render_template",
    "qwen_narrative",
]
