"""Reporting utilities for HydroSIS outputs."""

from .charts import plot_hydrograph, plot_metric_bars
from .markdown import (
    MarkdownReportBuilder,
    generate_evaluation_report,
    summarise_aggregated_metrics,
)

__all__ = [
    "MarkdownReportBuilder",
    "generate_evaluation_report",
    "plot_hydrograph",
    "plot_metric_bars",
    "summarise_aggregated_metrics",
]
