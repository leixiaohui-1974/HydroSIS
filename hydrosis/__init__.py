"""HydroSIS distributed hydrological modelling framework."""

from .config import ModelConfig
from .evaluation import ModelComparator, ModelScore, SimulationEvaluator
from .reporting import (
    MarkdownReportBuilder,
    generate_evaluation_report,
    plot_hydrograph,
    plot_metric_bars,
    summarise_aggregated_metrics,
)
from .model import HydroSISModel

__all__ = [
    "HydroSISModel",
    "ModelConfig",
    "ModelComparator",
    "ModelScore",
    "SimulationEvaluator",
    "MarkdownReportBuilder",
    "generate_evaluation_report",
    "plot_hydrograph",
    "plot_metric_bars",
    "summarise_aggregated_metrics",
]
