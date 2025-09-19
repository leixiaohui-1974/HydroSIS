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
from .utils import accumulate_subbasin_flows
from .workflow import EvaluationOutcome, ScenarioRun, WorkflowResult, run_workflow

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
    "accumulate_subbasin_flows",
    "EvaluationOutcome",
    "ScenarioRun",
    "WorkflowResult",
    "run_workflow",
]