"""High-level workflow orchestration utilities for HydroSIS."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from .config import ComparisonPlanConfig, EvaluationConfig, ModelConfig
from .evaluation import ModelComparator, ModelScore, SimulationEvaluator
from .evaluation.metrics import DEFAULT_METRICS, DEFAULT_ORIENTATION
from .io.outputs import write_simulation_results
from .model import HydroSISModel
from .reporting import EvaluationReportTemplate, generate_evaluation_report


@dataclass
class ScenarioRun:
    """Container holding the outputs of a single scenario simulation."""

    scenario_id: str
    local: Dict[str, List[float]]
    aggregated: Dict[str, List[float]]
    zone_discharge: Dict[str, Dict[str, List[float]]]


@dataclass
class EvaluationOutcome:
    """Evaluation results for a specific comparison plan."""

    plan: ComparisonPlanConfig
    scores: List[ModelScore]
    ranking: List[ModelScore]
    ranking_metric: Optional[str]


@dataclass
class WorkflowResult:
    """Return structure for :func:`run_workflow`."""

    baseline: ScenarioRun
    scenarios: Dict[str, ScenarioRun] = field(default_factory=dict)
    overall_scores: Optional[List[ModelScore]] = None
    evaluation_outcomes: List[EvaluationOutcome] = field(default_factory=list)
    report_path: Optional[Path] = None


def _instantiate_model(config: ModelConfig) -> HydroSISModel:
    """Create a new :class:`HydroSISModel` instance from configuration."""

    return HydroSISModel.from_config(config)


def _run_model(
    scenario_id: str,
    model: HydroSISModel,
    forcing: Mapping[str, Sequence[float]],
) -> ScenarioRun:
    """Execute a model run and package the results."""

    local = model.run(forcing)
    aggregated = model.accumulate_discharge(local)
    zone_discharge = model.parameter_zone_discharge(local)
    return ScenarioRun(
        scenario_id=scenario_id,
        local={sid: list(series) for sid, series in local.items()},
        aggregated={sid: list(series) for sid, series in aggregated.items()},
        zone_discharge={
            zone: {sid: list(series) for sid, series in flows.items()}
            for zone, flows in zone_discharge.items()
        },
    )


def _build_evaluator(config: EvaluationConfig | None) -> SimulationEvaluator:
    """Create an evaluator that honours the metrics listed in the configuration."""

    if config is None:
        return SimulationEvaluator()

    selected_metrics: Dict[str, Callable[[Sequence[float], Sequence[float]], float]] = {}
    selected_orientations: Dict[str, str] = {}
    for metric in config.metrics:
        if metric not in DEFAULT_METRICS:
            raise KeyError(f"Unsupported metric '{metric}' requested in evaluation")
        selected_metrics[metric] = DEFAULT_METRICS[metric]
        selected_orientations[metric] = DEFAULT_ORIENTATION[metric]
    return SimulationEvaluator(metrics=selected_metrics, orientations=selected_orientations)


def _filter_series(
    series: Mapping[str, Sequence[float]], subbasins: Optional[Iterable[str]]
) -> Dict[str, List[float]]:
    if subbasins is None:
        return {sid: list(values) for sid, values in series.items()}
    allowed = set(subbasins)
    return {sid: list(series[sid]) for sid in allowed if sid in series}


def _collect_candidate_simulations(
    baseline: ScenarioRun,
    scenarios: Mapping[str, ScenarioRun],
) -> Dict[str, Dict[str, List[float]]]:
    simulations: Dict[str, Dict[str, List[float]]] = {
        "baseline": baseline.aggregated,
    }
    for scenario_id, result in scenarios.items():
        simulations[scenario_id] = result.aggregated
    return simulations


def _resolve_reference_series(
    reference: str,
    simulations: Mapping[str, Mapping[str, Sequence[float]]],
    observations: Mapping[str, Sequence[float]] | None,
) -> Mapping[str, Sequence[float]]:
    if reference == "observed":
        if observations is None:
            raise ValueError("Observed discharge data required for evaluation")
        return observations
    if reference not in simulations:
        raise KeyError(f"Reference model '{reference}' not available for comparison")
    return simulations[reference]


def _evaluate_plan(
    plan: ComparisonPlanConfig,
    comparator: ModelComparator,
    simulations: Mapping[str, Mapping[str, Sequence[float]]],
    observations: Mapping[str, Sequence[float]] | None,
) -> EvaluationOutcome:
    plan_simulations: Dict[str, Dict[str, List[float]]] = {}
    for model_id in plan.models:
        if model_id not in simulations:
            continue
        plan_simulations[model_id] = _filter_series(
            simulations[model_id], plan.subbasins
        )

    reference_series = _resolve_reference_series(plan.reference, simulations, observations)
    reference_filtered = _filter_series(reference_series, plan.subbasins)

    if not plan_simulations:
        raise ValueError(f"No simulations available for comparison plan '{plan.id}'")
    if not reference_filtered:
        raise ValueError(
            f"Reference series for comparison plan '{plan.id}' does not cover requested subbasins"
        )

    scores = comparator.compare(plan_simulations, reference_filtered)
    ranking_metric = plan.ranking_metric or next(
        iter(comparator.evaluator.metric_names()), None
    )
    ranking = (
        comparator.rank(scores, ranking_metric)
        if ranking_metric and comparator.evaluator.metric_names()
        else list(scores)
    )

    return EvaluationOutcome(
        plan=plan,
        scores=scores,
        ranking=ranking,
        ranking_metric=ranking_metric,
    )


def run_workflow(
    config: ModelConfig,
    forcing: Mapping[str, Sequence[float]],
    observations: Mapping[str, Sequence[float]] | None = None,
    scenario_ids: Optional[Sequence[str]] = None,
    persist_outputs: bool = False,
    generate_report: bool = False,
    narrative_callback: Callable[[str], str] | None = None,
    report_template: EvaluationReportTemplate | None = None,
    template_context: Mapping[str, str] | None = None,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> WorkflowResult:
    """Run baseline and scenario simulations and optionally evaluate them.

    Parameters
    ----------
    config:
        Parsed model configuration describing delineation, parameter zones, I/O, etc.
    forcing:
        Mapping from subbasin identifiers to precipitation (or runoff) time-series.
    observations:
        Optional observed discharge series for evaluation; keyed by subbasin.
    scenario_ids:
        Optional sequence of scenario identifiers to execute. If omitted, all
        scenarios present in the configuration are evaluated.
    persist_outputs:
        When ``True``, aggregated discharge time-series are written to the results
        directory specified by the configuration.
    generate_report:
        When ``True`` and evaluation data are available, a Markdown report is
        emitted to the configured reports directory.
    narrative_callback:
        Optional callable used to turn模板提示语 into自然语言段落，例如接入大模型生成摘要。
    report_template:
        Custom evaluation report template. Defaults to :func:`default_evaluation_template`.
    template_context:
        Pre-filled段落文本，当提供时优先使用而不会触发 ``narrative_callback``。
    llm_provider:
        当未显式提供 ``narrative_callback`` 时，可指定自动注入的大模型提供方。
        支持从配置或 ``HYDROSIS_LLM_PROVIDER`` 环境变量中读取，当前实现内置
        了对千问 (Qwen) 的兼容。
    llm_model:
        当 ``llm_provider`` 为千问时，可用于覆盖默认模型名称。若未提供，则
        会尝试使用 ``HYDROSIS_LLM_MODEL`` 环境变量或回退至 ``qwen-plus``。
    progress_callback:
        可选回调函数，用于实时接收阶段性进度更新。回调的第一个参数为阶段
        名称，第二个参数为包含阶段信息的字典。
    """

    def _notify(stage: str, **payload: object) -> None:
        if progress_callback is None:
            return
        progress_callback(stage, payload)

    requested_ids = (
        list(scenario_ids)
        if scenario_ids is not None
        else [scenario.id for scenario in config.scenarios]
    )

    _notify("workflow", phase="start", scenario_total=len(requested_ids))

    baseline_model = _instantiate_model(config)
    _notify("baseline", phase="start", subbasins=len(forcing))
    baseline_result = _run_model("baseline", baseline_model, forcing)
    _notify(
        "baseline",
        phase="complete",
        subbasins=len(baseline_result.local),
    )

    scenario_results: Dict[str, ScenarioRun] = {}

    for index, scenario_id in enumerate(requested_ids, start=1):
        scenario_cfg = copy.deepcopy(config)
        scenario_model = _instantiate_model(scenario_cfg)
        scenario_cfg.apply_scenario(scenario_id, scenario_model.subbasins.values())
        _notify(
            "scenario",
            phase="start",
            scenario_id=scenario_id,
            index=index,
            total=len(requested_ids),
        )
        scenario_results[scenario_id] = _run_model(
            scenario_id, scenario_model, forcing
        )
        _notify(
            "scenario",
            phase="complete",
            scenario_id=scenario_id,
            index=index,
            total=len(requested_ids),
            subbasins=len(scenario_results[scenario_id].local),
        )

    if persist_outputs:
        _notify("persistence", phase="start", scenario_total=len(requested_ids))
        base_directory = config.io.results_directory
        write_simulation_results(base_directory / "baseline", baseline_result.aggregated)
        for scenario_id, result in scenario_results.items():
            write_simulation_results(
                base_directory / scenario_id, result.aggregated
            )
        _notify("persistence", phase="complete", scenario_total=len(requested_ids))

    evaluator = _build_evaluator(config.evaluation)
    comparator = ModelComparator(evaluator)

    simulations = _collect_candidate_simulations(baseline_result, scenario_results)

    overall_scores: Optional[List[ModelScore]] = None
    evaluation_outcomes: List[EvaluationOutcome] = []
    report_path: Optional[Path] = None

    if observations is not None:
        _notify("evaluation", phase="start", scenario_total=len(requested_ids))
        overall_scores = comparator.compare(simulations, observations)

        report_context: Dict[str, str] = dict(template_context or {})

        provider_hint = (
            llm_provider
            or config.evaluation.llm_provider
            or os.environ.get("HYDROSIS_LLM_PROVIDER")
        )
        model_hint = (
            llm_model
            or config.evaluation.llm_model
            or os.environ.get("HYDROSIS_LLM_MODEL")
        )
        if provider_hint:
            report_context.setdefault("llm_provider", provider_hint)
        if model_hint:
            report_context.setdefault("llm_model", model_hint)

        if config.evaluation is not None:
            total_plans = len(config.evaluation.comparisons)
            for index, plan in enumerate(config.evaluation.comparisons, start=1):
                _notify(
                    "evaluation_plan",
                    phase="start",
                    plan_id=plan.id,
                    index=index,
                    total=total_plans,
                )
                outcome = _evaluate_plan(plan, comparator, simulations, observations)
                evaluation_outcomes.append(outcome)
                _notify(
                    "evaluation_plan",
                    phase="complete",
                    plan_id=plan.id,
                    index=index,
                    total=total_plans,
                )

            if not provider_hint:
                if overall_scores:
                    model_ids = ", ".join(score.model_id for score in overall_scores)
                    metrics = ", ".join(
                        name.upper() for name in comparator.evaluator.metric_names()
                    )
                    report_context.setdefault(
                        "模型运行概述",
                        (
                            f"本次评估比较了 {len(overall_scores)} 套模拟方案（{model_ids}），"
                            f"评价指标包括 {metrics}。"
                        ),
                    )
                if evaluation_outcomes:
                    primary = evaluation_outcomes[0]
                    ranking_ids = [score.model_id for score in primary.ranking]
                    if ranking_ids:
                        report_context.setdefault(
                            "关键发现",
                            (
                                f"在 {primary.plan.description or primary.plan.id} 中，排序为 "
                                f"{' > '.join(ranking_ids)}。"
                            ),
                        )
                report_context.setdefault(
                    "后续建议",
                    "可继续针对关键指标开展参数分区校准或扩展新的情景对比。",
                )

            if generate_report:
                _notify("report", phase="start")
                report_directory = (
                    config.io.reports_directory
                    if config.io.reports_directory is not None
                    else config.io.results_directory / "reports"
                )
                figures_directory = (
                    config.io.figures_directory
                    if config.io.figures_directory is not None
                    else config.io.results_directory / "figures"
                )
                report_path = generate_evaluation_report(
                    report_directory / "evaluation.md",
                    overall_scores,
                    comparator.evaluator,
                    simulations=simulations,
                    observations=observations,
                    description="自动生成的模型精度评价报告",
                    figures_directory=figures_directory,
                    template=report_template,
                    narrative_callback=narrative_callback,
                    template_context=report_context,
                    llm_provider=provider_hint,
                )
                _notify("report", phase="complete", path=str(report_path))

        _notify("evaluation", phase="complete", scenario_total=len(requested_ids))

    _notify(
        "workflow",
        phase="complete",
        scenario_total=len(requested_ids),
        evaluation_performed=observations is not None,
    )

    return WorkflowResult(
        baseline=baseline_result,
        scenarios=scenario_results,
        overall_scores=overall_scores,
        evaluation_outcomes=evaluation_outcomes,
        report_path=report_path,
    )


__all__ = [
    "EvaluationOutcome",
    "ScenarioRun",
    "WorkflowResult",
    "run_workflow",
]
