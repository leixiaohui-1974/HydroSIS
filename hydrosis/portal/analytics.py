"""Utilities for summarising workflow results for presentation in the portal."""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, MutableMapping

from hydrosis.workflow import ScenarioRun, WorkflowResult

from .state import serialize_evaluation_outcome, serialize_model_score

Series = Iterable[float]


def summarise_workflow_result(result: WorkflowResult) -> Dict[str, object]:
    """Build a compact summary of a :class:`WorkflowResult` for API responses."""

    baseline_summary = _summarise_scenario_run(result.baseline)
    scenario_summaries = {
        scenario_id: _summarise_scenario_run(run)
        for scenario_id, run in result.scenarios.items()
    }

    deltas = _compute_deltas(baseline_summary["aggregated"], scenario_summaries)

    return {
        "baseline": baseline_summary,
        "scenarios": {
            scenario_id: {
                **summary,
                "delta_vs_baseline": deltas.get(scenario_id, {}),
            }
            for scenario_id, summary in scenario_summaries.items()
        },
        "overall_scores": [
            serialize_model_score(score) for score in result.overall_scores or []
        ],
        "evaluation_outcomes": [
            serialize_evaluation_outcome(outcome)
            for outcome in result.evaluation_outcomes
        ],
        "narrative": _build_narrative(scenario_summaries, deltas),
    }


def _summarise_scenario_run(run: ScenarioRun) -> Dict[str, MutableMapping[str, object]]:
    return {
        "scenario_id": run.scenario_id,
        "aggregated": {
            subbasin: _summarise_series(series)
            for subbasin, series in run.aggregated.items()
        },
        "zone_discharge": {
            zone: {
                subbasin: _summarise_series(series)
                for subbasin, series in flows.items()
            }
            for zone, flows in run.zone_discharge.items()
        },
    }


def _summarise_series(series: Series) -> Dict[str, object]:
    values = [float(value) for value in series]
    if not values:
        return {
            "total_volume": 0.0,
            "mean_flow": 0.0,
            "peak_flow": 0.0,
            "peak_index": None,
        }
    total = float(sum(values))
    peak_flow = max(values)
    peak_index = values.index(peak_flow)
    mean_flow = total / len(values)
    return {
        "total_volume": total,
        "mean_flow": mean_flow,
        "peak_flow": peak_flow,
        "peak_index": peak_index,
    }


def _compute_deltas(
    baseline: Mapping[str, Mapping[str, float]],
    scenarios: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    deltas: Dict[str, Dict[str, Dict[str, float]]] = {}
    for scenario_id, summary in scenarios.items():
        aggregated = summary.get("aggregated", {})
        scenario_deltas: Dict[str, Dict[str, float]] = {}
        for subbasin, stats in aggregated.items():
            if subbasin not in baseline:
                continue
            base_stats = baseline[subbasin]
            scenario_deltas[subbasin] = {
                key: float(stats.get(key, 0.0)) - float(base_stats.get(key, 0.0))
                for key in ("total_volume", "mean_flow", "peak_flow")
            }
        deltas[scenario_id] = scenario_deltas
    return deltas


def _build_narrative(
    scenarios: Mapping[str, Mapping[str, object]],
    deltas: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> str:
    if not scenarios:
        return "仅运行了基准情景，尚无情景对比可供总结。"

    parts = [
        "基准情景的主要指标已记录，可对比以下情景的变化：",
    ]
    for scenario_id, summary in scenarios.items():
        delta = deltas.get(scenario_id, {})
        if not delta:
            parts.append(f"- 情景 {scenario_id} 与基准在聚合流量上差异很小。")
            continue
        highlights = []
        for subbasin, stats in delta.items():
            volume_change = stats.get("total_volume", 0.0)
            peak_change = stats.get("peak_flow", 0.0)
            descriptor = []
            if abs(volume_change) > 1e-6:
                descriptor.append(
                    f"总径流{'增加' if volume_change >= 0 else '减少'} {abs(volume_change):.2f}"
                )
            if abs(peak_change) > 1e-6:
                descriptor.append(
                    f"峰值流量{'升高' if peak_change >= 0 else '降低'} {abs(peak_change):.2f}"
                )
            if descriptor:
                highlights.append(f"子流域 {subbasin}：{'，'.join(descriptor)}")
        if highlights:
            parts.append(f"- 情景 {scenario_id} 的主要变化：" + "；".join(highlights) + "。")
        else:
            parts.append(f"- 情景 {scenario_id} 的变化在统计量上不显著。")
    return "\n".join(parts)


__all__ = ["summarise_workflow_result"]
