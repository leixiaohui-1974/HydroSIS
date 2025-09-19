"""Tools for evaluating HydroSIS simulations and comparing multiple models."""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from .metrics import DEFAULT_METRICS, DEFAULT_ORIENTATION


MetricFunction = Callable[[Sequence[float], Sequence[float]], float]


@dataclass
class ModelScore:
    """Performance summary for a particular model run."""

    model_id: str
    per_subbasin: Dict[str, Dict[str, float]]
    aggregated: Dict[str, float]


class SimulationEvaluator:
    """Evaluate simulated hydrographs against observations using multiple metrics."""

    def __init__(
        self,
        metrics: Mapping[str, MetricFunction] | None = None,
        orientations: Mapping[str, str] | None = None,
    ) -> None:
        self.metrics: Dict[str, MetricFunction] = dict(metrics or DEFAULT_METRICS)
        self.orientations: Dict[str, str] = dict(orientations or DEFAULT_ORIENTATION)
        for metric in self.metrics:
            if metric not in self.orientations:
                raise KeyError(
                    f"Orientation not provided for metric '{metric}'."
                )

    def metric_names(self) -> Iterable[str]:
        return self.metrics.keys()

    def metric_orientation(self, metric: str) -> str:
        return self.orientations[metric]

    def evaluate_series(
        self, simulated: Sequence[float], observed: Sequence[float]
    ) -> Dict[str, float]:
        return {name: func(simulated, observed) for name, func in self.metrics.items()}

    def evaluate_catchment(
        self,
        simulated: Mapping[str, Sequence[float]],
        observed: Mapping[str, Sequence[float]],
    ) -> Dict[str, Dict[str, float]]:
        scores: Dict[str, Dict[str, float]] = {}
        for subbasin, obs in observed.items():
            if subbasin not in simulated:
                continue
            scores[subbasin] = self.evaluate_series(simulated[subbasin], obs)
        return scores


class ModelComparator:
    """Compare the performance of multiple model simulations."""

    def __init__(
        self,
        evaluator: SimulationEvaluator | None = None,
        aggregator: Callable[[Sequence[float]], float] | None = None,
    ) -> None:
        self.evaluator = evaluator or SimulationEvaluator()
        self.aggregator = aggregator or statistics.mean

    def compare(
        self,
        simulations: Mapping[str, Mapping[str, Sequence[float]]],
        observations: Mapping[str, Sequence[float]],
    ) -> List[ModelScore]:
        results: List[ModelScore] = []
        for model_id, simulated in simulations.items():
            per_subbasin = self.evaluator.evaluate_catchment(simulated, observations)
            aggregated = self._aggregate_metrics(per_subbasin)
            results.append(
                ModelScore(
                    model_id=model_id,
                    per_subbasin=per_subbasin,
                    aggregated=aggregated,
                )
            )
        return results

    def rank(
        self,
        scores: Iterable[ModelScore],
        metric: str,
    ) -> List[ModelScore]:
        orientation = self.evaluator.metric_orientation(metric)

        def metric_value(score: ModelScore) -> float:
            value = score.aggregated.get(metric)
            if value is None:
                return float("inf")
            if orientation == "minabs":
                return abs(value)
            return value

        reverse = orientation == "max"
        return sorted(scores, key=metric_value, reverse=reverse)

    def _aggregate_metrics(
        self, per_subbasin: Mapping[str, Mapping[str, float]]
    ) -> Dict[str, float]:
        aggregated: Dict[str, float] = {}
        if not per_subbasin:
            return aggregated
        for metric in self.evaluator.metric_names():
            values: List[float] = []
            for sub_scores in per_subbasin.values():
                if metric not in sub_scores:
                    continue
                value = sub_scores[metric]
                if self.evaluator.metric_orientation(metric) == "minabs":
                    value = abs(value)
                values.append(value)
            if values:
                aggregated[metric] = self.aggregator(values)
        return aggregated


__all__ = ["ModelComparator", "ModelScore", "SimulationEvaluator"]
