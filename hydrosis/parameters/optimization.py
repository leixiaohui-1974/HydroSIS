"""Optimization utilities for parameter zones."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, Sequence

from .zone import ParameterZone


ObjectiveSense = str


@dataclass
class ObjectiveDefinition:
    """Description of a single optimisation objective."""

    id: str
    weight: float = 1.0
    sense: ObjectiveSense = "min"


@dataclass
class OptimizationResult:
    """Container for optimisation outputs."""

    best_parameters: Mapping[str, Mapping[str, float]]
    objective_scores: Mapping[str, float]
    history: Sequence[Mapping[str, float]] = field(default_factory=list)


ParameterSampler = Callable[[ParameterZone], Iterable[Mapping[str, float]]]
EvaluationCallback = Callable[[Mapping[str, Mapping[str, float]]], Mapping[str, float]]


class ParameterZoneOptimizer:
    """Perform multi-objective optimisation on parameter zones.

    The optimiser expects an evaluation callback that returns objective
    values given a complete set of zone parameters.  Candidate parameter
    combinations are provided by ``sampler`` functions which can generate
    Latin Hypercube samples, Sobol sequences or any other sampling scheme.
    """

    def __init__(
        self,
        zones: Sequence[ParameterZone],
        evaluation: EvaluationCallback,
        objectives: Sequence[ObjectiveDefinition],
    ) -> None:
        self.zones = list(zones)
        self.evaluation = evaluation
        self.objectives = list(objectives)

    def optimise(
        self,
        samplers: Mapping[str, ParameterSampler],
        max_iterations: int = 50,
    ) -> OptimizationResult:
        history: List[Mapping[str, float]] = []
        best_score = float("inf")
        best_parameters: Dict[str, Mapping[str, float]] = {}
        best_objectives: Dict[str, float] = {}

        for _ in range(max_iterations):
            candidate = self._draw_candidate(samplers)
            objective_values = dict(self.evaluation(candidate))
            composite = self._composite_score(objective_values)
            history.append({"composite": composite, **objective_values})
            if composite < best_score:
                best_score = composite
                best_parameters = candidate
                best_objectives = objective_values

        return OptimizationResult(best_parameters=best_parameters, objective_scores=best_objectives, history=history)

    def _draw_candidate(
        self, samplers: Mapping[str, ParameterSampler]
    ) -> Dict[str, Mapping[str, float]]:
        candidate: Dict[str, Mapping[str, float]] = {}
        for zone in self.zones:
            sampler = samplers.get(zone.id)
            if sampler is None:
                candidate[zone.id] = dict(zone.parameters)
                continue
            try:
                sample = next(iter(sampler(zone)))
            except StopIteration:
                sample = dict(zone.parameters)
            candidate[zone.id] = dict(sample)
        return candidate

    def _composite_score(self, objective_values: Mapping[str, float]) -> float:
        score = 0.0
        for objective in self.objectives:
            value = objective_values.get(objective.id, 0.0)
            if objective.sense == "max":
                contribution = -objective.weight * value
            elif objective.sense == "minabs":
                contribution = objective.weight * abs(value)
            else:  # default to minimise
                contribution = objective.weight * value
            score += contribution
        return score


class UncertaintyAnalyzer:
    """Quantify parameter uncertainty via Monte Carlo style sampling."""

    def __init__(self, zones: Sequence[ParameterZone], evaluation: EvaluationCallback) -> None:
        self.zones = list(zones)
        self.evaluation = evaluation

    def analyse(
        self,
        samplers: Mapping[str, ParameterSampler],
        draws: int = 100,
    ) -> Mapping[str, Mapping[str, float]]:
        aggregates: Dict[str, List[float]] = {}
        for _ in range(draws):
            candidate = self._draw_candidate(samplers)
            metrics = self.evaluation(candidate)
            for metric, value in metrics.items():
                aggregates.setdefault(metric, []).append(value)

        summary: Dict[str, Dict[str, float]] = {}
        for metric, values in aggregates.items():
            if not values:
                continue
            array = list(values)
            mean = sum(array) / len(array)
            variance = sum((x - mean) ** 2 for x in array) / max(len(array) - 1, 1)
            summary[metric] = {
                "mean": mean,
                "std": variance ** 0.5,
                "min": min(array),
                "max": max(array),
            }
        return summary

    def _draw_candidate(
        self, samplers: Mapping[str, ParameterSampler]
    ) -> Dict[str, Mapping[str, float]]:
        candidate: Dict[str, Mapping[str, float]] = {}
        for zone in self.zones:
            sampler = samplers.get(zone.id)
            if sampler is None:
                candidate[zone.id] = dict(zone.parameters)
                continue
            try:
                sample = next(iter(sampler(zone)))
            except StopIteration:
                sample = dict(zone.parameters)
            candidate[zone.id] = dict(sample)
        return candidate


__all__ = [
    "ObjectiveDefinition",
    "OptimizationResult",
    "ParameterZoneOptimizer",
    "UncertaintyAnalyzer",
]
