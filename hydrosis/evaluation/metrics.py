"""Core performance metrics for rainfall-runoff accuracy assessment."""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def _validate_lengths(simulated: Sequence[float], observed: Sequence[float]) -> None:
    if len(simulated) != len(observed):
        raise ValueError(
            "Simulated and observed series must have the same length for evaluation"
        )


def rmse(simulated: Sequence[float], observed: Sequence[float]) -> float:
    """Compute the root-mean-square error between two series."""

    _validate_lengths(simulated, observed)
    if not simulated:
        return 0.0
    squared = [(s - o) ** 2 for s, o in zip(simulated, observed)]
    return math.sqrt(sum(squared) / len(squared))


def mae(simulated: Sequence[float], observed: Sequence[float]) -> float:
    """Compute the mean absolute error between two series."""

    _validate_lengths(simulated, observed)
    if not simulated:
        return 0.0
    return sum(abs(s - o) for s, o in zip(simulated, observed)) / len(simulated)


def percent_bias(simulated: Sequence[float], observed: Sequence[float]) -> float:
    """Percent bias indicating the mean tendency of simulated flows."""

    _validate_lengths(simulated, observed)
    obs_sum = sum(observed)
    sim_sum = sum(simulated)
    if math.isclose(obs_sum, 0.0, abs_tol=1e-12):
        if math.isclose(sim_sum, 0.0, abs_tol=1e-12):
            return 0.0
        return float("inf") if sim_sum > 0 else float("-inf")
    diff_sum = sim_sum - obs_sum
    return 100.0 * diff_sum / obs_sum


def nash_sutcliffe_efficiency(
    simulated: Sequence[float], observed: Sequence[float]
) -> float:
    """Nash-Sutcliffe efficiency (NSE) for hydrograph accuracy."""

    _validate_lengths(simulated, observed)
    if not simulated:
        return 1.0
    mean_obs = sum(observed) / len(observed)
    numerator = sum((o - s) ** 2 for s, o in zip(simulated, observed))
    denominator = sum((o - mean_obs) ** 2 for o in observed)
    if math.isclose(denominator, 0.0, abs_tol=1e-12):
        return 1.0
    return 1.0 - numerator / denominator


DEFAULT_METRICS = {
    "rmse": rmse,
    "mae": mae,
    "pbias": percent_bias,
    "nse": nash_sutcliffe_efficiency,
}


DEFAULT_ORIENTATION = {
    "rmse": "min",
    "mae": "min",
    "pbias": "minabs",
    "nse": "max",
}


def available_metrics() -> Iterable[str]:
    """Return the identifiers for the built-in evaluation metrics."""

    return DEFAULT_METRICS.keys()


__all__ = [
    "available_metrics",
    "DEFAULT_METRICS",
    "DEFAULT_ORIENTATION",
    "mae",
    "nash_sutcliffe_efficiency",
    "percent_bias",
    "rmse",
]
