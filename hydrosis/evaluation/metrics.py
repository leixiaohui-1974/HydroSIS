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


def log_nash_sutcliffe_efficiency(
    simulated: Sequence[float], observed: Sequence[float], epsilon: float = 1e-6
) -> float:
    """
    Log-transformed Nash-Sutcliffe efficiency for emphasizing low flows.

    Parameters
    ----------
    simulated : Sequence[float]
        Simulated discharge series
    observed : Sequence[float]
        Observed discharge series
    epsilon : float, optional
        Small value added before log transformation to avoid log(0)

    Returns
    -------
    float
        Log NSE value (range: -inf to 1.0, perfect=1.0)
    """
    _validate_lengths(simulated, observed)
    if not simulated:
        return 1.0

    log_sim = [math.log(s + epsilon) for s in simulated]
    log_obs = [math.log(o + epsilon) for o in observed]

    return nash_sutcliffe_efficiency(log_sim, log_obs)


def kling_gupta_efficiency(
    simulated: Sequence[float], observed: Sequence[float]
) -> float:
    """
    Kling-Gupta Efficiency (KGE) for comprehensive model evaluation.

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where:
    - r: correlation coefficient
    - alpha: ratio of standard deviations (sim/obs)
    - beta: ratio of means (sim/obs)

    Parameters
    ----------
    simulated : Sequence[float]
        Simulated discharge series
    observed : Sequence[float]
        Observed discharge series

    Returns
    -------
    float
        KGE value (range: -inf to 1.0, perfect=1.0)
    """
    _validate_lengths(simulated, observed)
    if not simulated:
        return 1.0

    n = len(simulated)
    if n == 0:
        return 1.0

    # Calculate means
    mean_sim = sum(simulated) / n
    mean_obs = sum(observed) / n

    # Calculate standard deviations
    var_sim = sum((s - mean_sim) ** 2 for s in simulated) / n
    var_obs = sum((o - mean_obs) ** 2 for o in observed) / n
    std_sim = math.sqrt(var_sim) if var_sim > 0 else 0.0
    std_obs = math.sqrt(var_obs) if var_obs > 0 else 0.0

    # Calculate correlation coefficient
    if std_sim == 0 or std_obs == 0:
        r = 1.0 if std_sim == std_obs else 0.0
    else:
        covariance = sum((s - mean_sim) * (o - mean_obs) for s, o in zip(simulated, observed)) / n
        r = covariance / (std_sim * std_obs)

    # Calculate alpha and beta
    alpha = std_sim / std_obs if std_obs > 0 else 1.0
    beta = mean_sim / mean_obs if mean_obs > 0 else 1.0

    # Calculate KGE
    kge = 1.0 - math.sqrt((r - 1.0)**2 + (alpha - 1.0)**2 + (beta - 1.0)**2)

    return kge


def pearson_correlation(
    simulated: Sequence[float], observed: Sequence[float]
) -> float:
    """
    Pearson correlation coefficient between simulated and observed series.

    Parameters
    ----------
    simulated : Sequence[float]
        Simulated discharge series
    observed : Sequence[float]
        Observed discharge series

    Returns
    -------
    float
        Correlation coefficient (range: -1 to 1, perfect=1)
    """
    _validate_lengths(simulated, observed)
    if not simulated:
        return 1.0

    n = len(simulated)
    mean_sim = sum(simulated) / n
    mean_obs = sum(observed) / n

    var_sim = sum((s - mean_sim) ** 2 for s in simulated)
    var_obs = sum((o - mean_obs) ** 2 for o in observed)

    if var_sim == 0 or var_obs == 0:
        return 1.0 if var_sim == var_obs else 0.0

    covariance = sum((s - mean_sim) * (o - mean_obs) for s, o in zip(simulated, observed))

    return covariance / math.sqrt(var_sim * var_obs)


DEFAULT_METRICS = {
    "rmse": rmse,
    "mae": mae,
    "pbias": percent_bias,
    "nse": nash_sutcliffe_efficiency,
    "log_nse": log_nash_sutcliffe_efficiency,
    "kge": kling_gupta_efficiency,
    "correlation": pearson_correlation,
}


DEFAULT_ORIENTATION = {
    "rmse": "min",
    "mae": "min",
    "pbias": "minabs",
    "nse": "max",
    "log_nse": "max",
    "kge": "max",
    "correlation": "max",
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
    "log_nash_sutcliffe_efficiency",
    "kling_gupta_efficiency",
    "pearson_correlation",
    "percent_bias",
    "rmse",
]
