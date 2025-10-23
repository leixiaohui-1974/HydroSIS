"""Model performance evaluation metrics for hydrological simulations.

This module provides statistical metrics to evaluate model performance by comparing
simulated and observed time series data. All metrics follow hydrology conventions.

Typical usage:
    >>> import numpy as np
    >>> from hydrosis.analysis.metrics import calculate_metrics
    >>>
    >>> observed = np.array([10.5, 12.3, 15.8, 14.2, 11.0])
    >>> simulated = np.array([10.0, 12.0, 16.5, 13.5, 11.2])
    >>>
    >>> metrics = calculate_metrics(observed, simulated)
    >>> print(f"NSE: {metrics['nse']:.3f}")
    >>> print(f"RMSE: {metrics['rmse']:.3f} m³/s")
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union
import numpy as np
import warnings

logger = logging.getLogger(__name__)


def _validate_inputs(observed: np.ndarray, simulated: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate and prepare input arrays.

    Args:
        observed: Observed values
        simulated: Simulated values

    Returns:
        Tuple of validated (observed, simulated) arrays

    Raises:
        ValueError: If inputs are invalid
    """
    obs = np.asarray(observed, dtype=float)
    sim = np.asarray(simulated, dtype=float)

    if obs.shape != sim.shape:
        raise ValueError(f"Shape mismatch: observed {obs.shape} vs simulated {sim.shape}")

    if len(obs) == 0:
        raise ValueError("Input arrays are empty")

    # Check for NaN or Inf
    if np.any(~np.isfinite(obs)):
        raise ValueError("Observed array contains NaN or Inf values")
    if np.any(~np.isfinite(sim)):
        raise ValueError("Simulated array contains NaN or Inf values")

    return obs, sim


def nash_sutcliffe_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 - Σ(Qobs - Qsim)² / Σ(Qobs - Q̄obs)²

    Range: (-∞, 1]
    - 1.0: Perfect match
    - 0.0: Model is as good as the mean of observations
    - < 0: Model is worse than using the mean

    Args:
        observed: Observed time series
        simulated: Simulated time series

    Returns:
        NSE value

    Example:
        >>> obs = np.array([10, 20, 30, 40, 50])
        >>> sim = np.array([12, 19, 31, 38, 51])
        >>> nse_value = nash_sutcliffe_efficiency(obs, sim)
        >>> print(f"NSE: {nse_value:.3f}")
    """
    obs, sim = _validate_inputs(observed, simulated)

    numerator = np.sum((obs - sim) ** 2)
    denominator = np.sum((obs - np.mean(obs)) ** 2)

    if denominator == 0:
        logger.warning("Denominator is zero (constant observations). NSE is undefined.")
        return np.nan

    nse_value = 1.0 - (numerator / denominator)

    logger.debug(f"NSE = {nse_value:.4f}")
    return float(nse_value)


def root_mean_square_error(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Root Mean Square Error (RMSE).

    RMSE = √[Σ(Qobs - Qsim)² / n]

    Units: Same as input (e.g., m³/s for discharge)
    Range: [0, ∞)
    - Lower is better
    - 0 means perfect match

    Args:
        observed: Observed time series
        simulated: Simulated time series

    Returns:
        RMSE value in same units as input
    """
    obs, sim = _validate_inputs(observed, simulated)

    mse = np.mean((obs - sim) ** 2)
    rmse_value = np.sqrt(mse)

    logger.debug(f"RMSE = {rmse_value:.4f}")
    return float(rmse_value)


def mean_absolute_error(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE).

    MAE = Σ|Qobs - Qsim| / n

    Units: Same as input
    Range: [0, ∞)
    - Lower is better
    - Less sensitive to outliers than RMSE

    Args:
        observed: Observed time series
        simulated: Simulated time series

    Returns:
        MAE value
    """
    obs, sim = _validate_inputs(observed, simulated)

    mae_value = np.mean(np.abs(obs - sim))

    logger.debug(f"MAE = {mae_value:.4f}")
    return float(mae_value)


def percent_bias(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Percent Bias (PBIAS).

    PBIAS = 100 × Σ(Qobs - Qsim) / Σ(Qobs)

    Units: Percentage
    Range: (-∞, ∞)
    - 0: Perfect match
    - Positive: Model underestimates (simulated < observed)
    - Negative: Model overestimates (simulated > observed)

    Interpretation (absolute value):
    - < 10%: Very good
    - 10-15%: Good
    - 15-25%: Satisfactory
    - > 25%: Unsatisfactory

    Args:
        observed: Observed time series
        simulated: Simulated time series

    Returns:
        PBIAS value in percentage
    """
    obs, sim = _validate_inputs(observed, simulated)

    total_obs = np.sum(obs)

    if total_obs == 0:
        logger.warning("Sum of observations is zero. PBIAS is undefined.")
        return np.nan

    pbias_value = 100.0 * np.sum(obs - sim) / total_obs

    logger.debug(f"PBIAS = {pbias_value:.2f}%")
    return float(pbias_value)


def kling_gupta_efficiency(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Kling-Gupta Efficiency (KGE).

    KGE = 1 - √[(r-1)² + (α-1)² + (β-1)²]

    where:
    - r: Pearson correlation coefficient
    - α: Ratio of standard deviations (σ_sim / σ_obs)
    - β: Ratio of means (μ_sim / μ_obs)

    Range: (-∞, 1]
    - 1.0: Perfect match
    - > 0.5: Generally good performance
    - KGE decomposes model performance into correlation, variability, and bias

    Args:
        observed: Observed time series
        simulated: Simulated time series

    Returns:
        KGE value

    Reference:
        Gupta et al. (2009), Decomposition of the mean squared error and NSE performance criteria
    """
    obs, sim = _validate_inputs(observed, simulated)

    # Pearson correlation
    r = np.corrcoef(obs, sim)[0, 1]

    # Ratio of standard deviations
    alpha = np.std(sim) / np.std(obs) if np.std(obs) > 0 else np.nan

    # Ratio of means
    beta = np.mean(sim) / np.mean(obs) if np.mean(obs) != 0 else np.nan

    if np.isnan(alpha) or np.isnan(beta):
        logger.warning("KGE components contain NaN. Returning NaN.")
        return np.nan

    # Calculate KGE
    kge_value = 1.0 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    logger.debug(f"KGE = {kge_value:.4f} (r={r:.3f}, α={alpha:.3f}, β={beta:.3f})")
    return float(kge_value)


def log_nash_sutcliffe(observed: np.ndarray, simulated: np.ndarray, epsilon: float = 0.01) -> float:
    """Calculate logarithmic Nash-Sutcliffe Efficiency.

    log-NSE = 1 - Σ(ln(Qobs+ε) - ln(Qsim+ε))² / Σ(ln(Qobs+ε) - ln(Q̄obs+ε))²

    This metric emphasizes low flow performance by taking logarithms.
    Useful for evaluating baseflow and recession periods.

    Args:
        observed: Observed time series
        simulated: Simulated time series
        epsilon: Small constant to avoid log(0), default 0.01

    Returns:
        log-NSE value

    Note:
        Negative values in input will be clipped to epsilon
    """
    obs, sim = _validate_inputs(observed, simulated)

    # Ensure all values are positive
    obs_pos = np.maximum(obs, epsilon)
    sim_pos = np.maximum(sim, epsilon)

    # Log transform
    log_obs = np.log(obs_pos)
    log_sim = np.log(sim_pos)

    # NSE on log-transformed values
    numerator = np.sum((log_obs - log_sim) ** 2)
    denominator = np.sum((log_obs - np.mean(log_obs)) ** 2)

    if denominator == 0:
        logger.warning("Denominator is zero. log-NSE is undefined.")
        return np.nan

    log_nse_value = 1.0 - (numerator / denominator)

    logger.debug(f"log-NSE = {log_nse_value:.4f}")
    return float(log_nse_value)


def volume_error(observed: np.ndarray, simulated: np.ndarray) -> float:
    """Calculate Volume Error (VE).

    VE = (V_sim - V_obs) / V_obs × 100%

    where V = Σ(Q)

    Units: Percentage
    Range: (-100, ∞)
    - 0: Perfect volume match
    - Positive: Overestimation
    - Negative: Underestimation

    Args:
        observed: Observed time series
        simulated: Simulated time series

    Returns:
        Volume error in percentage
    """
    obs, sim = _validate_inputs(observed, simulated)

    vol_obs = np.sum(obs)
    vol_sim = np.sum(sim)

    if vol_obs == 0:
        logger.warning("Total observed volume is zero. VE is undefined.")
        return np.nan

    ve_value = 100.0 * (vol_sim - vol_obs) / vol_obs

    logger.debug(f"VE = {ve_value:.2f}%")
    return float(ve_value)


def peak_error(observed: np.ndarray, simulated: np.ndarray) -> Dict[str, float]:
    """Calculate peak flow errors.

    Returns:
        Dictionary with:
        - peak_obs: Observed peak value
        - peak_sim: Simulated peak value
        - peak_error_abs: Absolute difference
        - peak_error_pct: Percentage error
        - peak_time_obs: Index of observed peak
        - peak_time_sim: Index of simulated peak
        - peak_time_error: Time difference in indices
    """
    obs, sim = _validate_inputs(observed, simulated)

    peak_obs = np.max(obs)
    peak_sim = np.max(sim)
    idx_obs = np.argmax(obs)
    idx_sim = np.argmax(sim)

    peak_error_abs = peak_sim - peak_obs
    peak_error_pct = 100.0 * peak_error_abs / peak_obs if peak_obs != 0 else np.nan
    time_error = idx_sim - idx_obs

    return {
        'peak_obs': float(peak_obs),
        'peak_sim': float(peak_sim),
        'peak_error_abs': float(peak_error_abs),
        'peak_error_pct': float(peak_error_pct),
        'peak_time_obs': int(idx_obs),
        'peak_time_sim': int(idx_sim),
        'peak_time_error': int(time_error),
    }


def calculate_metrics(
    observed: np.ndarray,
    simulated: np.ndarray,
    metrics: Optional[List[str]] = None,
    include_peak_errors: bool = True
) -> Dict[str, float]:
    """Calculate multiple performance metrics at once.

    Args:
        observed: Observed time series
        simulated: Simulated time series
        metrics: List of metric names to calculate. If None, calculates all.
                 Options: 'nse', 'rmse', 'mae', 'pbias', 'kge', 'log_nse', 've'
        include_peak_errors: Whether to include peak flow analysis

    Returns:
        Dictionary of metric names and values

    Example:
        >>> obs = np.array([10, 20, 30, 40, 50])
        >>> sim = np.array([12, 19, 31, 38, 51])
        >>> results = calculate_metrics(obs, sim, metrics=['nse', 'rmse', 'pbias'])
        >>> for name, value in results.items():
        ...     print(f"{name}: {value:.3f}")
    """
    obs, sim = _validate_inputs(observed, simulated)

    # Default: calculate all metrics
    if metrics is None:
        metrics = ['nse', 'rmse', 'mae', 'pbias', 'kge', 'log_nse', 've']

    # Map metric names to functions
    metric_functions = {
        'nse': nash_sutcliffe_efficiency,
        'rmse': root_mean_square_error,
        'mae': mean_absolute_error,
        'pbias': percent_bias,
        'kge': kling_gupta_efficiency,
        'log_nse': log_nash_sutcliffe,
        've': volume_error,
    }

    results = {}

    for metric_name in metrics:
        if metric_name not in metric_functions:
            logger.warning(f"Unknown metric: {metric_name}. Skipping.")
            continue

        try:
            value = metric_functions[metric_name](obs, sim)
            results[metric_name] = value
        except Exception as e:
            logger.error(f"Error calculating {metric_name}: {e}")
            results[metric_name] = np.nan

    # Add peak errors if requested
    if include_peak_errors:
        try:
            peak_results = peak_error(obs, sim)
            results.update(peak_results)
        except Exception as e:
            logger.error(f"Error calculating peak errors: {e}")

    return results


def get_metric_interpretation(metric_name: str, value: float) -> str:
    """Get human-readable interpretation of a metric value.

    Args:
        metric_name: Name of the metric ('nse', 'pbias', etc.)
        value: Metric value

    Returns:
        Interpretation string
    """
    if np.isnan(value):
        return "Undefined"

    metric_name = metric_name.lower()

    if metric_name in ['nse', 'log_nse', 'kge']:
        if value >= 0.9:
            return "Excellent"
        elif value >= 0.75:
            return "Very Good"
        elif value >= 0.65:
            return "Good"
        elif value >= 0.5:
            return "Satisfactory"
        else:
            return "Unsatisfactory"

    elif metric_name == 'pbias':
        abs_value = abs(value)
        if abs_value < 10:
            return "Very Good"
        elif abs_value < 15:
            return "Good"
        elif abs_value < 25:
            return "Satisfactory"
        else:
            return "Unsatisfactory"

    elif metric_name in ['rmse', 'mae']:
        return f"{value:.3f} (lower is better)"

    elif metric_name == 've':
        if abs(value) < 5:
            return "Excellent volume match"
        elif abs(value) < 10:
            return "Good volume match"
        elif abs(value) < 20:
            return "Acceptable volume match"
        else:
            return "Poor volume match"

    else:
        return f"{value:.3f}"


# Aliases for convenience
nse = nash_sutcliffe_efficiency
rmse = root_mean_square_error
mae = mean_absolute_error
pbias = percent_bias
kge = kling_gupta_efficiency


__all__ = [
    'nash_sutcliffe_efficiency',
    'root_mean_square_error',
    'mean_absolute_error',
    'percent_bias',
    'kling_gupta_efficiency',
    'log_nash_sutcliffe',
    'volume_error',
    'peak_error',
    'calculate_metrics',
    'get_metric_interpretation',
    'nse',
    'rmse',
    'mae',
    'pbias',
    'kge',
]
