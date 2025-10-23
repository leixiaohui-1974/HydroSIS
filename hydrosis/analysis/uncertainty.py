"""Uncertainty quantification and analysis for hydrological models.

This module provides tools to quantify and analyze parameter and output uncertainty:

1. Monte Carlo sampling: Random sampling from parameter distributions
2. Latin Hypercube Sampling (LHS): More efficient stratified sampling
3. GLUE: Generalized Likelihood Uncertainty Estimation

Typical usage:
    >>> from hydrosis.analysis.uncertainty import monte_carlo_analysis
    >>>
    >>> def model_wrapper(params):
    ...     return run_hbv_model(**params)
    >>>
    >>> results = monte_carlo_analysis(
    ...     model_function=model_wrapper,
    ...     param_distributions={
    ...         'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
    ...         'K0': {'dist': 'normal', 'mean': 0.3, 'std': 0.05}
    ...     },
    ...     n_samples=1000
    ... )
"""
from __future__ import annotations

import logging
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import pyDOE for Latin Hypercube Sampling (optional)
try:
    from pyDOE import lhs
    PYDOE_AVAILABLE = True
except ImportError:
    PYDOE_AVAILABLE = False
    logger.debug("pyDOE not available. LHS will use numpy fallback. "
                 "Install with: pip install pyDOE")


def _sample_parameter(distribution_spec: Dict[str, Any], size: int = 1) -> np.ndarray:
    """Sample from a parameter distribution.

    Args:
        distribution_spec: Dictionary specifying the distribution
                          - For uniform: {'dist': 'uniform', 'min': a, 'max': b}
                          - For normal: {'dist': 'normal', 'mean': μ, 'std': σ}
                          - For lognormal: {'dist': 'lognormal', 'mean': μ, 'std': σ}
        size: Number of samples to generate

    Returns:
        Array of sampled values
    """
    dist_type = distribution_spec.get('dist', 'uniform').lower()

    if dist_type == 'uniform':
        min_val = distribution_spec['min']
        max_val = distribution_spec['max']
        return np.random.uniform(min_val, max_val, size)

    elif dist_type == 'normal':
        mean = distribution_spec['mean']
        std = distribution_spec['std']
        samples = np.random.normal(mean, std, size)

        # Optionally clip to bounds if specified
        if 'min' in distribution_spec:
            samples = np.maximum(samples, distribution_spec['min'])
        if 'max' in distribution_spec:
            samples = np.minimum(samples, distribution_spec['max'])

        return samples

    elif dist_type == 'lognormal':
        mean = distribution_spec['mean']
        std = distribution_spec['std']
        return np.random.lognormal(mean, std, size)

    elif dist_type == 'truncnormal':
        # Truncated normal distribution
        mean = distribution_spec['mean']
        std = distribution_spec['std']
        min_val = distribution_spec['min']
        max_val = distribution_spec['max']

        samples = []
        while len(samples) < size:
            sample = np.random.normal(mean, std)
            if min_val <= sample <= max_val:
                samples.append(sample)

        return np.array(samples)

    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")


def monte_carlo_sampling(
    param_distributions: Dict[str, Dict[str, Any]],
    n_samples: int,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Generate Monte Carlo parameter samples.

    Args:
        param_distributions: Dictionary of parameter distributions
                            {param_name: distribution_spec}
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary of parameter arrays {param_name: array of samples}

    Example:
        >>> samples = monte_carlo_sampling(
        ...     param_distributions={
        ...         'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
        ...         'K0': {'dist': 'normal', 'mean': 0.3, 'std': 0.05, 'min': 0.1, 'max': 0.5}
        ...     },
        ...     n_samples=1000,
        ...     seed=42
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    logger.info(f"Generating {n_samples} Monte Carlo samples for {len(param_distributions)} parameters")

    samples = {}
    for param_name, dist_spec in param_distributions.items():
        samples[param_name] = _sample_parameter(dist_spec, n_samples)
        logger.debug(f"  {param_name}: {dist_spec['dist']} distribution, "
                    f"range=[{samples[param_name].min():.3f}, {samples[param_name].max():.3f}]")

    return samples


def latin_hypercube_sampling(
    param_distributions: Dict[str, Dict[str, Any]],
    n_samples: int,
    seed: Optional[int] = None
) -> Dict[str, np.ndarray]:
    """Generate Latin Hypercube Samples (LHS).

    LHS provides better parameter space coverage than Monte Carlo
    with the same number of samples.

    Args:
        param_distributions: Dictionary of parameter distributions
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility

    Returns:
        Dictionary of parameter arrays

    Example:
        >>> samples = latin_hypercube_sampling(
        ...     param_distributions={
        ...         'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
        ...         'K0': {'dist': 'uniform', 'min': 0.1, 'max': 0.5}
        ...     },
        ...     n_samples=100
        ... )
    """
    if seed is not None:
        np.random.seed(seed)

    param_names = list(param_distributions.keys())
    num_params = len(param_names)

    logger.info(f"Generating {n_samples} Latin Hypercube samples for {num_params} parameters")

    # Generate LHS matrix in [0, 1] space
    if PYDOE_AVAILABLE:
        lhs_matrix = lhs(num_params, samples=n_samples, criterion='maximin')
    else:
        # Fallback: Simple LHS implementation
        logger.warning("Using numpy fallback for LHS (install pyDOE for better results)")
        lhs_matrix = np.zeros((n_samples, num_params))
        for i in range(num_params):
            # Divide [0, 1] into n_samples intervals
            intervals = np.arange(n_samples) / n_samples
            # Randomly sample within each interval
            samples_i = intervals + np.random.uniform(0, 1/n_samples, n_samples)
            # Shuffle
            np.random.shuffle(samples_i)
            lhs_matrix[:, i] = samples_i

    # Transform from [0, 1] to actual parameter distributions
    samples = {}
    for i, param_name in enumerate(param_names):
        dist_spec = param_distributions[param_name]
        dist_type = dist_spec.get('dist', 'uniform').lower()

        uniform_samples = lhs_matrix[:, i]

        if dist_type == 'uniform':
            min_val = dist_spec['min']
            max_val = dist_spec['max']
            samples[param_name] = min_val + uniform_samples * (max_val - min_val)

        elif dist_type == 'normal':
            # Use inverse CDF (percent point function)
            from scipy.stats import norm
            mean = dist_spec['mean']
            std = dist_spec['std']
            samples[param_name] = norm.ppf(uniform_samples, loc=mean, scale=std)

            # Clip if bounds specified
            if 'min' in dist_spec:
                samples[param_name] = np.maximum(samples[param_name], dist_spec['min'])
            if 'max' in dist_spec:
                samples[param_name] = np.minimum(samples[param_name], dist_spec['max'])

        elif dist_type == 'lognormal':
            from scipy.stats import lognorm
            mean = dist_spec['mean']
            std = dist_spec['std']
            samples[param_name] = lognorm.ppf(uniform_samples, s=std, scale=np.exp(mean))

        else:
            logger.warning(f"LHS transform not implemented for {dist_type}, using uniform")
            min_val = dist_spec.get('min', 0)
            max_val = dist_spec.get('max', 1)
            samples[param_name] = min_val + uniform_samples * (max_val - min_val)

        logger.debug(f"  {param_name}: range=[{samples[param_name].min():.3f}, "
                    f"{samples[param_name].max():.3f}]")

    return samples


def monte_carlo_analysis(
    model_function: Callable,
    param_distributions: Dict[str, Dict[str, Any]],
    n_samples: int,
    sampling_method: str = 'monte_carlo',
    observed_data: Optional[np.ndarray] = None,
    metric_function: Optional[Callable] = None,
    seed: Optional[int] = None,
    n_jobs: int = 1
) -> Dict[str, Any]:
    """Perform Monte Carlo uncertainty analysis.

    Args:
        model_function: Function that takes parameters and returns model output
                       Signature: func(**params) -> np.ndarray or float
        param_distributions: Parameter distributions
        n_samples: Number of Monte Carlo samples
        sampling_method: 'monte_carlo' or 'lhs' (Latin Hypercube)
        observed_data: Optional observed data for calculating metrics
        metric_function: Optional function to calculate metric from (simulated, observed)
                        Signature: func(sim, obs) -> float
        seed: Random seed
        n_jobs: Number of parallel jobs (reserved for future)

    Returns:
        Dictionary with:
        - param_samples: Dictionary of parameter samples
        - outputs: Array of model outputs
        - metrics: Array of metric values (if metric_function provided)
        - output_mean: Mean of outputs
        - output_std: Standard deviation of outputs
        - output_percentiles: Percentiles [5, 25, 50, 75, 95]
        - param_statistics: Statistics for each parameter

    Example:
        >>> results = monte_carlo_analysis(
        ...     model_function=run_hbv,
        ...     param_distributions={
        ...         'FC': {'dist': 'uniform', 'min': 100, 'max': 200},
        ...         'K0': {'dist': 'uniform', 'min': 0.1, 'max': 0.5}
        ...     },
        ...     n_samples=1000
        ... )
    """
    logger.info(f"Starting Monte Carlo uncertainty analysis with {n_samples} samples")
    logger.info(f"Sampling method: {sampling_method}")

    # Generate parameter samples
    if sampling_method.lower() == 'lhs':
        param_samples = latin_hypercube_sampling(param_distributions, n_samples, seed)
    else:
        param_samples = monte_carlo_sampling(param_distributions, n_samples, seed)

    # Run model for all samples
    logger.info("Running model evaluations...")
    outputs = []
    metrics = [] if metric_function is not None and observed_data is not None else None

    param_names = list(param_samples.keys())
    n_params = len(param_names)

    for i in range(n_samples):
        # Extract parameter set
        params = {name: param_samples[name][i] for name in param_names}

        try:
            # Run model
            output = model_function(**params)
            outputs.append(output)

            # Calculate metric if requested
            if metrics is not None:
                metric_val = metric_function(output, observed_data)
                metrics.append(metric_val)

            if (i + 1) % 100 == 0:
                logger.debug(f"  Completed {i+1}/{n_samples} runs")

        except Exception as e:
            logger.error(f"Error in run {i+1}: {e}")
            outputs.append(np.nan)
            if metrics is not None:
                metrics.append(np.nan)

    # Convert to arrays
    outputs = np.array(outputs)
    if metrics is not None:
        metrics = np.array(metrics)

    logger.info(f"Completed {n_samples} model runs")

    # Calculate output statistics
    if outputs.ndim == 1:
        # Scalar outputs
        output_mean = np.nanmean(outputs)
        output_std = np.nanstd(outputs)
        output_percentiles = np.nanpercentile(outputs, [5, 25, 50, 75, 95])

        logger.info(f"Output statistics:")
        logger.info(f"  Mean: {output_mean:.4f}")
        logger.info(f"  Std:  {output_std:.4f}")
        logger.info(f"  5%-95% range: [{output_percentiles[0]:.4f}, {output_percentiles[4]:.4f}]")
    else:
        # Time series outputs
        output_mean = np.nanmean(outputs, axis=0)
        output_std = np.nanstd(outputs, axis=0)
        output_percentiles = np.nanpercentile(outputs, [5, 25, 50, 75, 95], axis=0)

        logger.info(f"Output statistics (time series):")
        logger.info(f"  Mean range: [{output_mean.min():.4f}, {output_mean.max():.4f}]")
        logger.info(f"  Std range:  [{output_std.min():.4f}, {output_std.max():.4f}]")

    # Calculate parameter statistics
    param_statistics = {}
    for param_name in param_names:
        param_statistics[param_name] = {
            'mean': np.mean(param_samples[param_name]),
            'std': np.std(param_samples[param_name]),
            'min': np.min(param_samples[param_name]),
            'max': np.max(param_samples[param_name]),
            'percentiles': np.percentile(param_samples[param_name], [5, 25, 50, 75, 95]),
        }

    results = {
        'param_samples': param_samples,
        'outputs': outputs,
        'output_mean': output_mean,
        'output_std': output_std,
        'output_percentiles': output_percentiles,
        'param_statistics': param_statistics,
        'n_samples': n_samples,
        'sampling_method': sampling_method,
    }

    if metrics is not None:
        results['metrics'] = metrics
        results['metric_mean'] = np.nanmean(metrics)
        results['metric_std'] = np.nanstd(metrics)
        results['metric_percentiles'] = np.nanpercentile(metrics, [5, 25, 50, 75, 95])

        logger.info(f"Metric statistics:")
        logger.info(f"  Mean: {results['metric_mean']:.4f}")
        logger.info(f"  5%-95% range: [{results['metric_percentiles'][0]:.4f}, "
                   f"{results['metric_percentiles'][4]:.4f}]")

    return results


def glue_analysis(
    model_function: Callable,
    param_distributions: Dict[str, Dict[str, Any]],
    observed_data: np.ndarray,
    likelihood_function: Callable,
    likelihood_threshold: float,
    n_samples: int = 10000,
    sampling_method: str = 'lhs',
    confidence_levels: List[float] = None,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """Perform GLUE (Generalized Likelihood Uncertainty Estimation) analysis.

    GLUE identifies "behavioral" parameter sets (those that exceed a likelihood
    threshold) and uses them to estimate output uncertainty bounds.

    Args:
        model_function: Function that takes parameters and returns time series
        param_distributions: Parameter distributions
        observed_data: Observed time series for comparison
        likelihood_function: Function to calculate likelihood from (simulated, observed)
                            Signature: func(sim, obs) -> float
                            Higher values = better fit
        likelihood_threshold: Minimum likelihood for "behavioral" parameters
        n_samples: Total number of samples to generate
        sampling_method: 'monte_carlo' or 'lhs'
        confidence_levels: Percentiles for uncertainty bounds (e.g., [0.05, 0.95])
        seed: Random seed

    Returns:
        Dictionary with:
        - behavioral_params: Parameter sets that exceed threshold
        - behavioral_outputs: Corresponding model outputs
        - behavioral_likelihoods: Likelihood values
        - acceptance_rate: Fraction of behavioral samples
        - uncertainty_bounds: Uncertainty envelopes at specified percentiles
        - posterior_param_stats: Posterior parameter statistics

    Example:
        >>> from hydrosis.analysis.metrics import nash_sutcliffe_efficiency
        >>>
        >>> results = glue_analysis(
        ...     model_function=run_hbv,
        ...     param_distributions={'FC': {'dist': 'uniform', 'min': 100, 'max': 200}},
        ...     observed_data=observed_flow,
        ...     likelihood_function=nash_sutcliffe_efficiency,
        ...     likelihood_threshold=0.5,
        ...     n_samples=10000
        ... )
    """
    if confidence_levels is None:
        confidence_levels = [0.05, 0.95]

    logger.info(f"Starting GLUE analysis with {n_samples} samples")
    logger.info(f"Likelihood threshold: {likelihood_threshold}")

    # Generate parameter samples
    if sampling_method.lower() == 'lhs':
        param_samples = latin_hypercube_sampling(param_distributions, n_samples, seed)
    else:
        param_samples = monte_carlo_sampling(param_distributions, n_samples, seed)

    param_names = list(param_samples.keys())

    # Run model and calculate likelihoods
    logger.info("Evaluating model and calculating likelihoods...")
    outputs = []
    likelihoods = []

    for i in range(n_samples):
        params = {name: param_samples[name][i] for name in param_names}

        try:
            output = model_function(**params)
            likelihood = likelihood_function(output, observed_data)

            outputs.append(output)
            likelihoods.append(likelihood)

            if (i + 1) % 100 == 0:
                logger.debug(f"  Completed {i+1}/{n_samples} runs")

        except Exception as e:
            logger.error(f"Error in run {i+1}: {e}")
            outputs.append(None)
            likelihoods.append(np.nan)

    likelihoods = np.array(likelihoods)

    # Identify behavioral parameter sets
    behavioral_mask = likelihoods >= likelihood_threshold
    n_behavioral = np.sum(behavioral_mask)
    acceptance_rate = n_behavioral / n_samples

    logger.info(f"Behavioral samples: {n_behavioral}/{n_samples} ({acceptance_rate*100:.1f}%)")

    if n_behavioral == 0:
        logger.error("No behavioral parameter sets found! Lower the threshold or increase samples.")
        return {
            'behavioral_params': None,
            'behavioral_outputs': None,
            'acceptance_rate': 0.0,
            'error': 'No behavioral samples'
        }

    # Extract behavioral sets
    behavioral_params = {name: param_samples[name][behavioral_mask] for name in param_names}
    behavioral_outputs = np.array([outputs[i] for i in range(n_samples) if behavioral_mask[i]])
    behavioral_likelihoods = likelihoods[behavioral_mask]

    # Calculate posterior parameter statistics
    posterior_param_stats = {}
    for param_name in param_names:
        behavioral_values = behavioral_params[param_name]
        posterior_param_stats[param_name] = {
            'mean': np.mean(behavioral_values),
            'std': np.std(behavioral_values),
            'min': np.min(behavioral_values),
            'max': np.max(behavioral_values),
            'median': np.median(behavioral_values),
            'percentiles': np.percentile(behavioral_values, [5, 25, 50, 75, 95]),
        }

        logger.info(f"{param_name} posterior: mean={posterior_param_stats[param_name]['mean']:.3f}, "
                   f"std={posterior_param_stats[param_name]['std']:.3f}")

    # Calculate uncertainty bounds
    percentile_values = [conf * 100 for conf in confidence_levels]
    uncertainty_bounds = {}

    if behavioral_outputs.ndim == 2:  # Time series
        for i, pct in enumerate(percentile_values):
            bound = np.percentile(behavioral_outputs, pct, axis=0)
            uncertainty_bounds[f'p{int(pct)}'] = bound
            logger.debug(f"  {pct}% percentile: range=[{bound.min():.2f}, {bound.max():.2f}]")

        # Also calculate weighted percentiles using likelihoods as weights
        # (Optional: more sophisticated GLUE uses likelihood weighting)
        weights = behavioral_likelihoods / behavioral_likelihoods.sum()
        weighted_mean = np.average(behavioral_outputs, axis=0, weights=weights)
        uncertainty_bounds['weighted_mean'] = weighted_mean

    else:  # Scalar outputs
        for i, pct in enumerate(percentile_values):
            uncertainty_bounds[f'p{int(pct)}'] = np.percentile(behavioral_outputs, pct)

    return {
        'behavioral_params': behavioral_params,
        'behavioral_outputs': behavioral_outputs,
        'behavioral_likelihoods': behavioral_likelihoods,
        'acceptance_rate': acceptance_rate,
        'n_behavioral': n_behavioral,
        'n_total': n_samples,
        'likelihood_threshold': likelihood_threshold,
        'posterior_param_stats': posterior_param_stats,
        'uncertainty_bounds': uncertainty_bounds,
        'confidence_levels': confidence_levels,
    }


__all__ = [
    'monte_carlo_sampling',
    'latin_hypercube_sampling',
    'monte_carlo_analysis',
    'glue_analysis',
]
