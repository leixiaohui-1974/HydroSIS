"""Parameter sensitivity analysis for hydrological models.

This module provides tools to analyze how model outputs respond to changes in
input parameters. Three main methods are implemented:

1. One-at-a-Time (OAT): Simple local sensitivity analysis
2. Sobol indices: Global variance-based sensitivity analysis
3. Morris screening: Efficient parameter screening method

Typical usage:
    >>> from hydrosis.analysis.sensitivity import one_at_a_time_sensitivity
    >>>
    >>> def model_wrapper(params):
    ...     return run_hbv_model(**params)
    >>>
    >>> results = one_at_a_time_sensitivity(
    ...     model_function=model_wrapper,
    ...     parameters={'FC': 150, 'K0': 0.3, 'BETA': 1.0},
    ...     param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5], 'BETA': [0.5, 2.0]},
    ...     variations=[0.8, 0.9, 1.0, 1.1, 1.2]
    ... )
"""
from __future__ import annotations

import logging
from typing import Dict, List, Callable, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

# Try to import SALib (optional dependency)
try:
    from SALib.sample import saltelli, morris as morris_sample
    from SALib.analyze import sobol, morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logger.warning("SALib not available. Sobol and Morris methods will not work. "
                   "Install with: pip install SALib")


def one_at_a_time_sensitivity(
    model_function: Callable,
    parameters: Dict[str, float],
    param_ranges: Dict[str, List[float]],
    variations: List[float] = None,
    metric_name: str = 'output',
    n_jobs: int = 1
) -> Dict[str, Any]:
    """Perform One-at-a-Time (OAT) sensitivity analysis.

    This method varies one parameter at a time while keeping others constant,
    to understand the individual effect of each parameter.

    Args:
        model_function: Function that takes parameters dict and returns a metric value
                       Signature: func(**params) -> float
        parameters: Baseline parameter values
        param_ranges: Valid ranges for each parameter {param: [min, max]}
        variations: Multipliers to apply to baseline values (e.g., [0.8, 0.9, 1.0, 1.1, 1.2])
                   If None, uses [0.5, 0.75, 1.0, 1.25, 1.5]
        metric_name: Name of the output metric being analyzed
        n_jobs: Number of parallel jobs (currently not implemented, reserved for future)

    Returns:
        Dictionary with:
        - param_effects: Dict[str, Dict] with results for each parameter
        - baseline_output: Output with baseline parameters
        - sensitivity_indices: Normalized sensitivity index for each parameter

    Example:
        >>> def my_model(FC, K0):
        ...     return FC * K0  # Simple example
        >>>
        >>> results = one_at_a_time_sensitivity(
        ...     model_function=my_model,
        ...     parameters={'FC': 150, 'K0': 0.3},
        ...     param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5]},
        ...     variations=[0.8, 1.0, 1.2]
        ... )
    """
    if variations is None:
        variations = [0.5, 0.75, 1.0, 1.25, 1.5]

    logger.info(f"Starting OAT sensitivity analysis for {len(parameters)} parameters")
    logger.info(f"Variations: {variations}")

    # Get baseline output
    logger.debug(f"Running baseline with parameters: {parameters}")
    baseline_output = model_function(**parameters)
    logger.info(f"Baseline {metric_name}: {baseline_output:.4f}")

    param_effects = {}
    sensitivity_indices = {}

    for param_name, baseline_value in parameters.items():
        logger.info(f"Analyzing parameter: {param_name} (baseline={baseline_value:.3f})")

        param_min, param_max = param_ranges[param_name]

        outputs = []
        param_values = []

        for multiplier in variations:
            # Calculate new parameter value
            new_value = baseline_value * multiplier

            # Clip to valid range
            new_value = np.clip(new_value, param_min, param_max)

            # Create modified parameters
            modified_params = parameters.copy()
            modified_params[param_name] = new_value

            # Run model
            try:
                output = model_function(**modified_params)
                outputs.append(output)
                param_values.append(new_value)

                change_pct = (multiplier - 1.0) * 100
                output_change_pct = (output - baseline_output) / baseline_output * 100 if baseline_output != 0 else 0
                logger.debug(f"  {param_name}={new_value:.3f} ({change_pct:+.0f}%) -> "
                           f"{metric_name}={output:.4f} ({output_change_pct:+.2f}%)")
            except Exception as e:
                logger.error(f"  Error running model with {param_name}={new_value:.3f}: {e}")
                outputs.append(np.nan)
                param_values.append(new_value)

        # Calculate sensitivity index
        # SI = (ΔOutput / Output_baseline) / (ΔParam / Param_baseline)
        if baseline_output != 0 and baseline_value != 0:
            output_range = np.nanmax(outputs) - np.nanmin(outputs)
            param_range = max(param_values) - min(param_values)

            relative_output_change = output_range / abs(baseline_output)
            relative_param_change = param_range / abs(baseline_value)

            si = relative_output_change / relative_param_change if relative_param_change > 0 else 0
        else:
            si = 0

        sensitivity_indices[param_name] = si

        param_effects[param_name] = {
            'baseline_value': baseline_value,
            'tested_values': param_values,
            'outputs': outputs,
            'sensitivity_index': si,
            'output_range': np.nanmax(outputs) - np.nanmin(outputs),
            'output_min': np.nanmin(outputs),
            'output_max': np.nanmax(outputs),
        }

        logger.info(f"  {param_name} sensitivity index: {si:.4f}")

    # Rank parameters by sensitivity
    ranked_params = sorted(sensitivity_indices.items(), key=lambda x: abs(x[1]), reverse=True)

    logger.info("Sensitivity ranking (most to least sensitive):")
    for i, (param, si) in enumerate(ranked_params, 1):
        logger.info(f"  {i}. {param}: SI={si:.4f}")

    return {
        'baseline_parameters': parameters,
        'baseline_output': baseline_output,
        'param_effects': param_effects,
        'sensitivity_indices': sensitivity_indices,
        'ranked_parameters': ranked_params,
        'metric_name': metric_name,
    }


def sobol_sensitivity(
    model_function: Callable,
    param_ranges: Dict[str, List[float]],
    n_samples: int = 1000,
    calc_second_order: bool = True,
    metric_name: str = 'output'
) -> Dict[str, Any]:
    """Perform global Sobol variance-based sensitivity analysis.

    Sobol indices decompose the output variance into contributions from
    individual parameters (first-order) and their interactions (second-order, total-order).

    Args:
        model_function: Function that takes parameters dict and returns a metric value
        param_ranges: Parameter ranges {param: [min, max]}
        n_samples: Base sample size (actual samples = n * (2d + 2) where d=num_params)
        calc_second_order: Whether to calculate second-order interactions
        metric_name: Name of the output metric

    Returns:
        Dictionary with:
        - S1: First-order Sobol indices (direct effect)
        - ST: Total-order Sobol indices (total effect including interactions)
        - S2: Second-order indices (pairwise interactions), if calc_second_order=True
        - param_names: List of parameter names
        - problem: SALib problem definition

    Requires:
        SALib package (pip install SALib)

    Example:
        >>> results = sobol_sensitivity(
        ...     model_function=my_model,
        ...     param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5]},
        ...     n_samples=1000
        ... )
        >>> print(f"First-order indices: {results['S1']}")
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required for Sobol analysis. Install with: pip install SALib")

    param_names = list(param_ranges.keys())
    num_params = len(param_names)

    logger.info(f"Starting Sobol sensitivity analysis for {num_params} parameters")
    logger.info(f"Base samples: {n_samples}, Total model runs: {n_samples * (2 * num_params + 2)}")

    # Define SALib problem
    problem = {
        'num_vars': num_params,
        'names': param_names,
        'bounds': [param_ranges[name] for name in param_names]
    }

    # Generate samples using Saltelli's scheme
    logger.info("Generating Saltelli samples...")
    param_values = saltelli.sample(problem, n_samples, calc_second_order=calc_second_order)

    total_samples = len(param_values)
    logger.info(f"Generated {total_samples} parameter sets")

    # Run model for all samples
    logger.info("Running model evaluations...")
    Y = np.zeros(total_samples)

    for i, params in enumerate(param_values):
        param_dict = {name: value for name, value in zip(param_names, params)}
        try:
            Y[i] = model_function(**param_dict)
            if (i + 1) % 100 == 0:
                logger.debug(f"  Completed {i+1}/{total_samples} runs")
        except Exception as e:
            logger.error(f"Error in run {i+1}: {e}")
            Y[i] = np.nan

    if np.any(np.isnan(Y)):
        logger.warning(f"{np.sum(np.isnan(Y))} model runs failed. Results may be unreliable.")

    # Analyze results
    logger.info("Analyzing Sobol indices...")
    Si = sobol.analyze(problem, Y, calc_second_order=calc_second_order, print_to_console=False)

    logger.info("Sobol Analysis Results:")
    logger.info("First-order indices (S1):")
    for name, s1 in zip(param_names, Si['S1']):
        logger.info(f"  {name:15s}: {s1:7.4f}")

    logger.info("Total-order indices (ST):")
    for name, st in zip(param_names, Si['ST']):
        logger.info(f"  {name:15s}: {st:7.4f}")

    results = {
        'S1': Si['S1'],  # First-order indices
        'S1_conf': Si['S1_conf'],  # Confidence intervals
        'ST': Si['ST'],  # Total-order indices
        'ST_conf': Si['ST_conf'],
        'param_names': param_names,
        'problem': problem,
        'metric_name': metric_name,
        'n_samples': total_samples,
    }

    if calc_second_order:
        results['S2'] = Si['S2']
        results['S2_conf'] = Si['S2_conf']

    return results


def morris_screening(
    model_function: Callable,
    param_ranges: Dict[str, List[float]],
    n_trajectories: int = 10,
    n_levels: int = 4,
    metric_name: str = 'output'
) -> Dict[str, Any]:
    """Perform Morris screening method for parameter importance.

    The Morris method efficiently identifies the most important parameters by
    sampling trajectories through the parameter space. It provides:
    - μ* (mu_star): Mean of absolute elementary effects (importance)
    - σ (sigma): Standard deviation of effects (interactions/non-linearity)

    Args:
        model_function: Function that takes parameters dict and returns a metric value
        param_ranges: Parameter ranges {param: [min, max]}
        n_trajectories: Number of trajectories (higher = more accurate, recommended 10-50)
        n_levels: Number of grid levels (typically 4 or 6)
        metric_name: Name of the output metric

    Returns:
        Dictionary with:
        - mu: Mean of elementary effects
        - mu_star: Mean of absolute elementary effects (parameter importance)
        - sigma: Standard deviation (interaction/non-linearity indicator)
        - param_names: List of parameter names
        - problem: SALib problem definition

    Requires:
        SALib package (pip install SALib)

    Interpretation:
        - High μ*: Important parameter
        - High σ: Parameter has interactions or non-linear effects
        - Low μ*, low σ: Unimportant parameter

    Example:
        >>> results = morris_screening(
        ...     model_function=my_model,
        ...     param_ranges={'FC': [100, 200], 'K0': [0.1, 0.5]},
        ...     n_trajectories=10
        ... )
        >>> for name, mu_star in zip(results['param_names'], results['mu_star']):
        ...     print(f"{name}: μ*={mu_star:.4f}")
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib is required for Morris screening. Install with: pip install SALib")

    param_names = list(param_ranges.keys())
    num_params = len(param_names)

    logger.info(f"Starting Morris screening for {num_params} parameters")
    logger.info(f"Trajectories: {n_trajectories}, Levels: {n_levels}")

    # Define SALib problem
    problem = {
        'num_vars': num_params,
        'names': param_names,
        'bounds': [param_ranges[name] for name in param_names]
    }

    # Generate Morris samples
    logger.info("Generating Morris samples...")
    param_values = morris_sample.sample(
        problem,
        N=n_trajectories,
        num_levels=n_levels,
        optimal_trajectories=None  # Can use 'optimal_trajectories' for better sampling
    )

    total_samples = len(param_values)
    logger.info(f"Generated {total_samples} parameter sets")

    # Run model for all samples
    logger.info("Running model evaluations...")
    Y = np.zeros(total_samples)

    for i, params in enumerate(param_values):
        param_dict = {name: value for name, value in zip(param_names, params)}
        try:
            Y[i] = model_function(**param_dict)
            if (i + 1) % 10 == 0:
                logger.debug(f"  Completed {i+1}/{total_samples} runs")
        except Exception as e:
            logger.error(f"Error in run {i+1}: {e}")
            Y[i] = np.nan

    if np.any(np.isnan(Y)):
        logger.warning(f"{np.sum(np.isnan(Y))} model runs failed. Results may be unreliable.")

    # Analyze results
    logger.info("Analyzing Morris indices...")
    Si = morris_analyze.analyze(
        problem,
        param_values,
        Y,
        conf_level=0.95,
        print_to_console=False,
        num_levels=n_levels
    )

    logger.info("Morris Screening Results:")
    logger.info(f"{'Parameter':15s} {'μ*':>10s} {'σ':>10s}  Interpretation")
    logger.info("-" * 60)

    for i, name in enumerate(param_names):
        mu_star = Si['mu_star'][i]
        sigma = Si['sigma'][i]

        if mu_star > 0.1 and sigma > 0.1:
            interp = "Important + Interactions"
        elif mu_star > 0.1:
            interp = "Important"
        elif sigma > 0.1:
            interp = "Weak + Interactions"
        else:
            interp = "Unimportant"

        logger.info(f"{name:15s} {mu_star:10.4f} {sigma:10.4f}  {interp}")

    results = {
        'mu': Si['mu'],  # Mean of elementary effects
        'mu_star': Si['mu_star'],  # Mean of absolute effects (importance)
        'sigma': Si['sigma'],  # Standard deviation (interactions)
        'mu_star_conf': Si['mu_star_conf'],  # Confidence intervals
        'param_names': param_names,
        'problem': problem,
        'metric_name': metric_name,
        'n_samples': total_samples,
    }

    return results


def visualize_sensitivity_results(
    results: Dict[str, Any],
    method: str,
    output_path: Optional[Path] = None,
    show_plot: bool = True
) -> None:
    """Visualize sensitivity analysis results.

    Args:
        results: Results dictionary from sensitivity analysis
        method: Analysis method ('oat', 'sobol', or 'morris')
        output_path: Path to save figure (optional)
        show_plot: Whether to display the plot

    Requires:
        matplotlib package
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib is required for visualization. Install with: pip install matplotlib")
        return

    method = method.lower()

    if method == 'oat':
        _plot_oat_results(results, output_path, show_plot)
    elif method == 'sobol':
        _plot_sobol_results(results, output_path, show_plot)
    elif method == 'morris':
        _plot_morris_results(results, output_path, show_plot)
    else:
        logger.error(f"Unknown method: {method}")


def _plot_oat_results(results, output_path, show_plot):
    """Plot OAT sensitivity results (tornado diagram)."""
    import matplotlib.pyplot as plt

    param_names = list(results['sensitivity_indices'].keys())
    si_values = [results['sensitivity_indices'][p] for p in param_names]

    # Sort by absolute sensitivity
    sorted_indices = sorted(zip(param_names, si_values), key=lambda x: abs(x[1]), reverse=True)
    sorted_names, sorted_si = zip(*sorted_indices)

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(sorted_names))

    colors = ['red' if si < 0 else 'blue' for si in sorted_si]
    ax.barh(y_pos, sorted_si, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Sensitivity Index')
    ax.set_title(f'OAT Sensitivity Analysis - {results["metric_name"]}')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved OAT plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def _plot_sobol_results(results, output_path, show_plot):
    """Plot Sobol sensitivity results (bar chart)."""
    import matplotlib.pyplot as plt

    param_names = results['param_names']
    S1 = results['S1']
    ST = results['ST']

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(param_names))
    width = 0.35

    ax.bar(x - width/2, S1, width, label='S1 (First-order)', alpha=0.8)
    ax.bar(x + width/2, ST, width, label='ST (Total-order)', alpha=0.8)

    ax.set_xlabel('Parameters')
    ax.set_ylabel('Sobol Index')
    ax.set_title(f'Sobol Sensitivity Analysis - {results["metric_name"]}')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Sobol plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def _plot_morris_results(results, output_path, show_plot):
    """Plot Morris screening results (scatter plot: μ* vs σ)."""
    import matplotlib.pyplot as plt

    param_names = results['param_names']
    mu_star = results['mu_star']
    sigma = results['sigma']

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(mu_star, sigma, s=100, alpha=0.6)

    for i, name in enumerate(param_names):
        ax.annotate(name, (mu_star[i], sigma[i]),
                   xytext=(5, 5), textcoords='offset points')

    ax.set_xlabel('μ* (Mean of Absolute Effects)')
    ax.set_ylabel('σ (Standard Deviation)')
    ax.set_title(f'Morris Screening - {results["metric_name"]}')
    ax.grid(alpha=0.3)

    # Add interpretation regions
    ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5, linewidth=0.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Morris plot to {output_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


__all__ = [
    'one_at_a_time_sensitivity',
    'sobol_sensitivity',
    'morris_screening',
    'visualize_sensitivity_results',
]
