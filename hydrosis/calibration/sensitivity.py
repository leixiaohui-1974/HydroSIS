"""
Parameter sensitivity analysis for hydrological models.

Provides methods for analyzing parameter sensitivity and using
sensitivity information to improve calibration efficiency.
"""
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Sequence, Tuple
from dataclasses import dataclass
import time


@dataclass
class SensitivityResult:
    """
    Results from parameter sensitivity analysis.

    Attributes
    ----------
    param_names : List[str]
        Names of the parameters
    sensitivity_indices : Dict[str, float]
        Sensitivity index for each parameter (0-1, higher = more sensitive)
    sensitivity_rankings : List[str]
        Parameters ranked by sensitivity (most to least)
    parameter_ranges : Dict[str, Tuple[float, float]]
        Original parameter bounds
    base_output : float
        Model output at base parameter values
    n_samples : int
        Number of samples used in analysis
    computation_time : float
        Time taken for analysis (seconds)
    method : str
        Method used for sensitivity analysis
    """

    param_names: List[str]
    sensitivity_indices: Dict[str, float]
    sensitivity_rankings: List[str]
    parameter_ranges: Dict[str, Tuple[float, float]]
    base_output: float
    n_samples: int
    computation_time: float
    method: str


def one_at_a_time_sensitivity(
    model_function: Callable[[Sequence[float]], float],
    param_names: List[str],
    param_bounds: Sequence[Tuple[float, float]],
    n_samples: int = 10,
    base_params: Sequence[float] | None = None,
) -> SensitivityResult:
    """
    One-at-a-time (OAT) parameter sensitivity analysis.

    Varies each parameter individually while keeping others constant,
    measuring the impact on model output.

    Parameters
    ----------
    model_function : Callable
        Model function that takes parameter values and returns a scalar output
    param_names : List[str]
        Names of the parameters
    param_bounds : Sequence[Tuple[float, float]]
        Parameter bounds as list of (min, max) tuples
    n_samples : int, optional
        Number of samples per parameter. Default is 10.
    base_params : Sequence[float] or None, optional
        Base parameter values. If None, uses midpoint of ranges.

    Returns
    -------
    SensitivityResult
        Object containing sensitivity analysis results

    Examples
    --------
    >>> def model(params):
    ...     return params[0]**2 + params[1]  # More sensitive to param[0]
    >>> result = one_at_a_time_sensitivity(
    ...     model,
    ...     param_names=['a', 'b'],
    ...     param_bounds=[(0, 10), (0, 10)],
    ...     n_samples=20
    ... )
    >>> print(result.sensitivity_rankings)
    ['a', 'b']  # 'a' is more sensitive
    """
    start_time = time.time()

    # Use midpoint as base if not provided
    if base_params is None:
        base_params = [np.mean(bounds) for bounds in param_bounds]
    else:
        base_params = list(base_params)

    base_output = model_function(base_params)

    # Store sensitivity measures
    sensitivity_indices = {}

    for i, param_name in enumerate(param_names):
        min_val, max_val = param_bounds[i]
        param_values = np.linspace(min_val, max_val, n_samples)

        outputs = []
        for val in param_values:
            # Vary one parameter, keep others at base
            test_params = base_params.copy()
            test_params[i] = val
            output = model_function(test_params)
            outputs.append(output)

        outputs = np.array(outputs)

        # Calculate sensitivity index as normalized standard deviation
        # Higher variation = higher sensitivity
        output_std = np.std(outputs)
        output_range = np.ptp(outputs)  # peak-to-peak (max - min)

        # Normalize by parameter range to make comparable
        param_range = max_val - min_val
        normalized_sensitivity = output_range / (param_range + 1e-10)

        sensitivity_indices[param_name] = float(normalized_sensitivity)

    # Rank parameters by sensitivity
    sensitivity_rankings = sorted(
        param_names,
        key=lambda x: sensitivity_indices[x],
        reverse=True
    )

    # Normalize sensitivity indices to [0, 1]
    max_sensitivity = max(sensitivity_indices.values()) if sensitivity_indices else 1.0
    if max_sensitivity > 0:
        sensitivity_indices = {
            k: v / max_sensitivity for k, v in sensitivity_indices.items()
        }

    end_time = time.time()

    return SensitivityResult(
        param_names=param_names,
        sensitivity_indices=sensitivity_indices,
        sensitivity_rankings=sensitivity_rankings,
        parameter_ranges={
            name: bounds for name, bounds in zip(param_names, param_bounds)
        },
        base_output=base_output,
        n_samples=n_samples * len(param_names),
        computation_time=end_time - start_time,
        method="One-at-a-time (OAT)"
    )


def morris_sensitivity(
    model_function: Callable[[Sequence[float]], float],
    param_names: List[str],
    param_bounds: Sequence[Tuple[float, float]],
    n_trajectories: int = 10,
    n_levels: int = 4,
) -> SensitivityResult:
    """
    Morris (Elementary Effects) sensitivity analysis.

    A more efficient global sensitivity analysis method that explores
    the parameter space using trajectories.

    Parameters
    ----------
    model_function : Callable
        Model function that takes parameter values and returns a scalar output
    param_names : List[str]
        Names of the parameters
    param_bounds : Sequence[Tuple[float, float]]
        Parameter bounds as list of (min, max) tuples
    n_trajectories : int, optional
        Number of trajectories to sample. Default is 10.
    n_levels : int, optional
        Number of levels for grid. Default is 4.

    Returns
    -------
    SensitivityResult
        Object containing sensitivity analysis results

    Notes
    -----
    Morris method is more efficient than full factorial designs while
    still providing global sensitivity information.
    """
    start_time = time.time()

    n_params = len(param_names)
    delta = n_levels / (2 * (n_levels - 1))  # Step size

    # Store elementary effects for each parameter
    elementary_effects = {name: [] for name in param_names}

    total_samples = 0

    for traj in range(n_trajectories):
        # Generate random starting point
        base_point = np.random.rand(n_params)

        # Scale to parameter bounds
        base_params = [
            min_val + base_point[i] * (max_val - min_val)
            for i, (min_val, max_val) in enumerate(param_bounds)
        ]

        base_output = model_function(base_params)
        total_samples += 1

        # Vary each parameter one at a time
        for i, param_name in enumerate(param_names):
            # Create perturbed parameters
            perturbed_params = base_params.copy()
            min_val, max_val = param_bounds[i]
            param_range = max_val - min_val

            # Add perturbation
            perturbation = delta * param_range
            perturbed_params[i] = min(max_val, base_params[i] + perturbation)

            perturbed_output = model_function(perturbed_params)
            total_samples += 1

            # Calculate elementary effect
            ee = (perturbed_output - base_output) / perturbation
            elementary_effects[param_name].append(abs(ee))

    # Calculate mean and standard deviation of elementary effects
    sensitivity_indices = {}
    for param_name in param_names:
        effects = elementary_effects[param_name]
        # Use mean absolute elementary effect as sensitivity measure
        mu_star = np.mean(effects)
        sensitivity_indices[param_name] = float(mu_star)

    # Normalize sensitivity indices
    max_sensitivity = max(sensitivity_indices.values()) if sensitivity_indices else 1.0
    if max_sensitivity > 0:
        sensitivity_indices = {
            k: v / max_sensitivity for k, v in sensitivity_indices.items()
        }

    # Rank parameters
    sensitivity_rankings = sorted(
        param_names,
        key=lambda x: sensitivity_indices[x],
        reverse=True
    )

    # Base output (average over trajectories)
    base_params = [np.mean(bounds) for bounds in param_bounds]
    base_output = model_function(base_params)

    end_time = time.time()

    return SensitivityResult(
        param_names=param_names,
        sensitivity_indices=sensitivity_indices,
        sensitivity_rankings=sensitivity_rankings,
        parameter_ranges={
            name: bounds for name, bounds in zip(param_names, param_bounds)
        },
        base_output=base_output,
        n_samples=total_samples,
        computation_time=end_time - start_time,
        method="Morris (Elementary Effects)"
    )


def adaptive_bounds_from_sensitivity(
    sensitivity_result: SensitivityResult,
    focus_factor: float = 2.0,
) -> Sequence[Tuple[float, float]]:
    """
    Create adaptive parameter bounds based on sensitivity analysis.

    High-sensitivity parameters get wider search ranges,
    low-sensitivity parameters get narrower ranges focused near baseline.

    Parameters
    ----------
    sensitivity_result : SensitivityResult
        Results from sensitivity analysis
    focus_factor : float, optional
        Factor controlling range adjustment. Higher = more aggressive.
        Default is 2.0.

    Returns
    -------
    Sequence[Tuple[float, float]]
        Adjusted parameter bounds

    Examples
    --------
    >>> # After sensitivity analysis
    >>> adaptive_bounds = adaptive_bounds_from_sensitivity(sens_result)
    >>> # Use in calibration
    >>> result = calibrate_parameters(
    ...     objective_function,
    ...     param_bounds=adaptive_bounds
    ... )
    """
    adaptive_bounds = []

    for param_name in sensitivity_result.param_names:
        original_min, original_max = sensitivity_result.parameter_ranges[param_name]
        original_mid = (original_min + original_max) / 2
        original_range = original_max - original_min

        sensitivity = sensitivity_result.sensitivity_indices[param_name]

        # High sensitivity (>0.7): keep full range or expand
        # Medium sensitivity (0.3-0.7): keep range
        # Low sensitivity (<0.3): narrow range around midpoint

        if sensitivity > 0.7:
            # High sensitivity: keep or slightly expand
            scale_factor = 1.0 + (sensitivity - 0.7) * focus_factor * 0.5
        elif sensitivity > 0.3:
            # Medium sensitivity: keep range
            scale_factor = 1.0
        else:
            # Low sensitivity: narrow range
            scale_factor = 0.3 + sensitivity * 2.0  # 0.3 to 0.9

        new_range = original_range * scale_factor
        new_min = original_mid - new_range / 2
        new_max = original_mid + new_range / 2

        # Ensure bounds don't exceed original bounds
        new_min = max(original_min, new_min)
        new_max = min(original_max, new_max)

        adaptive_bounds.append((new_min, new_max))

    return adaptive_bounds


def print_sensitivity_report(sensitivity_result: SensitivityResult):
    """
    Print a formatted sensitivity analysis report.

    Parameters
    ----------
    sensitivity_result : SensitivityResult
        Results from sensitivity analysis
    """
    print("\n" + "=" * 80)
    print("Parameter Sensitivity Analysis Report")
    print("=" * 80)

    print(f"\nMethod: {sensitivity_result.method}")
    print(f"Samples: {sensitivity_result.n_samples}")
    print(f"Computation Time: {sensitivity_result.computation_time:.2f}s")
    print(f"Base Output: {sensitivity_result.base_output:.6f}")

    print("\nParameter Sensitivity Rankings:")
    print("-" * 80)
    print(f"{'Rank':<6} {'Parameter':<20} {'Sensitivity':<15} {'Range':<30}")
    print("-" * 80)

    for rank, param_name in enumerate(sensitivity_result.sensitivity_rankings, 1):
        sensitivity = sensitivity_result.sensitivity_indices[param_name]
        param_range = sensitivity_result.parameter_ranges[param_name]

        # Visual bar
        bar_length = int(sensitivity * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)

        print(f"{rank:<6} {param_name:<20} {sensitivity:>6.4f} {bar}  "
              f"[{param_range[0]:.2f}, {param_range[1]:.2f}]")

    print("\nInterpretation:")
    print("-" * 80)

    high_sens = [p for p in sensitivity_result.param_names
                 if sensitivity_result.sensitivity_indices[p] > 0.7]
    med_sens = [p for p in sensitivity_result.param_names
                if 0.3 < sensitivity_result.sensitivity_indices[p] <= 0.7]
    low_sens = [p for p in sensitivity_result.param_names
                if sensitivity_result.sensitivity_indices[p] <= 0.3]

    if high_sens:
        print(f"High Sensitivity (>0.7): {', '.join(high_sens)}")
        print("  → Focus calibration efforts on these parameters")

    if med_sens:
        print(f"Medium Sensitivity (0.3-0.7): {', '.join(med_sens)}")
        print("  → Include in calibration but less critical")

    if low_sens:
        print(f"Low Sensitivity (<0.3): {', '.join(low_sens)}")
        print("  → Consider fixing these at reasonable values")

    print("\n" + "=" * 80)


__all__ = [
    "SensitivityResult",
    "one_at_a_time_sensitivity",
    "morris_sensitivity",
    "adaptive_bounds_from_sensitivity",
    "print_sensitivity_report",
]
