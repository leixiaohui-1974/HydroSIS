"""Automatic parameter calibration and optimization for hydrological models.

This module provides tools for automatic parameter calibration using various
optimization algorithms:

1. SCE-UA (Shuffled Complex Evolution - University of Arizona)
2. Differential Evolution (scipy-based)
3. PSO (Particle Swarm Optimization)
4. Bayesian optimization (optional)

Typical usage:
    >>> from hydrosis.analysis.calibration_optimization import calibrate_model
    >>>
    >>> def objective(params):
    ...     simulated = run_model(**params)
    ...     return nash_sutcliffe_efficiency(observed, simulated)
    >>>
    >>> result = calibrate_model(
    ...     objective_function=objective,
    ...     param_bounds={'FC': [100, 200], 'K0': [0.1, 0.5]},
    ...     method='sce_ua',
    ...     maximize=True
    ... )
"""
from __future__ import annotations

import logging
from typing import Dict, List, Callable, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import time
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

# Try to import optimization libraries
try:
    from scipy.optimize import differential_evolution, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some optimizers will not work.")


@dataclass
class CalibrationResult:
    """Result of parameter calibration.

    Attributes:
        best_params: Best parameter set found
        best_score: Best objective function value
        n_iterations: Number of iterations performed
        n_evaluations: Total number of function evaluations
        convergence_history: History of best scores over iterations
        param_history: History of best parameters over iterations
        computation_time: Total computation time in seconds
        success: Whether calibration was successful
        message: Status or error message
        method: Optimization method used
    """
    best_params: Dict[str, float]
    best_score: float
    n_iterations: int
    n_evaluations: int
    convergence_history: List[float]
    param_history: List[Dict[str, float]]
    computation_time: float
    success: bool
    message: str
    method: str
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def save_json(self, filepath: Path) -> None:
        """Save calibration result to JSON file."""
        data = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': self.n_iterations,
            'n_evaluations': self.n_evaluations,
            'convergence_history': self.convergence_history,
            'param_history': self.param_history,
            'computation_time': self.computation_time,
            'success': self.success,
            'message': self.message,
            'method': self.method,
            'additional_info': self.additional_info,
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Calibration result saved to {filepath}")

    def summary(self) -> str:
        """Generate a summary string of calibration results."""
        lines = [
            "=" * 60,
            "Parameter Calibration Results",
            "=" * 60,
            f"Method: {self.method}",
            f"Success: {self.success}",
            f"Message: {self.message}",
            f"",
            f"Best Score: {self.best_score:.6f}",
            f"Iterations: {self.n_iterations}",
            f"Function Evaluations: {self.n_evaluations}",
            f"Computation Time: {self.computation_time:.2f} seconds",
            f"",
            "Best Parameters:",
        ]
        for name, value in self.best_params.items():
            lines.append(f"  {name:15s}: {value:10.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def sce_ua_optimization(
    objective_function: Callable,
    param_bounds: Dict[str, List[float]],
    maximize: bool = True,
    n_complexes: int = 5,
    max_iterations: int = 1000,
    n_evolution_steps: int = None,
    min_change: float = 0.001,
    patience: int = 10,
    seed: Optional[int] = None,
    verbose: bool = True
) -> CalibrationResult:
    """Shuffled Complex Evolution (SCE-UA) global optimization algorithm.

    SCE-UA is a robust global optimization method widely used in hydrology.
    It combines features of:
    - Competitive Complex Evolution (CCE)
    - Downhill Simplex Method
    - Random shuffling

    Args:
        objective_function: Function to optimize, signature: func(**params) -> float
        param_bounds: Parameter bounds {name: [min, max]}
        maximize: Whether to maximize (True) or minimize (False) the objective
        n_complexes: Number of complexes (communities), typically 5-10
        max_iterations: Maximum number of iterations
        n_evolution_steps: Evolution steps per complex (default: 2*n_params + 1)
        min_change: Minimum improvement to consider as progress
        patience: Stop if no improvement for this many iterations
        seed: Random seed for reproducibility
        verbose: Print progress information

    Returns:
        CalibrationResult with optimization results

    Reference:
        Duan, Q., et al. (1992). Effective and efficient global optimization
        for conceptual rainfall-runoff models. Water Resources Research, 28(4).
    """
    if seed is not None:
        np.random.seed(seed)

    param_names = list(param_bounds.keys())
    n_params = len(param_names)

    # Algorithm parameters
    if n_evolution_steps is None:
        n_evolution_steps = 2 * n_params + 1

    # Points per complex (minimum for simplex is n_params + 1)
    points_per_complex = 2 * n_params + 1

    # Total population size
    population_size = n_complexes * points_per_complex

    logger.info(f"Starting SCE-UA optimization")
    logger.info(f"  Parameters: {n_params}")
    logger.info(f"  Complexes: {n_complexes}")
    logger.info(f"  Population size: {population_size}")
    logger.info(f"  Max iterations: {max_iterations}")

    # Get bounds arrays
    lower_bounds = np.array([param_bounds[name][0] for name in param_names])
    upper_bounds = np.array([param_bounds[name][1] for name in param_names])

    # Initialize population
    population = np.random.uniform(
        lower_bounds,
        upper_bounds,
        size=(population_size, n_params)
    )

    # Evaluate initial population
    start_time = time.time()
    scores = np.zeros(population_size)

    for i in range(population_size):
        params_dict = {name: population[i, j] for j, name in enumerate(param_names)}
        try:
            score = objective_function(**params_dict)
            scores[i] = score if maximize else -score
        except Exception as e:
            logger.error(f"Error evaluating parameters {params_dict}: {e}")
            scores[i] = -np.inf

    n_evaluations = population_size

    # Sort population by fitness (descending)
    sorted_indices = np.argsort(scores)[::-1]
    population = population[sorted_indices]
    scores = scores[sorted_indices]

    # Tracking
    best_score = scores[0]
    best_params = population[0].copy()
    convergence_history = [best_score]
    param_history = [{name: best_params[j] for j, name in enumerate(param_names)}]

    no_improvement_count = 0
    last_best = best_score

    if verbose:
        logger.info(f"Initial best score: {best_score:.6f}")

    # Main loop
    for iteration in range(max_iterations):
        # Partition population into complexes
        complexes = []
        for k in range(n_complexes):
            complex_indices = list(range(k, population_size, n_complexes))
            complex_pop = population[complex_indices]
            complex_scores = scores[complex_indices]
            complexes.append((complex_pop, complex_scores))

        # Evolve each complex
        new_population = []
        new_scores = []

        for complex_pop, complex_scores in complexes:
            # Evolve complex using Competitive Complex Evolution (CCE)
            evolved_pop, evolved_scores, new_evals = _evolve_complex(
                complex_pop,
                complex_scores,
                objective_function,
                param_names,
                lower_bounds,
                upper_bounds,
                n_evolution_steps,
                maximize
            )
            n_evaluations += new_evals

            new_population.append(evolved_pop)
            new_scores.append(evolved_scores)

        # Combine all complexes
        population = np.vstack(new_population)
        scores = np.concatenate(new_scores)

        # Sort by fitness
        sorted_indices = np.argsort(scores)[::-1]
        population = population[sorted_indices]
        scores = scores[sorted_indices]

        # Update best
        current_best = scores[0]
        if current_best > best_score:
            improvement = current_best - best_score
            best_score = current_best
            best_params = population[0].copy()

            if verbose and improvement > min_change:
                logger.info(f"Iteration {iteration+1}: Best score = {best_score:.6f} "
                          f"(+{improvement:.6f})")

        convergence_history.append(best_score)
        param_history.append({name: best_params[j] for j, name in enumerate(param_names)})

        # Check convergence
        if abs(best_score - last_best) < min_change:
            no_improvement_count += 1
        else:
            no_improvement_count = 0

        if no_improvement_count >= patience:
            logger.info(f"Converged: No improvement for {patience} iterations")
            break

        last_best = best_score

        # Progress report
        if verbose and (iteration + 1) % 10 == 0:
            logger.info(f"Iteration {iteration+1}/{max_iterations}: "
                       f"Score = {best_score:.6f}, Evals = {n_evaluations}")

    computation_time = time.time() - start_time

    # Convert best score back if minimizing
    final_score = best_score if maximize else -best_score
    convergence_history = [s if maximize else -s for s in convergence_history]

    best_params_dict = {name: float(best_params[j]) for j, name in enumerate(param_names)}

    logger.info(f"SCE-UA completed in {computation_time:.2f}s")
    logger.info(f"Final score: {final_score:.6f}")

    return CalibrationResult(
        best_params=best_params_dict,
        best_score=final_score,
        n_iterations=iteration + 1,
        n_evaluations=n_evaluations,
        convergence_history=convergence_history,
        param_history=param_history,
        computation_time=computation_time,
        success=True,
        message=f"SCE-UA completed after {iteration+1} iterations",
        method="SCE-UA",
        additional_info={
            'n_complexes': n_complexes,
            'population_size': population_size,
            'final_population_diversity': float(np.std(scores)),
        }
    )


def _evolve_complex(
    complex_pop: np.ndarray,
    complex_scores: np.ndarray,
    objective_function: Callable,
    param_names: List[str],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    n_steps: int,
    maximize: bool
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Evolve a complex using Competitive Complex Evolution (CCE).

    Uses a combination of:
    1. Triangular probability distribution for parent selection
    2. Simplex reflection/contraction
    3. Random mutation if reflection fails
    """
    n_points, n_params = complex_pop.shape
    n_evaluations = 0

    for _ in range(n_steps):
        # Select parents using triangular distribution (favor better points)
        # Probability: p(i) ∝ (n+1-i) where i is rank
        probs = np.arange(n_points, 0, -1, dtype=float)
        probs /= probs.sum()

        # Select n_params + 1 points for simplex
        selected_indices = np.random.choice(n_points, size=n_params+1, replace=False, p=probs)
        simplex = complex_pop[selected_indices]
        simplex_scores = complex_scores[selected_indices]

        # Sort simplex by score
        sorted_idx = np.argsort(simplex_scores)[::-1]
        simplex = simplex[sorted_idx]
        simplex_scores = simplex_scores[sorted_idx]

        # Calculate centroid (excluding worst point)
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        worst = simplex[-1]
        reflected = centroid + (centroid - worst)  # Reflection coefficient = 1.0

        # Clip to bounds
        reflected = np.clip(reflected, lower_bounds, upper_bounds)

        # Evaluate reflected point
        params_dict = {name: reflected[j] for j, name in enumerate(param_names)}
        try:
            reflected_score = objective_function(**params_dict)
            reflected_score = reflected_score if maximize else -reflected_score
            n_evaluations += 1
        except Exception:
            reflected_score = -np.inf

        # Accept or reject reflected point
        if reflected_score > simplex_scores[-1]:
            # Replace worst point
            worst_idx_in_complex = selected_indices[sorted_idx[-1]]
            complex_pop[worst_idx_in_complex] = reflected
            complex_scores[worst_idx_in_complex] = reflected_score
        else:
            # Contraction
            contracted = centroid + 0.5 * (worst - centroid)
            contracted = np.clip(contracted, lower_bounds, upper_bounds)

            params_dict = {name: contracted[j] for j, name in enumerate(param_names)}
            try:
                contracted_score = objective_function(**params_dict)
                contracted_score = contracted_score if maximize else -contracted_score
                n_evaluations += 1
            except Exception:
                contracted_score = -np.inf

            if contracted_score > simplex_scores[-1]:
                worst_idx_in_complex = selected_indices[sorted_idx[-1]]
                complex_pop[worst_idx_in_complex] = contracted
                complex_scores[worst_idx_in_complex] = contracted_score
            else:
                # Random mutation
                mutated = np.random.uniform(lower_bounds, upper_bounds)
                params_dict = {name: mutated[j] for j, name in enumerate(param_names)}
                try:
                    mutated_score = objective_function(**params_dict)
                    mutated_score = mutated_score if maximize else -mutated_score
                    n_evaluations += 1
                except Exception:
                    mutated_score = -np.inf

                worst_idx_in_complex = selected_indices[sorted_idx[-1]]
                complex_pop[worst_idx_in_complex] = mutated
                complex_scores[worst_idx_in_complex] = mutated_score

        # Re-sort complex
        sorted_indices = np.argsort(complex_scores)[::-1]
        complex_pop = complex_pop[sorted_indices]
        complex_scores = complex_scores[sorted_indices]

    return complex_pop, complex_scores, n_evaluations


def differential_evolution_optimization(
    objective_function: Callable,
    param_bounds: Dict[str, List[float]],
    maximize: bool = True,
    population_size: int = 15,
    max_iterations: int = 1000,
    tolerance: float = 0.001,
    seed: Optional[int] = None,
    verbose: bool = True
) -> CalibrationResult:
    """Differential Evolution optimization using scipy.

    Args:
        objective_function: Function to optimize
        param_bounds: Parameter bounds
        maximize: Whether to maximize the objective
        population_size: Population size multiplier (popsize * n_params)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        seed: Random seed
        verbose: Print progress

    Returns:
        CalibrationResult
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy is required for Differential Evolution. Install with: pip install scipy")

    param_names = list(param_bounds.keys())
    bounds = [param_bounds[name] for name in param_names]

    logger.info(f"Starting Differential Evolution optimization")
    logger.info(f"  Parameters: {len(param_names)}")
    logger.info(f"  Population size: {population_size * len(param_names)}")

    # Wrapper to handle maximize/minimize
    n_evaluations = [0]
    convergence_history = []
    param_history = []

    def wrapped_objective(x):
        params_dict = {name: x[i] for i, name in enumerate(param_names)}
        try:
            score = objective_function(**params_dict)
            n_evaluations[0] += 1

            # Track best
            if maximize:
                convergence_history.append(score if not convergence_history or score > max(convergence_history) else max(convergence_history))
            else:
                convergence_history.append(score if not convergence_history or score < min(convergence_history) else min(convergence_history))

            param_history.append(params_dict.copy())

            return -score if maximize else score
        except Exception as e:
            logger.error(f"Error evaluating parameters: {e}")
            return np.inf

    start_time = time.time()

    result = differential_evolution(
        wrapped_objective,
        bounds,
        seed=seed,
        maxiter=max_iterations,
        popsize=population_size,
        tol=tolerance,
        atol=tolerance,
        disp=verbose,
        polish=True
    )

    computation_time = time.time() - start_time

    best_params_dict = {name: float(result.x[i]) for i, name in enumerate(param_names)}
    best_score = -result.fun if maximize else result.fun

    logger.info(f"Differential Evolution completed in {computation_time:.2f}s")
    logger.info(f"Final score: {best_score:.6f}")

    return CalibrationResult(
        best_params=best_params_dict,
        best_score=best_score,
        n_iterations=result.nit,
        n_evaluations=n_evaluations[0],
        convergence_history=convergence_history,
        param_history=param_history,
        computation_time=computation_time,
        success=result.success,
        message=result.message,
        method="Differential Evolution",
        additional_info={'scipy_result': str(result)}
    )


def calibrate_model(
    objective_function: Callable,
    param_bounds: Dict[str, List[float]],
    method: str = 'sce_ua',
    maximize: bool = True,
    **kwargs
) -> CalibrationResult:
    """Calibrate model parameters using specified optimization method.

    Args:
        objective_function: Function to optimize, signature: func(**params) -> float
        param_bounds: Parameter bounds {name: [min, max]}
        method: Optimization method ('sce_ua', 'differential_evolution')
        maximize: Whether to maximize (True) or minimize (False) the objective
        **kwargs: Additional arguments passed to the specific optimizer

    Returns:
        CalibrationResult

    Example:
        >>> from hydrosis.analysis import calibrate_model, nash_sutcliffe_efficiency
        >>>
        >>> def objective(FC, K0, BETA):
        ...     simulated = run_hbv_model(FC=FC, K0=K0, BETA=BETA)
        ...     return nash_sutcliffe_efficiency(observed, simulated)
        >>>
        >>> result = calibrate_model(
        ...     objective_function=objective,
        ...     param_bounds={'FC': [100, 200], 'K0': [0.1, 0.5], 'BETA': [0.5, 2.0]},
        ...     method='sce_ua',
        ...     maximize=True,
        ...     max_iterations=100
        ... )
        >>> print(result.summary())
    """
    method = method.lower()

    if method == 'sce_ua':
        return sce_ua_optimization(objective_function, param_bounds, maximize, **kwargs)
    elif method in ['de', 'differential_evolution']:
        return differential_evolution_optimization(objective_function, param_bounds, maximize, **kwargs)
    else:
        raise ValueError(f"Unknown calibration method: {method}. "
                        f"Available: 'sce_ua', 'differential_evolution'")


__all__ = [
    'CalibrationResult',
    'sce_ua_optimization',
    'differential_evolution_optimization',
    'calibrate_model',
]
