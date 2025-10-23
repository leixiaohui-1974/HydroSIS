"""Optimization algorithms for parameter calibration."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Sequence

try:
    from scipy.optimize import differential_evolution, OptimizeResult
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    OptimizeResult = None  # type: ignore


@dataclass
class CalibrationResult:
    """
    Result of parameter calibration.

    Attributes
    ----------
    success : bool
        Whether calibration succeeded
    best_params : list[float]
        Optimized parameter values
    best_score : float
        Best objective function value achieved
    n_iterations : int
        Number of iterations performed
    n_evaluations : int
        Total number of function evaluations
    computation_time : float
        Total computation time in seconds
    convergence_history : list[float]
        History of best scores at each iteration
    message : str
        Status message from optimizer
    algorithm : str
        Name of the algorithm used
    """

    success: bool
    best_params: list[float]
    best_score: float
    n_iterations: int
    n_evaluations: int
    computation_time: float
    convergence_history: list[float] = field(default_factory=list)
    message: str = ""
    algorithm: str = ""


def differential_evolution_calibrate(
    objective_function: Callable[[Sequence[float]], float],
    param_bounds: Sequence[tuple[float, float]],
    maximize: bool = True,
    maxiter: int = 100,
    popsize: int = 15,
    seed: int | None = None,
    polish: bool = True,
    callback: Callable | None = None,
) -> CalibrationResult:
    """
    Calibrate parameters using Differential Evolution algorithm.

    Differential Evolution is a global optimization algorithm suitable for
    non-linear, non-convex parameter estimation problems commonly found in
    hydrological modeling.

    Parameters
    ----------
    objective_function : Callable
        Function to minimize (or maximize). Takes a sequence of parameter values
        and returns a scalar score.
    param_bounds : Sequence[tuple[float, float]]
        List of (min, max) bounds for each parameter
    maximize : bool, optional
        If True, maximizes the objective function. If False, minimizes it.
        Default is True (for metrics like NSE, KGE).
    maxiter : int, optional
        Maximum number of generations. Default is 100.
    popsize : int, optional
        Population size multiplier (actual population = popsize * n_params).
        Default is 15.
    seed : int or None, optional
        Random seed for reproducibility. Default is None.
    polish : bool, optional
        If True, uses local optimization to refine the result. Default is True.
    callback : Callable or None, optional
        Callback function called after each iteration.

    Returns
    -------
    CalibrationResult
        Object containing optimization results and statistics

    Raises
    ------
    ImportError
        If scipy is not installed

    Examples
    --------
    >>> def nse_objective(params):
    ...     # Run model with params and calculate NSE
    ...     return nse_value
    >>> bounds = [(100, 500), (1.0, 3.0), (0.1, 0.5)]
    >>> result = differential_evolution_calibrate(nse_objective, bounds, maximize=True)
    >>> print(f"Best NSE: {result.best_score:.4f}")
    >>> print(f"Optimized parameters: {result.best_params}")
    """
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for differential_evolution_calibrate. "
            "Install it with: pip install scipy"
        )

    start_time = time.time()
    convergence_history = []

    # Wrapper to handle maximize/minimize
    def wrapped_objective(x):
        score = objective_function(x)
        return -score if maximize else score

    # Callback to record convergence
    def iteration_callback(xk, convergence=None):
        raw_score = objective_function(xk)
        convergence_history.append(raw_score)
        if callback is not None:
            callback(xk, convergence)
        return False

    # Run differential evolution
    result: OptimizeResult = differential_evolution(
        wrapped_objective,
        bounds=param_bounds,
        strategy="best1bin",
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=seed,
        callback=iteration_callback,
        polish=polish,
        workers=1,  # Single-threaded for reproducibility
        updating="deferred",
        disp=False,
    )

    end_time = time.time()

    # Convert back to maximization if needed
    best_score = -result.fun if maximize else result.fun

    return CalibrationResult(
        success=result.success,
        best_params=result.x.tolist(),
        best_score=best_score,
        n_iterations=result.nit if hasattr(result, 'nit') else maxiter,
        n_evaluations=result.nfev,
        computation_time=end_time - start_time,
        convergence_history=convergence_history,
        message=result.message,
        algorithm="Differential Evolution",
    )


def calibrate_parameters(
    objective_function: Callable[[Sequence[float]], float],
    param_bounds: Sequence[tuple[float, float]],
    algorithm: str = "differential_evolution",
    maximize: bool = True,
    **kwargs,
) -> CalibrationResult:
    """
    Generic parameter calibration interface supporting multiple algorithms.

    This is a high-level function that dispatches to specific optimization
    algorithms based on the `algorithm` parameter.

    Parameters
    ----------
    objective_function : Callable
        Function to optimize. Takes parameter values, returns scalar score.
    param_bounds : Sequence[tuple[float, float]]
        Parameter bounds as list of (min, max) tuples
    algorithm : str, optional
        Optimization algorithm to use. Options:
        - "differential_evolution" (default): Scipy's differential evolution
        Default is "differential_evolution".
    maximize : bool, optional
        Whether to maximize (True) or minimize (False) the objective.
        Default is True.
    **kwargs
        Additional keyword arguments passed to the specific algorithm.
        For differential_evolution: maxiter, popsize, seed, polish

    Returns
    -------
    CalibrationResult
        Optimization results and statistics

    Raises
    ------
    ValueError
        If an unknown algorithm is specified
    ImportError
        If required dependencies are not installed

    Examples
    --------
    >>> # Using default algorithm (differential evolution)
    >>> result = calibrate_parameters(
    ...     objective_function=my_nse_function,
    ...     param_bounds=[(100, 500), (1.0, 3.0)],
    ...     maximize=True,
    ...     maxiter=100
    ... )

    >>> # Explicitly specify algorithm
    >>> result = calibrate_parameters(
    ...     objective_function=my_rmse_function,
    ...     param_bounds=[(0, 1), (0, 10)],
    ...     algorithm="differential_evolution",
    ...     maximize=False,
    ...     popsize=20
    ... )
    """
    algorithm = algorithm.lower().replace("-", "_").replace(" ", "_")

    if algorithm == "differential_evolution":
        return differential_evolution_calibrate(
            objective_function=objective_function,
            param_bounds=param_bounds,
            maximize=maximize,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown calibration algorithm: {algorithm}. "
            f"Supported algorithms: differential_evolution"
        )
