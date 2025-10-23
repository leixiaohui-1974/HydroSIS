"""Visualization tools for analysis results.

This module provides plotting functions for:
- Model performance comparison (observed vs simulated)
- Uncertainty envelopes
- Parameter distributions
- Calibration convergence
- Sensitivity analysis results

All functions require matplotlib.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Install with: pip install matplotlib")


def _check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


def plot_hydrograph_comparison(
    observed: np.ndarray,
    simulated: np.ndarray,
    time_index: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Hydrograph Comparison",
    xlabel: str = "Time",
    ylabel: str = "Discharge (m³/s)",
    figsize: Tuple[float, float] = (12, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot observed vs simulated hydrographs.

    Args:
        observed: Observed time series
        simulated: Simulated time series
        time_index: Optional time index (datetime or numeric)
        metrics: Optional performance metrics to display
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size (width, height)
        save_path: Path to save figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    if time_index is None:
        time_index = np.arange(len(observed))

    # Plot time series
    ax.plot(time_index, observed, 'k-', linewidth=1.5, label='Observed', alpha=0.8)
    ax.plot(time_index, simulated, 'r--', linewidth=1.2, label='Simulated', alpha=0.8)

    # Add metrics text box if provided
    if metrics:
        textstr = '\n'.join([f'{k.upper()}: {v:.3f}' for k, v in metrics.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved hydrograph to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_scatter(
    observed: np.ndarray,
    simulated: np.ndarray,
    metrics: Optional[Dict[str, float]] = None,
    title: str = "Observed vs Simulated",
    xlabel: str = "Observed",
    ylabel: str = "Simulated",
    figsize: Tuple[float, float] = (7, 7),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot scatter plot of observed vs simulated values.

    Args:
        observed: Observed values
        simulated: Simulated values
        metrics: Optional performance metrics
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Scatter plot
    ax.scatter(observed, simulated, alpha=0.5, s=20, edgecolors='none')

    # 1:1 line
    min_val = min(observed.min(), simulated.min())
    max_val = max(observed.max(), simulated.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5,
           label='1:1 line', alpha=0.7)

    # Add metrics text box if provided
    if metrics:
        textstr = '\n'.join([f'{k.upper()}: {v:.3f}' for k, v in metrics.items()])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_uncertainty_envelope(
    time_index: np.ndarray,
    observed: Optional[np.ndarray],
    mean_simulation: np.ndarray,
    percentiles: Dict[str, np.ndarray],
    title: str = "Uncertainty Envelope",
    xlabel: str = "Time",
    ylabel: str = "Discharge (m³/s)",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot uncertainty envelope with percentile bands.

    Args:
        time_index: Time index
        observed: Optional observed data
        mean_simulation: Mean simulated values
        percentiles: Dictionary of percentile arrays, e.g., {'p5': array, 'p95': array}
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure

    Example:
        >>> plot_uncertainty_envelope(
        ...     time_index=np.arange(100),
        ...     observed=obs,
        ...     mean_simulation=mean_sim,
        ...     percentiles={'p5': lower, 'p25': q1, 'p75': q3, 'p95': upper}
        ... )
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    # Sort percentile keys
    perc_keys = sorted([k for k in percentiles.keys() if k.startswith('p')],
                      key=lambda x: int(x[1:]))

    # Plot uncertainty bands
    if len(perc_keys) >= 2:
        # Outer band (e.g., 5-95%)
        lower_key = perc_keys[0]
        upper_key = perc_keys[-1]
        ax.fill_between(time_index, percentiles[lower_key], percentiles[upper_key],
                       alpha=0.2, color='blue', label=f'{lower_key[1:]}-{upper_key[1:]}% CI')

    if len(perc_keys) >= 4:
        # Inner band (e.g., 25-75%)
        lower_key = perc_keys[1]
        upper_key = perc_keys[-2]
        ax.fill_between(time_index, percentiles[lower_key], percentiles[upper_key],
                       alpha=0.3, color='blue', label=f'{lower_key[1:]}-{upper_key[1:]}% CI')

    # Plot mean
    ax.plot(time_index, mean_simulation, 'b-', linewidth=1.5, label='Mean simulation', alpha=0.8)

    # Plot observed if provided
    if observed is not None:
        ax.plot(time_index, observed, 'k-', linewidth=1.5, label='Observed', alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved uncertainty envelope to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_parameter_distributions(
    param_samples: Dict[str, np.ndarray],
    true_values: Optional[Dict[str, float]] = None,
    n_bins: int = 30,
    figsize: Optional[Tuple[float, float]] = None,
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot histograms of parameter posterior distributions.

    Args:
        param_samples: Dictionary of parameter samples {name: array}
        true_values: Optional true parameter values to mark
        n_bins: Number of histogram bins
        figsize: Figure size (auto if None)
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    n_params = len(param_samples)
    ncols = min(3, n_params)
    nrows = int(np.ceil(n_params / ncols))

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (param_name, samples) in enumerate(param_samples.items()):
        ax = axes[i]

        # Histogram
        ax.hist(samples, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')

        # Mark true value if provided
        if true_values and param_name in true_values:
            true_val = true_values[param_name]
            ax.axvline(true_val, color='red', linestyle='--', linewidth=2,
                      label=f'True: {true_val:.3f}')
            ax.legend(fontsize=9)

        # Mark mean and median
        mean_val = np.mean(samples)
        median_val = np.median(samples)
        ax.axvline(mean_val, color='green', linestyle='-', linewidth=1.5, alpha=0.7)
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)

        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{param_name}: μ={mean_val:.3f}, σ={np.std(samples):.3f}',
                    fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    # Hide extra subplots
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved parameter distributions to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_convergence_history(
    convergence_history: List[float],
    title: str = "Calibration Convergence",
    ylabel: str = "Objective Function",
    figsize: Tuple[float, float] = (10, 5),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot calibration convergence history.

    Args:
        convergence_history: List of best objective values over iterations
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    iterations = np.arange(len(convergence_history))
    ax.plot(iterations, convergence_history, 'b-', linewidth=2, alpha=0.8)

    # Mark initial and final
    ax.scatter([0], [convergence_history[0]], color='green', s=100, zorder=5,
              label=f'Initial: {convergence_history[0]:.4f}')
    ax.scatter([len(convergence_history)-1], [convergence_history[-1]],
              color='red', s=100, zorder=5,
              label=f'Final: {convergence_history[-1]:.4f}')

    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved convergence plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_tornado_sensitivity(
    sensitivity_indices: Dict[str, float],
    title: str = "Parameter Sensitivity (Tornado Diagram)",
    figsize: Tuple[float, float] = (10, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot tornado diagram for sensitivity analysis.

    Args:
        sensitivity_indices: Dictionary of sensitivity indices {param: value}
        title: Plot title
        figsize: Figure size
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure
    """
    _check_matplotlib()

    # Sort by absolute sensitivity
    sorted_items = sorted(sensitivity_indices.items(), key=lambda x: abs(x[1]), reverse=True)
    param_names = [item[0] for item in sorted_items]
    si_values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(param_names))
    colors = ['red' if si < 0 else 'blue' for si in si_values]

    ax.barh(y_pos, si_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(param_names, fontsize=11)
    ax.set_xlabel('Sensitivity Index', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved tornado diagram to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_multi_metric_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metric_names: Optional[List[str]] = None,
    title: str = "Multi-Model Performance Comparison",
    figsize: Tuple[float, float] = (12, 6),
    save_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """Plot grouped bar chart comparing multiple models on multiple metrics.

    Args:
        metrics_dict: Dictionary {model_name: {metric_name: value}}
        metric_names: Optional list of metrics to plot (all if None)
        title: Plot title
        figsize: Figure size
        save_path: Path to save
        show: Whether to display

    Returns:
        matplotlib Figure

    Example:
        >>> metrics = {
        ...     'Model A': {'nse': 0.85, 'rmse': 5.2, 'pbias': 3.1},
        ...     'Model B': {'nse': 0.78, 'rmse': 6.5, 'pbias': -2.4}
        ... }
        >>> plot_multi_metric_comparison(metrics)
    """
    _check_matplotlib()

    model_names = list(metrics_dict.keys())

    # Get all metrics if not specified
    if metric_names is None:
        all_metrics = set()
        for model_metrics in metrics_dict.values():
            all_metrics.update(model_metrics.keys())
        metric_names = sorted(all_metrics)

    # Prepare data
    n_models = len(model_names)
    n_metrics = len(metric_names)
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    # Plot bars for each model
    for i, model_name in enumerate(model_names):
        values = [metrics_dict[model_name].get(metric, 0) for metric in metric_names]
        offset = (i - n_models/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=model_name, alpha=0.8)

    ax.set_xlabel('Metrics', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metric_names], rotation=0)
    ax.legend(loc='best', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved multi-metric comparison to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def create_analysis_report_figures(
    observed: np.ndarray,
    simulated: np.ndarray,
    metrics: Dict[str, float],
    param_samples: Optional[Dict[str, np.ndarray]] = None,
    convergence_history: Optional[List[float]] = None,
    uncertainty_percentiles: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Path = Path("results/analysis_report"),
    time_index: Optional[np.ndarray] = None
) -> Dict[str, Path]:
    """Generate a complete set of analysis figures.

    Args:
        observed: Observed time series
        simulated: Simulated time series
        metrics: Performance metrics
        param_samples: Optional parameter posterior samples
        convergence_history: Optional calibration convergence
        uncertainty_percentiles: Optional uncertainty bounds
        output_dir: Output directory
        time_index: Optional time index

    Returns:
        Dictionary of figure paths {name: path}
    """
    _check_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = {}

    # 1. Hydrograph comparison
    fig_path = output_dir / "hydrograph_comparison.png"
    plot_hydrograph_comparison(
        observed, simulated, time_index, metrics,
        save_path=fig_path, show=False
    )
    figure_paths['hydrograph'] = fig_path

    # 2. Scatter plot
    fig_path = output_dir / "scatter_plot.png"
    plot_scatter(observed, simulated, metrics, save_path=fig_path, show=False)
    figure_paths['scatter'] = fig_path

    # 3. Parameter distributions (if available)
    if param_samples:
        fig_path = output_dir / "parameter_distributions.png"
        plot_parameter_distributions(param_samples, save_path=fig_path, show=False)
        figure_paths['param_dist'] = fig_path

    # 4. Convergence history (if available)
    if convergence_history:
        fig_path = output_dir / "convergence_history.png"
        plot_convergence_history(convergence_history, save_path=fig_path, show=False)
        figure_paths['convergence'] = fig_path

    # 5. Uncertainty envelope (if available)
    if uncertainty_percentiles and time_index is not None:
        fig_path = output_dir / "uncertainty_envelope.png"
        mean_sim = simulated  # Use simulated as mean for now
        plot_uncertainty_envelope(
            time_index, observed, mean_sim, uncertainty_percentiles,
            save_path=fig_path, show=False
        )
        figure_paths['uncertainty'] = fig_path

    logger.info(f"Generated {len(figure_paths)} analysis figures in {output_dir}")

    return figure_paths


__all__ = [
    'plot_hydrograph_comparison',
    'plot_scatter',
    'plot_uncertainty_envelope',
    'plot_parameter_distributions',
    'plot_convergence_history',
    'plot_tornado_sensitivity',
    'plot_multi_metric_comparison',
    'create_analysis_report_figures',
]
