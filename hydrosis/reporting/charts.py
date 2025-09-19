"""Chart generation helpers for HydroSIS results."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from hydrosis.evaluation import ModelScore, SimulationEvaluator


def _require_matplotlib():
    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - graceful fallback
        raise RuntimeError(
            "matplotlib is required for chart generation but is not installed."
        ) from exc
    return plt


def plot_hydrograph(
    output_path: Path,
    simulations: Mapping[str, Sequence[float]],
    observed: Sequence[float] | None = None,
    title: str | None = None,
    xlabel: str = "Time Step",
    ylabel: str = "Discharge",
) -> Path:
    """Plot simulated and observed hydrographs and save to a file."""

    plt = _require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    if not simulations:
        raise ValueError("At least one simulation series is required for plotting.")

    for model_id, series in simulations.items():
        ax.plot(range(len(series)), series, label=f"Simulated: {model_id}")

    if observed is not None:
        ax.plot(range(len(observed)), observed, label="Observed", linestyle="--", linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=":", linewidth=0.5)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_metric_bars(
    output_path: Path,
    scores: Sequence[ModelScore],
    evaluator: SimulationEvaluator,
    metric: str,
    title: str | None = None,
) -> Path:
    """Create a bar chart summarising aggregated metrics for each model."""

    plt = _require_matplotlib()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    values: list[float] = []
    orientation = evaluator.metric_orientation(metric)

    for score in scores:
        if metric not in score.aggregated:
            continue
        value = score.aggregated[metric]
        if orientation == "minabs":
            value = abs(value)
        labels.append(score.model_id)
        values.append(value)

    if not values:
        raise ValueError(f"Metric '{metric}' not available in aggregated scores.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values)

    best_index = max(range(len(values)), key=values.__getitem__)
    if orientation in {"min", "minabs"}:
        best_index = min(range(len(values)), key=values.__getitem__)
    for bar in bars:
        bar.set_color("#b0c4de")
    bars[best_index].set_color("#2ca02c")

    ax.set_ylabel(metric.upper())
    if title:
        ax.set_title(title)
    ax.grid(axis="y", linestyle=":", linewidth=0.5)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["plot_hydrograph", "plot_metric_bars"]
