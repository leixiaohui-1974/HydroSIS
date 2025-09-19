"""Chart generation helpers for HydroSIS results."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from hydrosis.evaluation import ModelScore, SimulationEvaluator


def _get_matplotlib():
    """Return matplotlib pyplot module when available."""

    try:  # pragma: no cover - optional dependency
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:  # pragma: no cover - graceful fallback
        return None
    return plt


def _ensure_svg_path(path: Path) -> Path:
    """Ensure the output path uses an SVG suffix."""

    if path.suffix.lower() != ".svg":
        return path.with_suffix(".svg")
    return path


def _write_svg(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _color_palette() -> list[str]:
    return [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]


def plot_hydrograph(
    output_path: Path,
    simulations: Mapping[str, Sequence[float]],
    observed: Sequence[float] | None = None,
    title: str | None = None,
    xlabel: str = "Time Step",
    ylabel: str = "Discharge",
) -> Path:
    """Plot simulated and observed hydrographs and save to a file."""

    output_path = Path(output_path)

    # Ensure a deterministic ordering for legends and line plotting so that
    # generated figures remain stable across runs regardless of dictionary
    # insertion order from upstream evaluators.
    sorted_simulations = {
        model_id: series
        for model_id, series in sorted(simulations.items(), key=lambda item: item[0])
    }

    if not sorted_simulations:
        raise ValueError("At least one simulation series is required for plotting.")

    plt = _get_matplotlib()
    if plt is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for model_id, series in sorted_simulations.items():
            ax.plot(range(len(series)), series, label=f"Simulated: {model_id}")

        if observed is not None:
            ax.plot(
                range(len(observed)),
                observed,
                label="Observed",
                linestyle="--",
                linewidth=2,
            )

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

    return _plot_hydrograph_svg(
        output_path,
        sorted_simulations,
        observed,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
    )


def plot_metric_bars(
    output_path: Path,
    scores: Sequence[ModelScore],
    evaluator: SimulationEvaluator,
    metric: str,
    title: str | None = None,
) -> Path:
    """Create a bar chart summarising aggregated metrics for each model."""

    output_path = Path(output_path)

    labels: list[str] = []
    values: list[float] = []
    orientation = evaluator.metric_orientation(metric)

    for score in sorted(scores, key=lambda item: item.model_id):
        if metric not in score.aggregated:
            continue
        value = score.aggregated[metric]
        if orientation == "minabs":
            value = abs(value)
        labels.append(score.model_id)
        values.append(value)

    if not values:
        raise ValueError(f"Metric '{metric}' not available in aggregated scores.")

    plt = _get_matplotlib()
    if plt is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
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

    return _plot_metric_bars_svg(
        output_path,
        labels,
        values,
        metric,
        orientation,
        title=title,
    )


def _plot_hydrograph_svg(
    output_path: Path,
    simulations: Mapping[str, Sequence[float]],
    observed: Sequence[float] | None,
    title: str | None,
    xlabel: str,
    ylabel: str,
) -> Path:
    output_path = _ensure_svg_path(output_path)

    width, height = 800, 450
    margins = {"left": 70, "right": 30, "top": 60, "bottom": 60}
    plot_width = width - margins["left"] - margins["right"]
    plot_height = height - margins["top"] - margins["bottom"]

    ordered_items = sorted(simulations.items(), key=lambda item: item[0])
    ordered_simulations = {model_id: series for model_id, series in ordered_items}

    all_series = list(ordered_simulations.values())
    if observed is not None:
        all_series.append(observed)
    flat_values = [value for series in all_series for value in series]
    if not flat_values:
        flat_values = [0.0]

    y_min = 0.0
    y_max = max(flat_values) if flat_values else 1.0
    if y_max == y_min:
        y_max = y_min + 1.0

    max_length = max(len(series) for series in all_series if series) if all_series else 1
    max_length = max(max_length, 1)
    x_scale = plot_width / (max_length - 1 if max_length > 1 else 1)

    colors = _color_palette()

    def to_point(idx: int, value: float) -> tuple[float, float]:
        x = margins["left"] + idx * x_scale
        scaled = (value - y_min) / (y_max - y_min)
        y = height - margins["bottom"] - scaled * plot_height
        return x, y

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    if title:
        parts.append(
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" fill="#111">{title}</text>'
        )

    parts.append(
        f'<text x="{margins["left"] / 2}" y="{margins["top"] - 20}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 {margins["left"] / 2},{margins["top"] + plot_height / 2})">{ylabel}</text>'
    )
    parts.append(
        f'<text x="{margins["left"] + plot_width / 2}" y="{height - 10}" text-anchor="middle" font-size="12" fill="#333">{xlabel}</text>'
    )

    # Axes
    x0 = margins["left"]
    y0 = height - margins["bottom"]
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{margins["top"]}" stroke="#444" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{margins["left"] + plot_width}" y2="{y0}" stroke="#444" stroke-width="1"/>'
    )

    # Horizontal grid lines and labels
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = height - margins["bottom"] - (value - y_min) / (y_max - y_min) * plot_height
        parts.append(
            f'<line x1="{x0}" y1="{y}" x2="{margins["left"] + plot_width}" y2="{y}" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        parts.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-size="11" fill="#333">{value:.2f}</text>'
        )

    for idx, (model_id, series) in enumerate(ordered_simulations.items()):
        color = colors[idx % len(colors)]
        points = [to_point(i, value) for i, value in enumerate(series)]
        if not points:
            continue
        formatted = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{formatted}"/>'
        )

    if observed is not None:
        points = [to_point(i, value) for i, value in enumerate(observed)]
        formatted = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline fill="none" stroke="#111" stroke-width="2" stroke-dasharray="6,4" points="{formatted}"/>'
        )

    # Legend
    legend_x = margins["left"] + 10
    legend_y = margins["top"] - 25
    legend_spacing = 90
    for idx, model_id in enumerate(ordered_simulations.keys()):
        color = colors[idx % len(colors)]
        x = legend_x + idx * legend_spacing
        parts.append(
            f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />'
        )
        parts.append(
            f'<text x="{x + 18}" y="{legend_y}" font-size="12" fill="#333">{model_id}</text>'
        )

    if observed is not None:
        obs_x = legend_x + len(ordered_simulations) * legend_spacing
        parts.append(
            f'<rect x="{obs_x}" y="{legend_y - 10}" width="12" height="12" fill="none" stroke="#111" stroke-width="2" stroke-dasharray="6,4" />'
        )
        parts.append(
            f'<text x="{obs_x + 18}" y="{legend_y}" font-size="12" fill="#333">Observed</text>'
        )

    parts.append("</svg>")
    return _write_svg(output_path, "".join(parts))


def _plot_metric_bars_svg(
    output_path: Path,
    labels: Sequence[str],
    values: Sequence[float],
    metric: str,
    orientation: str,
    title: str | None,
) -> Path:
    output_path = _ensure_svg_path(output_path)

    width, height = 800, 450
    margins = {"left": 70, "right": 30, "top": 60, "bottom": 70}
    plot_width = width - margins["left"] - margins["right"]
    plot_height = height - margins["top"] - margins["bottom"]

    max_value = max(values)
    if max_value == 0:
        max_value = 1.0

    best_index = max(range(len(values)), key=values.__getitem__)
    if orientation in {"min", "minabs"}:
        best_index = min(range(len(values)), key=values.__getitem__)

    band_width = plot_width / max(len(values), 1)
    bar_width = band_width * 0.6

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    if title:
        parts.append(
            f'<text x="{width / 2}" y="30" text-anchor="middle" font-size="20" fill="#111">{title}</text>'
        )

    parts.append(
        f'<text x="{margins["left"] / 2}" y="{margins["top"] - 20}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-90 {margins["left"] / 2},{margins["top"] + plot_height / 2})">{metric.upper()}</text>'
    )

    x0 = margins["left"]
    y0 = height - margins["bottom"]
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{margins["top"]}" stroke="#444" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{x0}" y1="{y0}" x2="{margins["left"] + plot_width}" y2="{y0}" stroke="#444" stroke-width="1"/>'
    )

    for i in range(6):
        value = max_value * i / 5
        y = y0 - value / max_value * plot_height
        parts.append(
            f'<line x1="{x0}" y1="{y}" x2="{margins["left"] + plot_width}" y2="{y}" stroke="#e0e0e0" stroke-width="1" stroke-dasharray="4,4"/>'
        )
        parts.append(
            f'<text x="{x0 - 10}" y="{y + 4}" text-anchor="end" font-size="11" fill="#333">{value:.2f}</text>'
        )

    for idx, (label, value) in enumerate(zip(labels, values)):
        x = x0 + idx * band_width + (band_width - bar_width) / 2
        height_value = value / max_value * plot_height
        y = y0 - height_value
        color = "#2ca02c" if idx == best_index else "#b0c4de"
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{height_value:.2f}" fill="{color}" />'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-size="11" fill="#333">{value:.2f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width / 2:.2f}" y="{y0 + 18:.2f}" text-anchor="middle" font-size="12" fill="#333">{label}</text>'
        )

    parts.append("</svg>")
    return _write_svg(output_path, "".join(parts))


__all__ = ["plot_hydrograph", "plot_metric_bars"]
