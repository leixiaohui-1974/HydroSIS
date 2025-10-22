# -*- coding: utf-8 -*-
"""Chart generation utilities for HydroSIS visualisations.

The helpers in this module provide both Plotly (interactive) and Matplotlib
fallbacks so callers can obtain hydrographs, scenario comparisons, metrics
dashboards, and simple choropleth maps regardless of whether the optional
Plotly dependency is installed.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import json

try:  # pragma: no cover - matplotlib is optional
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    import numpy as np

    HAS_MPL = True
except ImportError:  # pragma: no cover - Matplotlib is optional
    HAS_MPL = False

try:  # pragma: no cover - Plotly is optional
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:  # pragma: no cover
    HAS_PLOTLY = False

ChartReturn = Union[str, bytes]


def create_hydrograph_chart(
    data: Mapping[str, Sequence[float]],
    *,
    timestamps: Optional[Sequence[float]] = None,
    title: str = "Hydrograph",
    x_label: str = "Time",
    y_label: str = "Discharge (m³/s)",
    colors: Optional[Sequence[str]] = None,
    use_plotly: bool = True,
    width: int = 800,
    height: int = 500,
) -> ChartReturn:
    """Render a hydrograph for one or more time series."""

    if use_plotly and HAS_PLOTLY:
        return _create_plotly_hydrograph(
            data=data,
            timestamps=timestamps,
            title=title,
            x_label=x_label,
            y_label=y_label,
            colors=colors,
            width=width,
            height=height,
        )
    if HAS_MPL:
        return _create_mpl_hydrograph(
            data=data,
            timestamps=timestamps,
            title=title,
            x_label=x_label,
            y_label=y_label,
            colors=colors,
        )
    raise ImportError("Neither Plotly nor Matplotlib is available for chart generation")


def _create_plotly_hydrograph(
    *,
    data: Mapping[str, Sequence[float]],
    timestamps: Optional[Sequence[float]],
    title: str,
    x_label: str,
    y_label: str,
    colors: Optional[Sequence[str]],
    width: int,
    height: int,
) -> str:
    """Use Plotly to produce an interactive hydrograph."""

    figure = go.Figure()
    default_colors = px.colors.qualitative.Set1

    if timestamps is None:
        timestamps = list(range(max(len(series) for series in data.values())))

    for index, (name, series) in enumerate(data.items()):
        colour = (
            colors[index]
            if colors and index < len(colors)
            else default_colors[index % len(default_colors)]
        )
        figure.add_trace(
            go.Scatter(
                x=list(timestamps[: len(series)]),
                y=list(series),
                mode="lines",
                name=name,
                line={"color": colour, "width": 2},
            )
        )

    figure.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        width=width,
        height=height,
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return pio.to_html(figure, include_plotlyjs="cdn", div_id="hydrograph-chart")


def _create_mpl_hydrograph(
    *,
    data: Mapping[str, Sequence[float]],
    timestamps: Optional[Sequence[float]],
    title: str,
    x_label: str,
    y_label: str,
    colors: Optional[Sequence[str]],
) -> bytes:
    """Use Matplotlib to produce a static hydrograph."""

    if not HAS_MPL:  # sanity guard
        raise RuntimeError("Matplotlib is not available")

    fig, ax = plt.subplots(figsize=(10, 6))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    if timestamps is None:
        x_axis = list(range(max(len(series) for series in data.values())))
    else:
        x_axis = list(timestamps)

    for index, (name, series) in enumerate(data.items()):
        colour = (
            colors[index]
            if colors and index < len(colors)
            else default_colors[index % len(default_colors)]
        )
        ax.plot(x_axis[: len(series)], series, label=name, color=colour, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_comparison_chart(
    baseline: Mapping[str, Sequence[float]],
    scenarios: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    subbasin_id: Optional[str] = None,
    title: str = "Scenario comparison",
    x_label: str = "Time",
    y_label: str = "Discharge (m³/s)",
    use_plotly: bool = True,
    width: int = 800,
    height: int = 500,
) -> ChartReturn:
    """Compare baseline and scenario time series for a subbasin or the full network."""

    if subbasin_id:
        baseline_series = list(baseline.get(subbasin_id, []))
    else:
        horizon = max(len(series) for series in baseline.values())
        baseline_series = [
            sum(series[i] for series in baseline.values() if i < len(series))
            for i in range(horizon)
        ]

    plot_data: Dict[str, Sequence[float]] = {"Baseline": baseline_series}

    if subbasin_id:
        for scenario_name, series_map in scenarios.items():
            if subbasin_id in series_map:
                plot_data[scenario_name] = series_map[subbasin_id]
    else:
        for scenario_name, series_map in scenarios.items():
            horizon = max(len(series) for series in series_map.values())
            plot_data[scenario_name] = [
                sum(series[i] for series in series_map.values() if i < len(series))
                for i in range(horizon)
            ]

    return create_hydrograph_chart(
        plot_data,
        title=title,
        x_label=x_label,
        y_label=y_label,
        use_plotly=use_plotly,
        width=width,
        height=height,
    )


def create_metrics_chart(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    title: str = "Model metrics",
    use_plotly: bool = True,
    width: int = 800,
    height: int = 500,
) -> ChartReturn:
    """Render grouped metric bars for each model."""

    if use_plotly and HAS_PLOTLY:
        return _create_plotly_metrics(metrics, title=title, width=width, height=height)
    if HAS_MPL:
        return _create_mpl_metrics(metrics, title=title)
    raise ImportError("Neither Plotly nor Matplotlib is available for chart generation")


def _create_plotly_metrics(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    title: str,
    width: int,
    height: int,
) -> str:
    metric_names = list(next(iter(metrics.values())).keys())
    model_names = list(metrics.keys())

    fig = make_subplots(
        rows=1,
        cols=len(metric_names),
        subplot_titles=metric_names,
        shared_yaxes=True,
    )

    for idx, metric_name in enumerate(metric_names):
        values = [metrics[model][metric_name] for model in model_names]
        fig.add_trace(
            go.Bar(x=model_names, y=values, name=metric_name, showlegend=False),
            row=1,
            col=idx + 1,
        )

    fig.update_layout(title_text=title, width=width, height=height, bargap=0.3)
    return pio.to_html(fig, include_plotlyjs="cdn", div_id="metrics-chart")


def _create_mpl_metrics(metrics: Mapping[str, Mapping[str, float]], *, title: str) -> bytes:
    if not HAS_MPL:
        raise RuntimeError("Matplotlib is not available")

    metric_names = list(next(iter(metrics.values())).keys())
    model_names = list(metrics.keys())

    fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metric_names):
        values = [metrics[model][metric_name] for model in model_names]
        ax.bar(model_names, values)
        ax.set_title(metric_name)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def create_map_visualization(
    geojson_data: Dict[str, Any],
    *,
    values: Optional[Mapping[str, float]] = None,
    color_property: str = "value",
    title: str = "Catchment map",
    use_plotly: bool = True,
    width: int = 800,
    height: int = 600,
) -> str:
    """Render a simple choropleth map using Plotly."""

    if not (use_plotly and HAS_PLOTLY):
        raise ImportError("Plotly is required to generate the map visualisation")

    features = geojson_data.setdefault("features", [])
    locations: List[str] = []
    colour_values: List[float] = []

    for feature in features:
        properties = feature.setdefault("properties", {})
        feature_id = feature.get("id") or properties.get("id")
        if feature_id is None:
            continue
        properties["id"] = feature_id
        locations.append(str(feature_id))

        if values and feature_id in values:
            properties[color_property] = float(values[feature_id])
        colour_values.append(float(properties.get(color_property, 0.0)))

    figure = go.Figure(
        go.Choroplethmap(
            geojson=geojson_data,
            locations=locations,
            z=colour_values,
            featureidkey="properties.id",
            colorscale="Viridis",
            marker_line_width=0.5,
            marker_line_color="#ffffff",
            hoverinfo="location+z",
        )
    )

    figure.update_layout(title=title, width=width, height=height, margin=dict(l=0, r=0, t=30, b=0))
    figure.update_geos(fitbounds="locations", visible=False)
    return pio.to_html(figure, include_plotlyjs="cdn", div_id="map-visualization")


def create_dashboard_html(
    hydrograph_html: str,
    comparison_html: str,
    metrics_html: str,
    *,
    map_html: Optional[str] = None,
    title: str = "HydroSIS dashboard",
) -> str:
    """Combine multiple charts into a lightweight HTML dashboard."""

    map_block = (
        f'<div class="chart-container full-width"><h2>Catchment map</h2>{map_html}</div>'
        if map_html
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    body {{
      font-family: "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 20px;
      background-color: #f5f7fb;
      color: #1f2933;
    }}
    .dashboard {{
      max-width: 1400px;
      margin: 0 auto;
    }}
    .header {{
      text-align: center;
      margin-bottom: 30px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
      gap: 20px;
    }}
    .chart-container {{
      background-color: #ffffff;
      border-radius: 8px;
      box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
      padding: 20px;
    }}
    .chart-container h2 {{
      margin-top: 0;
      color: #1b5e20;
    }}
    .full-width {{
      grid-column: 1 / -1;
    }}
  </style>
</head>
<body>
  <div class="dashboard">
    <div class="header">
      <h1>{title}</h1>
    </div>
    <div class="grid">
      <div class="chart-container">
        <h2>Hydrograph</h2>
        {hydrograph_html}
      </div>
      <div class="chart-container">
        <h2>Scenario comparison</h2>
        {comparison_html}
      </div>
      <div class="chart-container">
        <h2>Model metrics</h2>
        {metrics_html}
      </div>
      {map_block}
    </div>
  </div>
</body>
</html>
"""


__all__ = [
    "create_hydrograph_chart",
    "create_comparison_chart",
    "create_metrics_chart",
    "create_map_visualization",
    "create_dashboard_html",
]
