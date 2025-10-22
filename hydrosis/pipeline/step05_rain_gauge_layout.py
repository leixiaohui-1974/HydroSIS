"""Step 05: Rain Gauge Layout

Design spatial distribution of rain gauges.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple
from types import SimpleNamespace

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import yaml
except ImportError:
    yaml = None

from .core import (
    ProjectContext,
    PipelineConfigurationError,
    load_project_context,
    dump_project_config,
    step_directory,
    configure_logger,
    write_csv,
    build_report_path,
    resolve_input_path,
    load_subbasin_geometries,
    reset_runoff_initial_states,
    load_base_precipitation_series,
    compute_basic_stats,
)
from hydrosis.reporting.markdown import MarkdownReportBuilder, TableData


# Step-specific imports
try:
    from shapely.geometry import shape, Point
    from shapely.ops import unary_union
except ImportError:
    shape = Point = unary_union = None

def run_step05_rain_gauge_layout(config_path: Path | str) -> Dict[str, Path]:
    """Generate spatial layout for rain gauges based on project settings."""

    context = load_project_context(config_path)
    step_index = 5
    step_name = "rain_gauge_layout"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step05_rain_gauge_layout.log"
    logger = configure_logger("step05", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 05 – Rain gauge layout started.")

    project_cfg = context.config.setdefault("project", {})
    gauge_cfg = project_cfg.setdefault("rain_gauge", {})
    rainfall_cfg = project_cfg.setdefault("rainfall", {})
    delineation_cfg = context.config.get("delineation") or {}
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not parameter_dir_entry:
        raise PipelineConfigurationError(
            "Parameter directory not configured. Run Step 03 before Step 05."
        )

    parameter_dir = resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = load_subbasin_geometries(subbasin_geojson)

    default_base_series = rainfall_cfg.get(
        "base_series_path", "results/upper_truckee_channel_demo/storm_forcing.csv"
    )
    rainfall_cfg["base_series_path"] = default_base_series
    base_series_path = resolve_input_path(context, default_base_series)
    base_column = rainfall_cfg.get("base_column")

    import json
    import numpy as np
    import pandas as pd
    from shapely.geometry import mapping
    from shapely.ops import unary_union

    base_series = load_base_precipitation_series(base_series_path, column=base_column)
    if base_column is None:
        rainfall_cfg["base_column"] = base_series.name

    station_count = int(gauge_cfg.get("station_count", 10))
    heterogeneity = float(gauge_cfg.get("heterogeneity", 0.4))
    min_burst_events = int(gauge_cfg.get("min_burst_events", 1))
    max_burst_events = int(gauge_cfg.get("max_burst_events", 3))
    if max_burst_events < min_burst_events:
        max_burst_events = min_burst_events
    gauge_cfg["station_count"] = station_count
    gauge_cfg["heterogeneity"] = heterogeneity
    gauge_cfg["min_burst_events"] = min_burst_events
    gauge_cfg["max_burst_events"] = max_burst_events

    seed = gauge_cfg.get("seed")
    if seed is None:
        seed = 42
        gauge_cfg["seed"] = seed
    else:
        seed = int(seed)

    logger.info(
        "Generating rain gauge layout (stations=%d, seed=%d).", station_count, seed
    )
    rain_inputs = _generate_rain_gauge_inputs(
        base_series,
        subbasin_geometries,
        station_count=station_count,
        seed=seed,
        heterogeneity=heterogeneity,
        min_burst_events=min_burst_events,
        max_burst_events=max_burst_events,
    )

    locations_geojson = step_dir / "rain_gauge_locations.geojson"
    layout_csv = step_dir / "rain_gauge_layout.csv"
    coverage_csv = step_dir / "station_coverage.csv"
    polygons_geojson = step_dir / "rain_gauge_thiessen_polygons.geojson"
    layout_map_path = step_dir / "rain_gauge_layout_map.png"

    features = []
    for station_id, point in rain_inputs.station_positions.items():
        features.append(
            {
                "type": "Feature",
                "geometry": mapping(point),
                "properties": {"station_id": station_id},
            }
        )
    locations_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2),
        encoding="utf-8",
    )

    import math
    import numpy as np

    coverage_rows: list[tuple[str, float, float, float, float]] = []
    for station_id, polygon in rain_inputs.thiessen_polygons.items():
        area_m2 = float(polygon.area)
        area_km2 = area_m2 / 1_000_000.0
        radius_m = math.sqrt(area_m2 / math.pi) if area_m2 > 0 else 0.0
        coverage_rows.append(
            (
                station_id,
                rain_inputs.station_positions[station_id].x,
                rain_inputs.station_positions[station_id].y,
                area_km2,
                radius_m,
            )
        )

    layout_df = pd.DataFrame(
        coverage_rows,
        columns=["station_id", "x", "y", "coverage_area_km2", "coverage_radius_m"],
    )
    layout_df.to_csv(layout_csv, index=False)

    layout_df[["station_id", "coverage_radius_m"]].to_csv(coverage_csv, index=False)

    polygon_features = []
    for station_id, polygon in rain_inputs.thiessen_polygons.items():
        polygon_features.append(
            {
                "type": "Feature",
                "geometry": mapping(polygon),
                "properties": {"station_id": station_id},
            }
        )
    polygons_geojson.write_text(
        json.dumps({"type": "FeatureCollection", "features": polygon_features}, indent=2),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon as MplPolygon

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for geom in subbasin_geometries.values():
        if geom.geom_type == "Polygon":
            patch = MplPolygon(np.asarray(geom.exterior.coords), facecolor="#e0f3ff", edgecolor="#4a90e2", alpha=0.4, linewidth=0.6)
            ax.add_patch(patch)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                patch = MplPolygon(np.asarray(part.exterior.coords), facecolor="#e0f3ff", edgecolor="#4a90e2", alpha=0.4, linewidth=0.6)
                ax.add_patch(patch)

    for station_id, row in layout_df.iterrows():
        circle = Circle(
            (row["x"], row["y"]),
            radius=row["coverage_radius_m"],
            facecolor="none",
            edgecolor="orange",
            linewidth=1.0,
            alpha=0.7,
        )
        ax.add_patch(circle)
        ax.scatter(row["x"], row["y"], c="crimson", s=40, edgecolors="white", linewidths=0.8)
        ax.text(row["x"], row["y"], row["station_id"], fontsize=8, color="black", ha="left", va="bottom")

    try:
        extent_geom = unary_union(list(subbasin_geometries.values()))
        minx, miny, maxx, maxy = extent_geom.bounds
    except Exception:  # pragma: no cover - fallback when geometry union fails
        all_points = np.array([[pt.x, pt.y] for pt in rain_inputs.station_positions.values()])
        if all_points.size:
            minx, miny = all_points[:, 0].min(), all_points[:, 1].min()
            maxx, maxy = all_points[:, 0].max(), all_points[:, 1].max()
        else:
            minx = miny = maxx = maxy = 0.0

    pad = max((maxx - minx), (maxy - miny)) * 0.05 + 1000.0
    ax.set_xlim(minx - pad, maxx + pad)
    ax.set_ylim(miny - pad, maxy + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Rain Gauge Layout")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(layout_map_path, dpi=220)
    plt.close()

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 05 – Rain Gauge Layout")
    builder.add_paragraph(
        "根据基准降雨序列与子流域形状生成雨量站布局，并输出站点覆盖半径统计与示意图。"
    )
    builder.add_heading("配置参数", level=2)
    builder.add_list(
        [
            f"站点数：{station_count}",
            f"随机种子：{seed}",
            f"空间异质性：{heterogeneity:.2f}",
            f"突发事件范围：{min_burst_events}–{max_burst_events}",
        ]
    )
    builder.add_heading("覆盖统计", level=2)
    summary_table = TableData(
        headers=["站点", "覆盖面积 (km²)", "等效半径 (m)"],
        rows=[
            [
                row["station_id"],
                f"{row['coverage_area_km2']:.2f}",
                f"{row['coverage_radius_m']:.1f}",
            ]
            for _, row in layout_df.head(10).iterrows()
        ],
    )
    builder.add_table(summary_table)
    builder.add_paragraph(f"布局完成时间：{timestamp.isoformat()}")
    builder.write(report_path)

    gauge_cfg["layout_geojson"] = context.to_relative(locations_geojson)
    gauge_cfg["layout_csv"] = context.to_relative(layout_csv)
    gauge_cfg["thiessen_geojson"] = context.to_relative(polygons_geojson)
    project_cfg["last_step05_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 05 – Rain gauge layout completed successfully.")

    return {
        "locations_geojson": locations_geojson,
        "layout_csv": layout_csv,
        "coverage_csv": coverage_csv,
        "thiessen_geojson": polygons_geojson,
        "layout_map": layout_map_path,
        "report": report_path,
        "log": log_path,
    }


def run_step06_rain_sequence(config_path: Path | str) -> Dict[str, Path]:
