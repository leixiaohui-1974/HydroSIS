"""Step 07: Thiessen Polygon Weights

Compute areal weights for rain gauges.
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

def run_step07_thiessen_weights(config_path: Path | str) -> Dict[str, Path]:
    """Compute Thiessen polygons and precipitation weights for parameter subbasins."""

    context = load_project_context(config_path)
    step_index = 7
    step_name = "thiessen_weights"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step07_thiessen_weights.log"
    logger = configure_logger("step07", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 07 – Thiessen weight computation started.")

    project_cfg = context.config.setdefault("project", {})
    gauge_cfg = project_cfg.setdefault("rain_gauge", {})
    rainfall_cfg = project_cfg.setdefault("rainfall", {})
    delineation_cfg = context.config.get("delineation") or {}
    parameter_dir_entry = delineation_cfg.get("parameter_directory")
    if not parameter_dir_entry:
        raise PipelineConfigurationError(
            "Parameter directory not configured. Run Step 03 before Step 07."
        )

    parameter_dir = resolve_input_path(context, parameter_dir_entry)
    subbasin_geojson = parameter_dir / "parameter_subbasins.geojson"
    subbasin_geometries = load_subbasin_geometries(subbasin_geojson)

    import json
    import pandas as pd
    from shapely.geometry import shape, mapping

    thiessen_src = gauge_cfg.get("thiessen_geojson")
    layout_src = gauge_cfg.get("layout_geojson")
    if not thiessen_src or not layout_src:
        raise PipelineConfigurationError(
            "Rain gauge layout not available. Run Step 05 before Step 07."
        )
    thiessen_path = resolve_input_path(context, thiessen_src)
    layout_path = resolve_input_path(context, layout_src)
    locations_data = json.loads(layout_path.read_text(encoding="utf-8"))
    polygon_data = json.loads(thiessen_path.read_text(encoding="utf-8"))

    station_positions: Dict[str, "BaseGeometry"] = {}
    for feature in locations_data.get("features", []):
        props = feature.get("properties", {}) or {}
        station_id = str(props.get("station_id") or props.get("id") or "").strip()
        if not station_id:
            continue
        station_positions[station_id] = shape(feature.get("geometry"))

    thiessen_polygons: Dict[str, "BaseGeometry"] = {}
    for feature in polygon_data.get("features", []):
        props = feature.get("properties", {}) or {}
        station_id = str(props.get("station_id") or props.get("id") or "").strip()
        if not station_id:
            continue
        thiessen_polygons[station_id] = shape(feature.get("geometry"))

    if not thiessen_polygons:
        raise ValueError("No Thiessen polygons found in the provided layout.")

    from hydrosis.precipitation import compute_subbasin_station_weights

    weights = compute_subbasin_station_weights(subbasin_geometries, thiessen_polygons)

    weights_json_path = step_dir / "rain_gauge_weights.json"
    weights_json_path.write_text(json.dumps(weights, indent=2), encoding="utf-8")

    weights_rows: list[tuple[str, str, float]] = []
    for sub_id, station_map in weights.items():
        for station_id, value in sorted(station_map.items()):
            weights_rows.append((sub_id, station_id, float(value)))
    weights_df = pd.DataFrame(
        weights_rows, columns=["subbasin_id", "station_id", "weight"]
    )
    weights_table_path = step_dir / "weights_table.csv"
    weights_df.to_csv(weights_table_path, index=False)

    polygons_out_path = step_dir / "rain_gauge_thiessen_polygons.geojson"
    polygons_out_path.write_text(
        json.dumps(
            {"type": "FeatureCollection", "features": polygon_data.get("features", [])},
            indent=2,
        ),
        encoding="utf-8",
    )

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Polygon as MplPolygon

    color_map = plt.get_cmap("tab20")
    station_ids = sorted(thiessen_polygons.keys())
    station_to_color = {sid: color_map(index % 20) for index, sid in enumerate(station_ids)}

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    patches = []
    colors = []
    xs_all: list[float] = []
    ys_all: list[float] = []

    for station_id, polygon in thiessen_polygons.items():
        if polygon.geom_type == "Polygon":
            coords = np.asarray(polygon.exterior.coords)
            patches.append(MplPolygon(coords, closed=True))
            colors.append(station_to_color[station_id])
            xs_all.extend(coords[:, 0])
            ys_all.extend(coords[:, 1])
        elif polygon.geom_type == "MultiPolygon":
            for part in polygon.geoms:
                coords = np.asarray(part.exterior.coords)
                patches.append(MplPolygon(coords, closed=True))
                colors.append(station_to_color[station_id])
                xs_all.extend(coords[:, 0])
                ys_all.extend(coords[:, 1])

    if patches:
        collection = PatchCollection(patches, facecolors=colors, edgecolor="black", linewidth=0.6, alpha=0.5)
        ax.add_collection(collection)

    for station_id, point in station_positions.items():
        ax.scatter(
            point.x,
            point.y,
            c=[station_to_color.get(station_id, "red")],
            s=40,
            edgecolors="white",
            linewidths=0.8,
            label=station_id,
        )
        ax.text(point.x, point.y, station_id, fontsize=8, ha="left", va="bottom")

    if xs_all and ys_all:
        ax.set_xlim(min(xs_all) - 1000, max(xs_all) + 1000)
        ax.set_ylim(min(ys_all) - 1000, max(ys_all) + 1000)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Thiessen Polygons by Station")
    ax.axis("off")
    plt.tight_layout()
    thiessen_map_path = step_dir / "thiessen_map.png"
    plt.savefig(thiessen_map_path, dpi=220)
    plt.close()

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 07 – Thiessen Weights")
    builder.add_paragraph(
        "根据雨量站布设与子流域形状计算泰森多边形及站点权重，为后续面雨量插值提供输入。"
    )
    builder.add_heading("统计概览", level=2)
    top_rows = weights_df.sort_values("weight", ascending=False).head(10)
    builder.add_table(
        TableData(
            headers=["子流域", "雨量站", "权重"],
            rows=[
                [row["subbasin_id"], row["station_id"], f"{row['weight']:.3f}"]
                for _, row in top_rows.iterrows()
            ],
        )
    )
    builder.add_paragraph(f"权重计算时间：{timestamp.isoformat()}")
    builder.write(report_path)

    rainfall_cfg["weights_json"] = context.to_relative(weights_json_path)
    rainfall_cfg["thiessen_polygons"] = context.to_relative(polygons_out_path)
    project_cfg["last_step07_run"] = timestamp.isoformat()
    dump_project_config(context)
    logger.info("Step 07 – Thiessen weights completed successfully.")

    return {
        "weights_json": weights_json_path,
        "weights_table": weights_table_path,
        "thiessen_polygons": polygons_out_path,
        "thiessen_map": thiessen_map_path,
        "report": report_path,
        "log": log_path,
    }


def run_step08_areal_precipitation(config_path: Path | str) -> Dict[str, Path]:
