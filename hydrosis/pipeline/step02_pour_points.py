"""Step 02: Pour Point Extraction

Generate or load pour points for watershed delineation.
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
    import rasterio
    from hydrosis.delineation import utils as dutils
except ImportError:
    rasterio = None
    dutils = None

def run_step02_pour_points(config_path: Path | str) -> Dict[str, Path]:
    """Derive channel-aware pour points and accompanying diagnostics."""

    context = load_project_context(config_path)
    step_index = 2
    step_name = "pour_points"
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step02_pour_points.log"
    logger = configure_logger("step02", log_path)
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    logger.info("Step 02 – Pour point extraction started.")
    delineation_cfg = context.config.get("delineation")
    if not isinstance(delineation_cfg, MutableMapping):
        raise PipelineConfigurationError("Configuration is missing the 'delineation' section.")

    flow_acc_entry = delineation_cfg.get("flow_accumulation_path")
    flow_dir_entry = delineation_cfg.get("flow_direction_path")
    dem_entry = delineation_cfg.get("dem_path")
    if not flow_acc_entry or not flow_dir_entry:
        raise PipelineConfigurationError(
            "Run Step 01 first to populate flow accumulation and direction paths."
        )
    if not dem_entry:
        raise PipelineConfigurationError("The delineation.dem_path entry must be configured.")

    flow_acc_path = resolve_input_path(context, str(flow_acc_entry))
    flow_dir_path = resolve_input_path(context, str(flow_dir_entry))
    dem_path = resolve_input_path(context, str(dem_entry))

    if not flow_acc_path.exists() or not flow_dir_path.exists():
        raise FileNotFoundError(
            "Flow accumulation or direction rasters not found. Ensure Step 01 completed successfully."
        )

    project_cfg = context.config.get("project", {})
    pour_cfg = project_cfg.get("pour_points", {})
    count = int(pour_cfg.get("count", 8))
    min_distance = int(pour_cfg.get("min_distance_cells", 30))
    main_fraction = float(pour_cfg.get("main_stem_fraction", 0.5))
    main_fraction = max(0.0, min(1.0, main_fraction))
    area_tolerance = float(pour_cfg.get("area_balance_tolerance", 0.35))
    accumulation_threshold = float(delineation_cfg.get("accumulation_threshold", 0.0))

    logger.info(
        "Generating %d pour points (min_distance=%d, main_fraction=%.2f, area_tol=%.2f).",
        count,
        min_distance,
        main_fraction,
        area_tolerance,
    )

    try:
        import numpy as np
        import rasterio

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource

        from hydrosis.delineation import utils as dutils
    except ImportError as exc:  # pragma: no cover - depends on optional libs
        logger.error("Required dependency missing during pour point generation: %s", exc)
        raise

    geojson_path = step_dir / "pour_points.geojson"
    accumulation_fig_path = step_dir / "accumulation_overview.png"
    source_entry = pour_cfg.get("source_geojson")
    use_existing = False
    pour_points: List[Any]

    # Load transform early in case we need it for coordinate conversion
    with rasterio.open(flow_acc_path) as acc_ds:
        transform = acc_ds.transform

    if source_entry:
        source_path = resolve_input_path(context, str(source_entry))
        if source_path.exists():
            logger.info("Using existing pour point dataset from %s.", source_path)
            geojson_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            data = json.loads(source_path.read_text(encoding="utf-8"))
            pour_points = []
            for feature in data.get("features", []):
                props = feature.get("properties", {}) or {}
                coord = feature.get("geometry", {}).get("coordinates", [math.nan, math.nan])
                pid = str(
                    props.get("id")
                    or props.get("pour_id")
                    or props.get("name")
                    or f"P{len(pour_points)+1}"
                )
                row = props.get("row")
                col = props.get("col")
                if row is None or col is None:
                    x, y = coord
                    col_val, row_val = ~transform * (x, y)
                    row = int(round(row_val))
                    col = int(round(col_val))
                accumulation_value = props.get("accumulation")
                if accumulation_value is None:
                    accumulation_value = 0.0
                pour_points.append(
                    SimpleNamespace(
                        id=pid,
                        x=float(coord[0]),
                        y=float(coord[1]),
                        row=int(row),
                        col=int(col),
                        accumulation=float(accumulation_value),
                    )
                )
            count = len(pour_points)
            use_existing = True
        else:
            logger.warning(
                "Configured pour_points.source_geojson %s not found; generating automatically.",
                source_path,
            )

    if not use_existing:
        pour_points = dutils.generate_tree_pour_points(
            flow_acc_path,
            flow_dir_path,
            count=count,
            accumulation_threshold=accumulation_threshold,
            min_distance_cells=min_distance,
            main_stem_fraction=main_fraction,
            area_balance_tolerance=area_tolerance,
            output_geojson=geojson_path,
            accumulation_plot=accumulation_fig_path,
        )

    if not pour_points:
        raise RuntimeError("Pour point generation produced no outputs.")

    # Read accumulation data (transform was already loaded earlier)
    with rasterio.open(flow_acc_path) as acc_ds:
        accumulation = acc_ds.read(1)
    with rasterio.open(dem_path) as dem_ds:
        dem_array = dem_ds.read(1, masked=True).filled(np.nan)
        dem_transform = dem_ds.transform

    cell_area_km2 = abs(transform.a * transform.e) / 1_000_000.0
    accumulation = np.where(np.isfinite(accumulation), accumulation, 0.0)
    log_accum = np.log1p(accumulation)

    areas = [(pp.accumulation + 1.0) * cell_area_km2 for pp in pour_points]
    ranked_indices = sorted(range(len(pour_points)), key=lambda idx: areas[idx], reverse=True)
    main_count = max(1, int(round(len(pour_points) * main_fraction))) if pour_points else 0
    main_ids = {pour_points[idx].id for idx in ranked_indices[:main_count]}

    table_rows = []
    for pp, area in zip(pour_points, areas):
        role = "main_stem" if pp.id in main_ids else "tributary"
        table_rows.append(
            (
                pp.id,
                pp.x,
                pp.y,
                pp.row,
                pp.col,
                pp.accumulation,
                area,
                role,
            )
        )

    table_path = step_dir / "pour_points_table.csv"
    csv_rows = [
        (
            row_id,
            f"{x:.3f}",
            f"{y:.3f}",
            row,
            col,
            f"{acc:.1f}",
            f"{area:.4f}",
            role,
        )
        for row_id, x, y, row, col, acc, area, role in table_rows
    ]
    write_csv(
        table_path,
        ("id", "x", "y", "row", "col", "accumulation", "estimated_area_km2", "channel_role"),
        csv_rows,
    )

    logger.info("Rendering pour point overview map.")
    ls = LightSource(azdeg=315, altdeg=45)
    dem_valid = np.where(np.isfinite(dem_array), dem_array, np.nan)
    if np.isfinite(dem_valid).sum() == 0:
        hillshade_input = np.zeros_like(dem_array)
    else:
        fill_value = float(np.nanmedian(dem_valid))
        hillshade_input = np.where(np.isfinite(dem_valid), dem_valid, fill_value)
    hillshade = ls.hillshade(
        hillshade_input,
        vert_exag=1.0,
        dx=abs(dem_transform.a),
        dy=abs(dem_transform.e),
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(hillshade, cmap="gray", origin="upper")
    plt.imshow(log_accum, cmap="inferno", alpha=0.4, origin="upper")

    if areas:
        min_area = min(areas)
        max_area = max(areas)
        if max_area - min_area < 1e-9:
            sizes = [120.0] * len(areas)
        else:
            sizes = np.interp(areas, (min_area, max_area), (60.0, 240.0)).tolist()
    else:
        sizes = [120.0] * len(pour_points)

    main_cols = [pp.col for pp in pour_points if pp.id in main_ids]
    main_rows = [pp.row for pp in pour_points if pp.id in main_ids]
    main_sizes = [sizes[idx] for idx, pp in enumerate(pour_points) if pp.id in main_ids]

    trib_cols = [pp.col for pp in pour_points if pp.id not in main_ids]
    trib_rows = [pp.row for pp in pour_points if pp.id not in main_ids]
    trib_sizes = [sizes[idx] for idx, pp in enumerate(pour_points) if pp.id not in main_ids]

    if main_cols:
        plt.scatter(
            main_cols,
            main_rows,
            s=main_sizes,
            c="dodgerblue",
            edgecolors="white",
            linewidths=0.8,
            alpha=0.9,
            label="Main stem",
        )
    if trib_cols:
        plt.scatter(
            trib_cols,
            trib_rows,
            s=trib_sizes,
            c="goldenrod",
            edgecolors="black",
            linewidths=0.8,
            alpha=0.9,
            label="Tributary",
        )

    for pp in pour_points:
        plt.text(pp.col + 2, pp.row + 2, pp.id, color="white", fontsize=8, ha="left", va="bottom")

    plt.title("Generated Pour Points (Main Stem vs. Tributary)")
    plt.axis("off")
    if main_cols or trib_cols:
        plt.legend(loc="upper right")
    plt.tight_layout()
    pour_map_path = step_dir / "pour_points_map.png"
    plt.savefig(pour_map_path, dpi=200)
    plt.close()

    total_area = sum(area for _, _, _, _, _, _, area, _ in table_rows)
    main_area = sum(area for _, _, _, _, _, _, area, role in table_rows if role == "main_stem")
    tributary_area = total_area - main_area

    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 02 – Pour Point Extraction")
    builder.add_paragraph(
        "This step analyses the flow accumulation tree to identify main-stem and "
        "tributary pour points that meet spacing and area-balance requirements."
    )
    builder.add_heading("Inputs", level=2)
    builder.add_list(
        [
            f"Flow accumulation raster: `{context.to_relative(flow_acc_path)}`",
            f"Flow direction raster: `{context.to_relative(flow_dir_path)}`",
            f"DEM: `{context.to_relative(dem_path)}`",
        ]
    )
    builder.add_heading("Parameters", level=2)
    builder.add_list(
        [
            f"Target count: {count}",
            f"Minimum cell spacing: {min_distance}",
            f"Main stem fraction: {main_fraction:.2f}",
            f"Area balance tolerance: {area_tolerance:.2f}",
        ]
    )
    builder.add_heading("Area Summary", level=2)
    builder.add_table(
        TableData(
            headers=["Category", "Estimated Area (km²)"],
            rows=[
                ["Main stem", f"{main_area:.3f}"],
                ["Tributaries", f"{tributary_area:.3f}"],
                ["Total", f"{total_area:.3f}"],
            ],
        )
    )
    builder.add_paragraph(f"Pour points generated at {timestamp.isoformat()}.")
    builder.write(report_path)

    logger.info("Updating configuration with the new pour point layer.")
    delineation_cfg["pour_points_path"] = context.to_relative(geojson_path)
    partition_cfg = context.config.setdefault("partition", {})
    partition_cfg["pour_points_path"] = context.to_relative(geojson_path)
    context.config.setdefault("project", {})["last_step02_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 02 – Pour point extraction completed successfully.")

    outputs: Dict[str, Path] = {
        "geojson": geojson_path,
        "table": table_path,
        "map": pour_map_path,
        "report": report_path,
        "log": log_path,
        "accumulation_overview": accumulation_fig_path,
    }
    return outputs


def _extract_cross_sections(
