"""Step 01: DEM Preprocessing

Derive flow products, statistics, and markdown summary from DEM.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

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
    import richdem as rd
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.colors import LightSource
except ImportError:
    rasterio = None
    rd = None
    matplotlib = None
    plt = None

def run_step01_dem_preprocessing(config_path: Path | str) -> Dict[str, Path]:
    """Generate flow products, statistics, and a markdown summary for the DEM."""

    context = load_project_context(config_path)
    step_name = "dem_processing"
    step_index = 1
    step_dir = step_directory(context, step_index, step_name)
    log_path = context.logs_directory / "step01_dem_processing.log"
    logger = configure_logger("step01", log_path)

    logger.info("Step 01 – DEM preprocessing started.")
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc)

    delineation_cfg = context.config.get("delineation")
    if not isinstance(delineation_cfg, Mapping):
        raise PipelineConfigurationError("Configuration is missing the 'delineation' section.")

    dem_path = delineation_cfg.get("dem_path")
    if not dem_path:
        raise PipelineConfigurationError("The delineation.dem_path entry must be configured.")
    dem_path = resolve_input_path(context, str(dem_path))

    try:
        import numpy as np
        import rasterio
        import richdem as rd

        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LightSource
    except ImportError as exc:  # pragma: no cover - optional heavy deps
        logger.error("Required dependency missing during DEM preprocessing: %s", exc)
        raise

    figures: Dict[str, Path] = {}
    tables: Dict[str, Path] = {}
    additional_outputs: Dict[str, Path] = {}

    logger.info("Loading DEM from %s", dem_path)
    with rasterio.open(dem_path) as dataset:
        dem_data = dataset.read(1, masked=True)
        transform = dataset.transform
        dem_meta = dataset.meta.copy()
        cell_area_m2 = abs(transform.a * transform.e)
        resolution_x, resolution_y = abs(transform.a), abs(transform.e)
        crs_text = dataset.crs.to_string() if dataset.crs else "Unknown"

    dem_array = np.asarray(dem_data.filled(np.nan), dtype=float)
    valid_mask = np.isfinite(dem_array)
    total_valid_cells = int(np.count_nonzero(valid_mask))
    total_area_km2 = float(total_valid_cells * cell_area_m2 / 1_000_000.0)

    logger.info("Computing flow direction, accumulation, and slope grids.")
    rd_dem = rd.rdarray(dem_array, no_data=np.nan)
    rd_dem.geotransform = (
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
    if dem_meta.get("crs"):
        rd_dem.projection = dem_meta["crs"].to_wkt()

    filled = rd.FillDepressions(rd_dem, in_place=False)
    try:
        flow_direction = rd.FlowDirectionD8(filled, in_place=False)
        flow_accum = rd.FlowAccumulation(flow_direction, method="D8")
    except AttributeError:
        logger.warning(
            "richdem.FlowDirectionD8 unavailable; using FlowProportions-based D8 fallback."
        )
        try:
            proportions = rd.FlowProportions(filled, method="D8")
        except AttributeError as exc:  # pragma: no cover - unexpected missing API
            logger.error("richdem.FlowProportions not available: %s", exc)
            raise

        props_array = np.asarray(proportions, dtype=float)
        if props_array.ndim != 3 or props_array.shape[2] not in (8, 9):
            raise RuntimeError(
                f"Unexpected FlowProportions shape {props_array.shape}; expected (rows, cols, 8/9)."
            )

        if props_array.shape[2] == 9:
            direction_codes = np.array([0, 16, 32, 64, 128, 1, 2, 4, 8], dtype=np.uint8)
        else:
            direction_codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)

        positive = np.where(props_array > 0.0, props_array, 0.0)
        max_idx = np.argmax(positive, axis=2)
        max_vals = np.take_along_axis(positive, max_idx[..., None], axis=2)[..., 0]
        flow_dir_array = np.where(max_vals > 0.0, direction_codes[max_idx], 0).astype(np.uint8)

        flow_direction = flow_dir_array
        flow_accum_rd = rd.FlowAccumFromProps(proportions)
        flow_accum = np.asarray(flow_accum_rd, dtype=float)
    else:
        flow_accum = np.asarray(flow_accum, dtype=float)
        flow_direction = np.asarray(flow_direction, dtype=np.uint8)

    slope_radians = rd.TerrainAttribute(filled, attrib="slope_radians")
    slope_degrees = np.degrees(slope_radians, dtype=float)

    logger.info("Writing derived raster products.")
    float_meta = dem_meta.copy()
    float_meta.update({"dtype": "float32", "nodata": math.nan})
    int_meta = dem_meta.copy()
    int_meta.update({"dtype": "int32", "nodata": -1})

    flow_dir_path = step_dir / "flow_direction.tif"
    flow_acc_path = step_dir / "flow_accumulation.tif"
    slope_path = step_dir / "slope_degrees.tif"

    with rasterio.open(flow_dir_path, "w", **int_meta) as dst:
        dst.write(np.asarray(flow_direction, dtype="int32"), 1)

    with rasterio.open(flow_acc_path, "w", **float_meta) as dst:
        dst.write(np.asarray(flow_accum, dtype="float32"), 1)

    with rasterio.open(slope_path, "w", **float_meta) as dst:
        dst.write(np.asarray(slope_degrees, dtype="float32"), 1)

    additional_outputs.update(
        {
            "flow_direction_raster": flow_dir_path,
            "flow_accumulation_raster": flow_acc_path,
            "slope_raster": slope_path,
        }
    )

    logger.info("Generating summary tables.")
    dem_stats = compute_basic_stats(dem_array)
    dem_stats_rows = [
        ("valid_cells", total_valid_cells, ""),
        ("total_area_km2", total_area_km2, "km^2"),
        ("min_elevation", dem_stats["min"], "m"),
        ("max_elevation", dem_stats["max"], "m"),
        ("mean_elevation", dem_stats["mean"], "m"),
        ("std_elevation", dem_stats["std"], "m"),
        ("median_elevation", dem_stats["median"], "m"),
        ("p90_elevation", dem_stats["p90"], "m"),
        ("p99_elevation", dem_stats["p99"], "m"),
        ("resolution_x", resolution_x, "m"),
        ("resolution_y", resolution_y, "m"),
    ]
    dem_summary_path = step_dir / "dem_summary.csv"
    write_csv(dem_summary_path, ("metric", "value", "units"), dem_stats_rows)
    tables["dem_summary"] = dem_summary_path

    flow_stats = compute_basic_stats(np.asarray(flow_accum, dtype=float))
    flow_stat_rows = [
        ("min", flow_stats["min"], ""),
        ("max", flow_stats["max"], ""),
        ("mean", flow_stats["mean"], ""),
        ("median", flow_stats["median"], ""),
        ("p90", flow_stats["p90"], ""),
        ("p99", flow_stats["p99"], ""),
    ]
    flow_stats_path = step_dir / "flow_accumulation_stats.csv"
    write_csv(flow_stats_path, ("statistic", "value", "units"), flow_stat_rows)
    tables["flow_accumulation"] = flow_stats_path

    logger.info("Rendering figures for DEM derivatives.")
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(
        np.where(valid_mask, dem_array, np.nan),
        vert_exag=1.0,
        dx=resolution_x,
        dy=resolution_y,
    )

    plt.figure(figsize=(8, 6))
    plt.imshow(hillshade, cmap="gray", origin="upper")
    plt.title("DEM Hillshade")
    plt.axis("off")
    dem_hillshade_path = step_dir / "dem_hillshade.png"
    plt.tight_layout()
    plt.savefig(dem_hillshade_path, dpi=200)
    plt.close()
    figures["dem_hillshade"] = dem_hillshade_path

    plt.figure(figsize=(8, 6))
    plt.imshow(np.asarray(flow_direction, dtype=float), cmap="viridis", origin="upper")
    plt.title("Flow Direction (D8)")
    plt.colorbar(label="D8 Code")
    plt.tight_layout()
    flow_direction_fig_path = step_dir / "flow_direction.png"
    plt.savefig(flow_direction_fig_path, dpi=200)
    plt.close()
    figures["flow_direction"] = flow_direction_fig_path

    log_accum = np.log1p(np.asarray(flow_accum, dtype=float))
    plt.figure(figsize=(8, 6))
    plt.imshow(log_accum, cmap="inferno", origin="upper")
    plt.title("Log Flow Accumulation")
    plt.colorbar(label="log(1 + accumulation)")
    plt.tight_layout()
    flow_accum_fig_path = step_dir / "flow_accumulation.png"
    plt.savefig(flow_accum_fig_path, dpi=200)
    plt.close()
    figures["flow_accumulation"] = flow_accum_fig_path

    # Simple watershed mask visualisation based on accumulation quantiles.
    finite_log = log_accum[np.isfinite(log_accum)]
    if finite_log.size >= 5:
        quantile_edges = np.quantile(finite_log, [0.2, 0.4, 0.6, 0.8])
        classes = np.digitize(log_accum, quantile_edges, right=True)
    else:
        classes = np.zeros_like(log_accum, dtype=int)
    plt.figure(figsize=(8, 6))
    plt.imshow(classes, cmap="tab20", origin="upper")
    plt.title("Accumulation-based Watershed Mask")
    plt.axis("off")
    plt.tight_layout()
    watershed_mask_path = step_dir / "watershed_masks.png"
    plt.savefig(watershed_mask_path, dpi=200)
    plt.close()
    figures["watershed_masks"] = watershed_mask_path

    logger.info("Composing markdown report for Step 01.")
    report_path = build_report_path(context, step_index, step_name)
    builder = MarkdownReportBuilder("Step 01 – DEM Preprocessing")
    builder.add_paragraph(
        "This step derives primary hydrologic rasters from the project DEM and "
        "computes descriptive statistics that inform downstream delineation thresholds."
    )
    builder.add_heading("Inputs", level=2)
    builder.add_list(
        [
            f"DEM path: `{context.to_relative(dem_path)}`",
            f"Projection: `{crs_text}`",
            f"Cell size: {resolution_x:.2f} m × {resolution_y:.2f} m",
        ]
    )
    builder.add_heading("Key Metrics", level=2)
    builder.add_table(
        TableData(
            headers=["Metric", "Value", "Units"],
            rows=[[name, f"{value:.3f}", units] for name, value, units in dem_stats_rows],
        )
    )
    builder.add_heading("Derived Artefacts", level=2)
    builder.add_list(
        [
            f"Flow direction raster: `{context.to_relative(flow_dir_path)}`",
            f"Flow accumulation raster: `{context.to_relative(flow_acc_path)}`",
            f"Slope raster: `{context.to_relative(slope_path)}`",
        ]
    )
    builder.add_paragraph(f"Report generated at {timestamp.isoformat()}.")
    builder.write(report_path)

    logger.info("Updating configuration with derived raster paths.")
    delineation_cfg = context.config.setdefault("delineation", {})
    delineation_cfg["flow_direction_path"] = context.to_relative(flow_dir_path)
    delineation_cfg["flow_accumulation_path"] = context.to_relative(flow_acc_path)
    delineation_cfg.setdefault(
        "intermediate_directory",
        context.to_relative(context.base_results / "03_partitioning" / "intermediate"),
    )
    context.config.setdefault("project", {})["last_step01_run"] = timestamp.isoformat()

    dump_project_config(context)
    logger.info("Step 01 – DEM preprocessing completed successfully.")

    outputs: Dict[str, Path] = {}
    outputs.update(figures)
    outputs.update(tables)
    outputs.update(additional_outputs)
    outputs["report"] = report_path
    outputs["log"] = log_path
    return outputs


